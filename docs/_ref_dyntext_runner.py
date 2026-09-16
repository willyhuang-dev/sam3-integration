#!/usr/bin/env python3
"""Smallest complete TensorRT runner for the exported graph.

Not production code -- it is the reference that shows the three things that are
easy to get wrong, and that cost you either correctness or an order of magnitude
of throughput:

  1. building the text tensors and zeroing unused slots
  2. fetching ONLY the masks that survived, never the whole pred_masks tensor
  3. mapping mask-space coordinates back to frame pixels

    python3 minimal_runner.py model.engine frame.jpg person "cardboard box"
    python3 minimal_runner.py model.engine frame.jpg --save out.jpg person hand

With --save it writes a mask overlay so you can actually look at what the model
found, rather than reading coordinates.

Requires: tensorrt, pycuda, opencv-python, numpy.
"""
import sys
import os

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 -- creates the CUDA context

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import text_library as TL  # noqa: E402

QUERY = 200
THRESHOLD = 0.5
NMS_IOU = 0.85
LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concept_library")

# Distinguishable at a glance and stable per slot, so the same concept keeps its
# colour across frames.
PALETTE = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67),
           (49, 130, 245), (180, 30, 145), (244, 212, 66), (230, 50, 240),
           (0, 252, 124), (36, 99, 154)]


def draw_overlay(frame, results, out_path):
    """Mask fill + outline + label. Deliberately simple: the point is to SEE the
    masks, not to be pretty."""
    canvas = frame.copy()
    fill = frame.copy()
    for r in results:
        colour = PALETTE[r["slot"] % len(PALETTE)]
        polys = [np.array(c, np.int32) for c in r["contours"] if len(c) >= 3]
        if polys:
            cv2.fillPoly(fill, polys, colour)
            cv2.polylines(canvas, polys, True, colour, 2, cv2.LINE_AA)
    canvas = cv2.addWeighted(fill, 0.35, canvas, 0.65, 0)
    for r in results:
        colour = PALETTE[r["slot"] % len(PALETTE)]
        x0, y0, _, _ = r["box"]
        tag = "%s %.2f" % (r["label"], r["score"])
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(th + 6, y0)
        cv2.rectangle(canvas, (x0, y - th - 6), (x0 + tw + 8, y + 4), colour, -1)
        cv2.putText(canvas, tag, (x0 + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    print(f"overlay -> {out_path}")


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def nms(dets, thr):
    kept = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["box"], k["box"]) < thr for k in kept):
            kept.append(d)
    return kept


def main(engine_path, image_path, concepts, save_path=None):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    shapes = {n: tuple(int(d) for d in engine.get_tensor_shape(n)) for n in names}
    size = shapes["images"][-1]
    n_max = shapes["text_features"][0]
    mask_h, mask_w = shapes["pred_masks"][-2:]
    print(f"engine: size={size} n_max={n_max} mask={mask_h}x{mask_w}")

    if len(concepts) > n_max:
        raise SystemExit(f"{len(concepts)} concepts exceeds engine n_max={n_max}")

    # --- allocate. pred_masks gets a device buffer but NO full host buffer:
    # at 1008 it is 663 MB, and copying it every frame is what kills throughput.
    buf = {}
    for n in names:
        dt = trt.nptype(engine.get_tensor_dtype(n))
        vol = int(trt.volume(shapes[n]))
        host = None if n == "pred_masks" else cuda.pagelocked_empty(vol, dt)
        dev = cuda.mem_alloc(vol * np.dtype(dt).itemsize)
        buf[n] = (host, dev, dt)
        ctx.set_tensor_address(n, int(dev))
    mask_scratch = cuda.pagelocked_empty(mask_h * mask_w, np.float32)
    stream = cuda.Stream()

    # --- text: library vectors, unused slots left as zeros
    tf, tm, slots = TL.assemble(LIB, concepts, n_max)
    for name, arr in (("text_features", tf), ("text_mask", tm)):
        host, dev, dt = buf[name]
        np.copyto(host, arr.astype(dt).ravel())
        cuda.memcpy_htod(dev, host)

    # --- image: plain squash-resize, NO letterbox (the coordinate math below
    # depends on this), then RGB and x/127.5-1
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"cannot read {image_path}")
    fh, fw = frame.shape[:2]
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

    host, dev, _ = buf["images"]
    np.copyto(host, blob.ravel())
    cuda.memcpy_htod_async(dev, host, stream)
    ctx.execute_async_v3(stream.handle)

    # Only the SMALL outputs come back. pred_masks stays on the device.
    for n in ("pred_boxes", "pred_logits"):
        host, dev, _ = buf[n]
        cuda.memcpy_dtoh_async(host, dev, stream)
    stream.synchronize()

    boxes = buf["pred_boxes"][0].reshape(n_max, QUERY, 4)
    logits = buf["pred_logits"][0].reshape(n_max, QUERY)

    _, masks_dev, _ = buf["pred_masks"]
    results = []
    for slot, label in enumerate(slots):
        if not label:
            continue                                    # zeroed slot
        scores = 1.0 / (1.0 + np.exp(-logits[slot]))    # sigmoid for SCORES
        cand = []
        for q in np.argsort(-scores)[:50]:
            q = int(q)
            if scores[q] < THRESHOLD:
                break
            b = boxes[slot, q]
            x0, y0, x1, y1 = b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh
            if x1 > x0 and y1 > y0:
                cand.append({"score": float(scores[q]), "box": [x0, y0, x1, y1], "q": q})

        for d in nms(cand, NMS_IOU):
            # Selective copy: ONE (slot, query) mask block, not the full tensor.
            offset = ((slot * QUERY) + d["q"]) * mask_h * mask_w * 4
            cuda.memcpy_dtoh(mask_scratch, int(masks_dev) + offset)
            logit = mask_scratch.reshape(mask_h, mask_w)

            # sigmoid(x) >= 0.5  <=>  x >= 0. Identical result, no exp() at all.
            mask_bool = (logit >= 0.0).astype(np.uint8)

            cnts, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            sx, sy = fw / mask_w, fh / mask_h
            # min-area is in LOW-RES px^2; scale it or small real detections vanish
            min_area = max(1.0, 50.0 / (sx * sy))
            contours = [
                [[int(round(p[0][0] * sx)), int(round(p[0][1] * sy))] for p in c]
                for c in cnts if cv2.contourArea(c) >= min_area
            ]
            results.append({"slot": slot, "label": label,
                            "score": round(d["score"], 3),
                            "box": [int(v) for v in d["box"]], "contours": contours})

    for r in results:
        pts = sum(len(c) for c in r["contours"])
        print(f"  {r['label']:<22} {r['score']:.3f}  box={r['box']}  "
              f"{len(r['contours'])} contour(s), {pts} pts")
    print(f"{len(results)} detection(s)")
    if save_path:
        draw_overlay(frame, results, save_path)


if __name__ == "__main__":
    argv = sys.argv[1:]
    save = None
    if "--save" in argv:
        i = argv.index("--save")
        save = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) < 3:
        raise SystemExit(__doc__)
    main(argv[0], argv[1], argv[2:], save)
