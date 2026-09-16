#!/usr/bin/env python3
"""Run the integrated engine on a real frame -- the check speed cannot make.

A fast engine that detects nothing is still a failed integration, and the three
things being combined here each have a way of producing exactly that:

  * strategy C prunes rows -- if the preprocessing does not put the black bar
    where the pruning assumes, the content is cropped away and detections
    vanish silently.
  * INT8 -- a mis-transplanted scale set builds and runs, and returns noise.
  * multi-concept -- a swapped repeat/tile pairs image b with the wrong
    concept; every score stays plausible.

So this prints per-concept detections, and `--compare` scores one engine
against another on the same frame so INT8 can be checked against fp16 directly.

Coordinate mapping is the part that differs from the dyntext runner. Boxes come
back normalised to the SQUARE canvas, but the content only occupies the top
`content_h` rows of it, so y must be divided by the content fraction before
scaling to the frame. Getting this wrong squashes every box towards the top of
the image -- which looks like a model problem, not a coordinate problem.

Usage:
    python3 -m src.run_engine --engine e.engine --image f.jpg person hand
    python3 -m src.run_engine --engine int8.engine --compare fp16.engine \
        --image f.jpg person "cardboard box"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERY = 200
THRESHOLD = 0.5
NMS_IOU = 0.85


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


class Engine:
    """Persistent buffers, deliberately.

    pycuda's per-call mem_alloc/free crashes the big Myelin-fused SAM3 graph
    with `Error 709 destroying event` / SIGABRT. Allocate once, reuse, never
    free -- the process exits with os._exit anyway.
    """

    def __init__(self, path):
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
        self.cuda = cuda
        with open(path, "rb") as f:
            self.engine = trt.Runtime(
                trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        self.shapes = {n: tuple(int(d) for d in self.engine.get_tensor_shape(n))
                       for n in self.names}
        self.dtypes = {n: trt.nptype(self.engine.get_tensor_dtype(n))
                       for n in self.names}
        self.buf = {}
        for n in self.names:
            vol = int(np.prod(self.shapes[n]))
            host = None if n.endswith("pred_masks") else \
                cuda.pagelocked_empty(vol, self.dtypes[n])
            dev = cuda.mem_alloc(vol * np.dtype(self.dtypes[n]).itemsize)
            self.buf[n] = (host, dev)
            self.ctx.set_tensor_address(n, int(dev))
        self.stream = cuda.Stream()

    def key(self, suffix):
        for n in self.names:
            if n == suffix or n.endswith("/" + suffix):
                return n
        raise KeyError(f"{suffix} not among {self.names}")

    @property
    def size(self):
        return self.shapes[self.key("images")][-1]

    @property
    def n_max(self):
        return self.shapes[self.key("text_features")][0]

    def run(self, images, text_features, text_mask):
        for name, arr in (("images", images), ("text_features", text_features),
                          ("text_mask", text_mask)):
            n = self.key(name)
            host, dev = self.buf[n]
            np.copyto(host, arr.astype(self.dtypes[n]).ravel())
            self.cuda.memcpy_htod_async(dev, host, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        out = {}
        for name in ("pred_boxes", "pred_logits", "presence_logits"):
            n = self.key(name)
            host, dev = self.buf[n]
            self.cuda.memcpy_dtoh_async(host, dev, self.stream)
            out[name] = host
        self.stream.synchronize()
        for name in out:
            out[name] = out[name].reshape(self.shapes[self.key(name)]).copy()
        return out


def detect(eng, frame_rgb, concepts, lib_dir):
    import text_library as TL

    from src.preprocess import prepare

    tf, tm, slots = TL.assemble(lib_dir, concepts, eng.n_max)
    x, content_h = prepare(frame_rgb, eng.size)
    out = eng.run(x, tf.astype(np.float32), tm.astype(np.int64))

    fh, fw = frame_rgb.shape[:2]
    frac = content_h / eng.size          # the content's share of the canvas
    boxes = out["pred_boxes"][0]
    logits = out["pred_logits"][0]
    presence = out["presence_logits"][0]

    results = []
    for slot, label in enumerate(slots):
        if not label:
            continue
        # The verified SAM3 confidence, not sigmoid(logits) alone: presence is a
        # per-(image, concept) gate and dropping it inflates scores on concepts
        # that are simply absent.
        scores = sigmoid(logits[slot]) * sigmoid(presence[slot])
        cand = []
        for q in np.argsort(-scores)[:50]:
            q = int(q)
            if scores[q] < THRESHOLD:
                break
            x0, y0, x1, y1 = boxes[slot, q]
            # canvas-normalised -> content-normalised -> frame pixels
            cand.append({"score": float(scores[q]),
                         "box": [x0 * fw, y0 / frac * fh, x1 * fw, y1 / frac * fh],
                         "q": q, "slot": slot, "label": label})
        results += nms(cand, NMS_IOU)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--compare", default=None,
                   help="second engine to score against the first on the same frame")
    p.add_argument("--image", required=True)
    p.add_argument("--lib-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "concept_library"))
    p.add_argument("concepts", nargs="+")
    args = p.parse_args(argv)

    import cv2
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    eng = Engine(args.engine)
    print(f"engine: size={eng.size} n_max={eng.n_max}", flush=True)
    a = detect(eng, rgb, args.concepts, args.lib_dir)
    for r in sorted(a, key=lambda r: (r["slot"], -r["score"])):
        b = [int(v) for v in r["box"]]
        print(f"  {r['label']:<22} {r['score']:.3f}  box={b}", flush=True)
    print(f"{len(a)} detection(s)", flush=True)

    if args.compare:
        eng2 = Engine(args.compare)
        b = detect(eng2, rgb, args.concepts, args.lib_dir)
        print(f"\n[compare] {os.path.basename(args.engine)} -> {len(a)} det, "
              f"{os.path.basename(args.compare)} -> {len(b)} det", flush=True)
        matched, ious, dscore = 0, [], []
        for da in a:
            best, biou = None, 0.0
            for db in b:
                if db["label"] != da["label"]:
                    continue
                v = iou(da["box"], db["box"])
                if v > biou:
                    best, biou = db, v
            if best is not None and biou >= 0.5:
                matched += 1
                ious.append(biou)
                dscore.append(abs(da["score"] - best["score"]))
        print(f"[compare] matched {matched}/{len(a)}"
              f"  mean IoU {np.mean(ious) if ious else float('nan'):.3f}"
              f"  score MAE {np.mean(dscore) if dscore else float('nan'):.3f}",
              flush=True)

    sys.stdout.flush()
    os._exit(0)   # pycuda + TRT teardown SIGABRTs and dumps a multi-GB core


if __name__ == "__main__":
    main()
