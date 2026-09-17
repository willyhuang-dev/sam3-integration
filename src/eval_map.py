#!/usr/bin/env python3
"""COCO box mAP on real annotations -- the measurement that says "more accurate",
which engine-vs-engine agreement cannot.

THE PROTOCOL: stretch, do not crop
----------------------------------
Strategy C needs the content in the top rows and the bar at the bottom. The
obvious way to get that is a 16:9 centre-crop, and the first version of this
script did exactly that -- which threw away 12.9% of COCO's boxes and charged
the model for objects it was never shown. It scored mAP 0.251 and the number
was not trustworthy.

Instead the whole frame is resized into the content band (aspect distorted, no
crop). Nothing is lost, and the inverse map is exact: x spans the full original
width, y the full original height. **So the ground truth is used untouched and
this is a standard COCO evaluation.**

For the actual product input (1920x1080) the two are the SAME operation -- a
16:9 source is not cropped by a 16:9 crop. The distinction exists only for
evaluating on datasets that are not 16:9.

Aspect distortion is close to free: the ROI experiment measured
aspect-preserving letterbox vs stretched-to-square at -0.2% det50.

Classes: only the COCO categories the concept library actually has a vector for.
Using a concept the library lacks would measure the text tower, not the engine.

Usage:
    python3 -m src.eval_map --engine out/engines/int8_rect_r1008_b1.engine \
        --res 1008 --images 500 --tag int8_rect@1008
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# concept-library name -> COCO category name
CONCEPTS = {"person": "person", "bottle": "bottle", "cup": "cup", "chair": "chair"}
# Low on purpose: mAP integrates the precision-recall curve, so cutting at 0.5
# would truncate the curve and flatter the model.
THRESHOLD = 0.05


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
        if all(iou(d["xyxy"], k["xyxy"]) < thr for k in kept):
            kept.append(d)
    return kept


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--images", type=int, default=500)
    p.add_argument("--coco", default="/root/willy/datasets/coco")
    p.add_argument("--lib-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "concept_library"))
    p.add_argument("--nms", type=float, default=0.6,
                   help="NMS IoU. The engine-agreement script uses 0.85 because "
                        "there it only has to avoid double-counting; for mAP a "
                        "loose NMS leaves duplicates that count as false "
                        "positives and depress precision.")
    p.add_argument("--square", action="store_true",
                   help="CONTROL: plain square resize, no letterbox, no bar -- "
                        "stock SAM3 preprocessing, where the inverse map is just "
                        "x*W, y*H. Only valid on a dense engine (a pruned one "
                        "would cut the bottom off the picture). If this scores "
                        "much higher than the letterbox path, the bug is in the "
                        "letterbox handling and not in the model.")
    p.add_argument("--max-det", type=int, default=100,
                   help="per image, across all concepts -- COCO's maxDets")
    p.add_argument("--out", default="results_map.jsonl")
    args = p.parse_args(argv)

    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    import text_library as TL

    from src.preprocess import prepare
    from src.run_engine import Engine

    coco = COCO(os.path.join(args.coco, "annotations", "instances_val2017.json"))
    name2cat = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    cat_ids = [name2cat[v] for v in CONCEPTS.values()]
    concepts = list(CONCEPTS.keys())
    # Only images that actually contain one of these categories: including
    # images with none of them adds no signal to AP and dilutes nothing, but it
    # does waste inference time.
    img_ids = sorted({i for c in cat_ids for i in coco.getImgIds(catIds=[c])})[:args.images]
    print(f"[cfg] {args.tag}  res={args.res}  images={len(img_ids)}  "
          f"concepts={concepts}  nms={args.nms}", flush=True)

    eng = Engine(args.engine)
    tf, tm, slots = TL.assemble(args.lib_dir, concepts, eng.n_max)

    results = []
    for i, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(os.path.join(args.coco, "val2017", info["file_name"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        if args.square:
            from src.preprocess import to_tensor
            x = to_tensor(cv2.resize(rgb, (eng.size, eng.size),
                                     interpolation=cv2.INTER_LINEAR))
            frac = 1.0
        else:
            x, content_h = prepare(rgb, eng.size, stretch=True)
            frac = content_h / eng.size
        out = eng.run(x, tf.astype(np.float32), tm.astype(np.int64))
        boxes, logits = out["pred_boxes"][0], out["pred_logits"][0]
        presence = out["presence_logits"][0]

        per_img = []
        for slot, label in enumerate(slots):
            if not label:
                continue
            scores = sigmoid(logits[slot]) * sigmoid(presence[slot])
            cand = []
            for q in np.argsort(-scores)[:100]:
                q = int(q)
                if scores[q] < THRESHOLD:
                    break
                bx0, by0, bx1, by1 = (float(v) for v in boxes[slot, q])
                # canvas-normalised -> ORIGINAL image pixels. Exact, because the
                # whole frame was stretched into the content band.
                cand.append({"score": float(scores[q]),
                             "xyxy": [bx0 * W, by0 / frac * H,
                                      bx1 * W, by1 / frac * H],
                             "category_id": name2cat[CONCEPTS[label]],
                             "image_id": iid})
            per_img += nms(cand, args.nms)
        per_img.sort(key=lambda d: -d["score"])
        results += per_img[:args.max_det]
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(img_ids)}  ({len(results)} dets)", flush=True)

    if not results:
        raise SystemExit("no detections at all")
    coco_dt = [{"image_id": d["image_id"], "category_id": d["category_id"],
                "bbox": [d["xyxy"][0], d["xyxy"][1],
                         d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1]],
                "score": d["score"]} for d in results]

    ev = COCOeval(coco, coco.loadRes(coco_dt), "bbox")
    ev.params.imgIds = img_ids
    ev.params.catIds = cat_ids          # score only the concepts we can prompt
    ev.evaluate(); ev.accumulate(); ev.summarize()

    row = {"tag": args.tag, "res": args.res, "images": len(img_ids),
           "n_det": len(results), "nms": args.nms,
           "mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
           "AP75": float(ev.stats[2]), "AP_small": float(ev.stats[3]),
           "AP_medium": float(ev.stats[4]), "AP_large": float(ev.stats[5]),
           "AR100": float(ev.stats[8])}
    per = {}
    for ci, cid in enumerate(cat_ids):
        pr = ev.eval["precision"][:, :, ci, 0, 2]
        pr = pr[pr > -1]
        per[coco.loadCats([cid])[0]["name"]] = float(np.mean(pr)) if pr.size else float("nan")
    row["per_class"] = per
    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\n{args.tag}: mAP={row['mAP']:.4f} AP50={row['AP50']:.4f} "
          f"small={row['AP_small']:.4f} large={row['AP_large']:.4f}", flush=True)
    print(f"  per class: { {k: round(v, 4) for k, v in per.items()} }", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
