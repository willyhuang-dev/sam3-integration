#!/usr/bin/env python3
"""Why is INT8-vs-fp16 box agreement 0.930 here and 0.995 in the quantization run?

Three things differ between the two measurements, and a single number cannot
tell them apart, so this measures each one separately with the other two held
fixed:

  fp16_dense vs int8_dense   INT8 alone, no token pruning
  fp16_rect  vs int8_rect    INT8 with pruning  (this is the 0.930 pair)
  fp16_dense vs fp16_rect    pruning alone, no quantization

and it reports every pair in TWO coordinate spaces:

  canvas   the model's own normalised output
  frame    after letterbox un-mapping to source pixels

MEASURED RESULT, kept here because it is worth not re-deriving: the two spaces
give BIT-IDENTICAL IoU, so the "letterbox amplifies y error" hypothesis is dead
-- and it was dead on arrival for a reason I should have seen without measuring:
**IoU is invariant under any per-axis scaling applied to both boxes.** Dividing
every y by 0.5625 scales both boxes' heights, their intersection and their union
by the same factor, so the ratio does not move. The check stays in because it
costs nothing and it pins that invariance.

MATCHING IS DONE ONCE, in canvas space, and the same matched pairs are then
scored in both spaces. Matching per-space would change which boxes are compared
and confound the very thing being measured.

Usage:
    python3 -m src.iou_diag --res 1008 --images N --concepts person ...
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        if all(iou(d["canvas"], k["canvas"]) < thr for k in kept):
            kept.append(d)
    return kept


def detect(eng, rgb, concepts, lib_dir, squash=False):
    """Returns detections carrying BOTH coordinate forms of every box.

    `squash` reproduces the preprocessing the quantization experiment used --
    a plain resize to a square, no letterbox, no black bar. It needs no engine
    rebuild because preprocessing lives outside the engine, which makes it a
    free way to test whether the letterbox input distribution (43% of tokens
    constant black) is part of why INT8 agreement is lower here.
    """
    import text_library as TL

    from src.preprocess import prepare, to_tensor

    tf, tm, slots = TL.assemble(lib_dir, concepts, eng.n_max)
    if squash:
        import cv2
        x = to_tensor(cv2.resize(rgb, (eng.size, eng.size),
                                 interpolation=cv2.INTER_LINEAR))
        content_h = eng.size
    else:
        x, content_h = prepare(rgb, eng.size)
    out = eng.run(x, tf.astype(np.float32), tm.astype(np.int64))

    fh, fw = rgb.shape[:2]
    frac = content_h / eng.size
    boxes, logits = out["pred_boxes"][0], out["pred_logits"][0]
    presence = out["presence_logits"][0]

    res = []
    for slot, label in enumerate(slots):
        if not label:
            continue
        scores = sigmoid(logits[slot]) * sigmoid(presence[slot])
        cand = []
        for q in np.argsort(-scores)[:50]:
            q = int(q)
            if scores[q] < THRESHOLD:
                break
            x0, y0, x1, y1 = (float(v) for v in boxes[slot, q])
            cand.append({
                "label": label, "score": float(scores[q]),
                "canvas": [x0, y0, x1, y1],
                "frame": [x0 * fw, y0 / frac * fh, x1 * fw, y1 / frac * fh],
            })
        res += nms(cand, NMS_IOU)
    return res


def compare(ref, test):
    """Greedy match in CANVAS space, then score the same pairs in both spaces.

    Also keeps, per matched pair, the reference detection's SCORE, canvas AREA
    and LABEL, so the aggregate can be stratified. The quantization experiment
    reported IoU 0.995 on ~54 person boxes from near-duplicate demo scenes and
    said in the same breath that INT8's cost lands on "MARGINAL/low-confidence"
    detections. Stratifying is how that claim gets tested rather than repeated.
    """
    stats = {"n_ref": len(ref), "n_test": len(test), "matched": 0,
             "iou_canvas": [], "iou_frame": [], "score_mae": [],
             "ref_score": [], "ref_area": [], "ref_label": [], "ref_edge": []}
    used = set()
    for a in ref:
        best, biou = None, 0.0
        for j, b in enumerate(test):
            if j in used or b["label"] != a["label"]:
                continue
            v = iou(a["canvas"], b["canvas"])
            if v > biou:
                best, biou = j, v
        if best is not None and biou >= 0.5:
            used.add(best)
            b = test[best]
            stats["matched"] += 1
            stats["iou_canvas"].append(biou)
            stats["iou_frame"].append(iou(a["frame"], b["frame"]))
            stats["score_mae"].append(abs(a["score"] - b["score"]))
            c = a["canvas"]
            stats["ref_score"].append(a["score"])
            stats["ref_area"].append(max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1]))
            stats["ref_label"].append(a["label"])
            stats["ref_edge"].append(
                c[0] <= 0.01 or c[1] <= 0.01 or c[2] >= 0.99 or c[3] >= 0.99)
    return stats


SCORE_BINS = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
AREA_BINS = [(0.0, 0.01, "small <1%"), (0.01, 0.10, "medium 1-10%"),
             (0.10, 1.01, "large >10%")]


def stratify(agg):
    """Print IoU broken down by reference score and by box area."""
    io = np.array(agg["iou_canvas"])
    sc = np.array(agg["ref_score"])
    ar = np.array(agg["ref_area"])
    lb = np.array(agg["ref_label"])
    if io.size == 0:
        return
    print("      依信心分層:", end="", flush=True)
    for lo, hi in SCORE_BINS:
        m = (sc >= lo) & (sc < hi)
        print(f"  [{lo:.1f},{hi if hi <= 1 else 1.0:.1f}) n={m.sum():<4d} "
              f"IoU={io[m].mean():.4f}" if m.any() else
              f"  [{lo:.1f},{hi:.1f}) n=0", end="", flush=True)
    print(flush=True)
    print("      依框大小分層:", end="", flush=True)
    for lo, hi, name in AREA_BINS:
        m = (ar >= lo) & (ar < hi)
        print(f"  {name} n={m.sum():<4d} IoU={io[m].mean():.4f}" if m.any() else
              f"  {name} n=0", end="", flush=True)
    print(flush=True)
    print("      依概念分層:", end="", flush=True)
    for c in sorted(set(lb.tolist())):
        m = lb == c
        print(f"  {c} n={m.sum():<4d} IoU={io[m].mean():.4f}", end="", flush=True)
    print(flush=True)

    # A box the 16:9 centre-crop cut through is unstable under any perturbation:
    # it is pinned to the crop edge rather than to the object, so a tiny change
    # in features can move it a lot. That is an artefact of THIS pipeline's
    # preprocessing, not of quantization, so it is separated out.
    ed = np.array(agg["ref_edge"])
    for name, m in (("觸邊框", ed), ("未觸邊框", ~ed)):
        if m.any():
            print(f"      {name}: n={m.sum():<4d} IoU={io[m].mean():.4f}", flush=True)

    # The closest analogue to the quantization experiment's own sample:
    # person only, high confidence, large boxes, not clipped by the crop.
    m = (lb == "person") & (sc >= 0.9) & (ar >= 0.10) & (~ed)
    print(f"      ⭐ 對照量化實驗的樣本型態（person + 信心≥0.9 + 大框 + 未觸邊）: "
          f"n={m.sum()} " + (f"IoU={io[m].mean():.4f}" if m.any() else "（無樣本）"),
          flush=True)
    for lo, extra in ((0.9, "信心≥0.9"), (0.8, "信心≥0.8"), (0.7, "信心≥0.7")):
        m = (lb == "person") & (sc >= lo) & (ar >= 0.05) & (~ed)
        if m.any():
            print(f"         person + {extra} + 框≥5% + 未觸邊: n={m.sum():<4d} "
                  f"IoU={io[m].mean():.4f}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, default=1008)
    p.add_argument("--engine-dir", default="out/engines")
    # COCO val2017, not the handful of demo photos: the first attempt ran on a
    # single image and 4 detections, which this project has been misled by three
    # times already (20/40-frame previews all reversed on the full set).
    p.add_argument("--image-dir", default="/root/willy/datasets/coco/val2017")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--concepts", nargs="+",
                   default=["person", "bottle", "cup", "chair", "helmet"])
    p.add_argument("--lib-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "concept_library"))
    p.add_argument("--squash", action="store_true",
                   help="plain square resize instead of letterbox -- what the "
                        "quantization experiment fed its engines")
    p.add_argument("--pairs", nargs="+", default=[
        "fp16_dense:int8_dense", "fp16_rect:int8_rect", "fp16_dense:fp16_rect"])
    args = p.parse_args(argv)

    import cv2

    from src.run_engine import Engine

    files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    files = [f for f in files if "result" not in os.path.basename(f)][:args.images]
    if not files:
        raise SystemExit(f"no images under {args.image_dir}")
    print(f"[cfg] res={args.res} images={len(files)} concepts={args.concepts}",
          flush=True)

    tags = sorted({t for pair in args.pairs for t in pair.split(":")})
    dets = {t: [] for t in tags}
    for t in tags:
        path = os.path.join(args.engine_dir, f"{t}_r{args.res}_b1.engine")
        if not os.path.exists(path):
            print(f"[skip] {path} missing", flush=True)
            dets.pop(t)
            continue
        eng = Engine(path)
        for i, f in enumerate(files):
            bgr = cv2.imread(f)
            if bgr is None:
                dets[t].append([])
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets[t].append(detect(eng, rgb, args.concepts, args.lib_dir,
                                  squash=args.squash))
            if (i + 1) % 50 == 0:
                print(f"      {t} {i + 1}/{len(files)}", flush=True)
        n = sum(len(d) for d in dets[t])
        print(f"[run] {t:12s} {n} detections over {len(files)} images", flush=True)
        del eng

    print(f"\n{'pair':28s} {'matched':>9s} {'IoU canvas':>11s} {'IoU frame':>10s} "
          f"{'score MAE':>10s} {'n_test':>8s}", flush=True)
    print("-" * 82, flush=True)
    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        if a not in dets or b not in dets:
            continue
        LISTS = ("iou_canvas", "iou_frame", "score_mae",
                 "ref_score", "ref_area", "ref_label", "ref_edge")
        agg = {"matched": 0, "n_ref": 0, **{k: [] for k in LISTS}}
        for da, db in zip(dets[a], dets[b]):
            s = compare(da, db)
            agg["matched"] += s["matched"]
            agg["n_ref"] += s["n_ref"]
            for k in LISTS:
                agg[k] += s[k]
        mc = np.mean(agg["iou_canvas"]) if agg["iou_canvas"] else float("nan")
        mf = np.mean(agg["iou_frame"]) if agg["iou_frame"] else float("nan")
        ms = np.mean(agg["score_mae"]) if agg["score_mae"] else float("nan")
        n_test = sum(len(d) for d in dets[b])
        print(f"{pair:28s} {agg['matched']:4d}/{agg['n_ref']:<4d} {mc:11.4f} "
              f"{mf:10.4f} {ms:10.4f} {n_test:8d}", flush=True)
        rows.append((pair, mc, mf, ms, agg["matched"], agg["n_ref"], n_test))
        stratify(agg)

    print("\n讀法：", flush=True)
    for pair, mc, mf, ms, m, nr, nt in rows:
        if np.isnan(mc):
            continue
        print(f"  {pair}: IoU {mc:.4f}（canvas 與 frame 差 {mc - mf:+.6f}）"
              f"、配對 {m}/{nr}、對側偵測數 {nt}", flush=True)
    print("\n  canvas 與 frame 恆等是預期的：IoU 對兩個框都套用的逐軸縮放不變。",
          flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
