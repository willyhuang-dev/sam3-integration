#!/usr/bin/env python3
"""Cosine similarity of the INT8 VE's features against fp16 -- the apples-to-apples
number against the quantization experiment's published 0.869-0.879.

Detection-level IoU compares two pipelines through thresholding, NMS and greedy
matching, all of which depend on what is in the picture. Feature cosine does
not: it is a direct read of how much the quantization moved the vision encoder's
output, on whatever images you feed it. So it is the right instrument for
"is MY INT8 VE as good as THEIRS", where the IoU comparison is confounded by
the two experiments having measured completely different image sets.

Builds VE-only engines (the merged ones do not expose fpn_feat_2) and compares
`fpn_feat_2`, which is the tensor their number was reported on.

Usage:
    python3 -m src.ve_fidelity --res 1008 --variant dense --images 16
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRTEXEC = os.environ.get("TRTEXEC", "trtexec")


def build(onnx, engine, res, int8):
    if os.path.exists(engine):
        print(f"[build] {engine} exists, skip", flush=True)
        return True
    shape = f"images:1x3x{res}x{res}"
    cmd = [TRTEXEC, f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16",
           f"--minShapes={shape}", f"--optShapes={shape}", f"--maxShapes={shape}",
           "--skipInference"]
    if int8:
        cmd.insert(3, "--int8")
    print(f"[build] {os.path.basename(engine)} ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(engine)
    print(f"[build] {'ok' if ok else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
    if not ok:
        print(((r.stdout or "") + (r.stderr or ""))[-800:], flush=True)
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, default=1008)
    p.add_argument("--variant", default="dense", choices=["dense", "rect"])
    p.add_argument("--images", type=int, default=16)
    p.add_argument("--image-dir", default="/root/willy/datasets/coco/val2017")
    p.add_argument("--squash", action="store_true",
                   help="feed squash-resized frames at INFERENCE time")
    p.add_argument("--int8-onnx", default=None,
                   help="override the INT8 VE graph, e.g. a squash-CALIBRATED one")
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", default="out/engines")
    args = p.parse_args(argv)

    src_dir = f"out/res={args.res}" + ("_dense" if args.variant == "dense" else "")
    fp16_onnx = os.path.join(src_dir, "ve.onnx")
    int8_onnx = args.int8_onnx or f"out/ve_int8_{args.variant}_{args.res}.onnx"
    fp16_eng = os.path.join(args.out_dir, f"veonly_fp16_{args.variant}_{args.res}.engine")
    int8_eng = os.path.join(
        args.out_dir, f"veonly_int8_{args.variant}_{args.res}{args.tag}.engine")
    for o in (fp16_onnx, int8_onnx):
        if not os.path.exists(o):
            raise SystemExit(f"missing {o}")
    if not build(fp16_onnx, fp16_eng, args.res, False):
        raise SystemExit("fp16 VE build failed")
    if not build(int8_onnx, int8_eng, args.res, True):
        raise SystemExit("int8 VE build failed")

    import cv2

    from src.preprocess import prepare, to_tensor
    from src.run_engine import Engine

    files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))[:args.images]
    a, b = Engine(fp16_eng), Engine(int8_eng)

    def feats(eng, x):
        n = eng.key("images")
        host, dev = eng.buf[n]
        np.copyto(host, x.astype(eng.dtypes[n]).ravel())
        eng.cuda.memcpy_htod_async(dev, host, eng.stream)
        eng.ctx.execute_async_v3(eng.stream.handle)
        out = {}
        for name in ("fpn_feat_2", "fpn_pos_2"):
            k = eng.key(name)
            h, d = eng.buf[k]
            eng.cuda.memcpy_dtoh_async(h, d, eng.stream)
            out[name] = h
        eng.stream.synchronize()
        return {k: v.reshape(eng.shapes[eng.key(k)]).copy() for k, v in out.items()}

    cos_feat, cos_pos, rel = [], [], []
    for f in files:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if args.squash:
            x = to_tensor(cv2.resize(rgb, (args.res, args.res)))
        else:
            x, _ = prepare(rgb, args.res)
        fa, fb = feats(a, x), feats(b, x)
        for key, acc in (("fpn_feat_2", cos_feat), ("fpn_pos_2", cos_pos)):
            u = fa[key].astype(np.float64).ravel()
            v = fb[key].astype(np.float64).ravel()
            acc.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
        u = fa["fpn_feat_2"].astype(np.float64).ravel()
        v = fb["fpn_feat_2"].astype(np.float64).ravel()
        rel.append(float(np.linalg.norm(u - v) / (np.linalg.norm(u) + 1e-12)))

    print(f"\nVE fidelity  res={args.res} variant={args.variant} "
          f"n={len(cos_feat)} images  "
          f"({'squash' if args.squash else 'letterbox'})", flush=True)
    print(f"  fpn_feat_2  cos = {np.mean(cos_feat):.4f}  "
          f"(min {np.min(cos_feat):.4f})   rel_L2 = {np.mean(rel):.4f}", flush=True)
    # fpn_pos_2 is a sine position embedding: NOT quantized and deterministic, so
    # cos must be 1.0. If it is not, the two engines are not fed the same input
    # and the fpn_feat_2 number means nothing.
    print(f"  fpn_pos_2   cos = {np.mean(cos_pos):.6f}  "
          f"(必須是 1.0 -- 非量化路徑，否則兩顆 engine 吃到的輸入不同)", flush=True)
    print(f"\n  對照 sam3_quantization 實驗公布的 VE@1008 transplant: cos 0.869 / "
          f"proper-calib @644: 0.879", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
