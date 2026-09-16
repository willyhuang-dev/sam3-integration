#!/usr/bin/env python3
"""INT8 (W8A8) PTQ of the VE graph, mixed precision: FPN neck kept fp16.

Config is not a guess -- it is the end of a ladder the quantization experiment
climbed and then stopped climbing:

  * FP8 does NOT accelerate on sm_120 (FP32-accumulate throttle): 1.05x.
    INT8 does: 1.49x. NVFP4 compiles and lands on INT8's number. So INT8.
  * `disable_mha_qdq=True` -- ViT-H attention outliers break PTQ, and the
    attention core is only ~7% of VE time anyway. Quantising it buys ~0%.
  * `calibration_method="max"` -- entropy was measured to be identical
    (cos 0.875 either way) and takes 50% longer.
  * SmoothQuant and more calibration data were both measured NOT to help: the
    ViT-H error is broadly-distributed 8-bit rounding, not the concentrated
    per-channel outliers those techniques migrate.
  * `nodes_to_exclude` on the neck convs IS what helped, and it is free: they
    produce fpn_feat_2 directly, so they are disproportionately sensitive.
    shopee mAP retention went 97.3% -> 99.4% at identical speed.
  * `calibration_eps=["cpu"]` dodges the sm_120 cuDNN segfault in the ORT
    calibration session.

⚠️ `nodes_to_exclude` patterns are matched with `re.match` -- ANCHORED AT THE
START of the node name. The reference recipe used `/neck/.*`, which matches
nothing here because this export nests the model one level deeper
(`/vision_encoder/neck/...`). That failure is SILENT: quantization succeeds,
the file is written, and you get plain full-INT8 wearing an "mp_neck" label.
`_assert_neck_untouched` is the gate; do not remove it.

Usage:
    python3 -m src.quantize --ve out/res=644/ve.onnx --out out/ve_644.mp_neck.onnx
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

NECK_PATTERN = ".*/neck/.*"


def build_calibration(files, size, limit, squash=False):
    """SAM3-normalised calibration frames, letterboxed by default.

    Letterbox is the deployed distribution, which is the principled choice: with
    strategy C every ViT layer sees content-only tokens and the patch embedding
    sees a black bar that squashing would not have.

    `squash` exists because the principle lost to a measurement. The deployed
    (letterbox-calibrated) VE scores cos 0.8515 against fp16, while the
    quantization experiment published 0.869 for the same recipe with
    squash-resized calibration -- so the calibration distribution is a real
    variable here and the two are worth comparing rather than assumed.
    """
    import cv2

    from .preprocess import prepare, to_tensor

    out = []
    for f in files[:limit]:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if squash:
            x = to_tensor(cv2.resize(rgb, (size, size),
                                     interpolation=cv2.INTER_LINEAR))
        else:
            x, _ = prepare(rgb, size)
        out.append(x.astype(np.float32))
    if not out:
        raise SystemExit("no calibration images could be read")
    return out


def _assert_neck_untouched(path):
    """The gate for the re.match anchoring trap.

    Two independent claims, because either alone can be satisfied by a broken
    run: SOMETHING was quantized, and NOTHING in the neck was.
    """
    import onnx
    m = onnx.load(path, load_external_data=False)
    q = [n for n in m.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    neck_q = [n for n in q if re.search("/neck/", n.name)]
    print(f"    Q/DQ nodes: {len(q)}   of which in the neck: {len(neck_q)}", flush=True)
    if not q:
        raise SystemExit("FAIL: nothing was quantized at all")
    if neck_q:
        raise SystemExit(
            f"FAIL: {len(neck_q)} Q/DQ nodes landed in the neck -- the exclusion "
            f"pattern did not match. This is the re.match anchoring trap; the "
            f"file would be plain INT8 mislabelled as mp_neck. "
            f"Examples: {[n.name for n in neck_q[:3]]}")
    return len(q)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ve", required=True, help="fp32 VE ONNX to quantize")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, required=True,
                   help="resolution of --ve; calibration frames are built at it")
    p.add_argument("--images", default="/root/willy/models/pretrained_weights/"
                                        "sam3_huggingface/images")
    p.add_argument("--n-calib", type=int, default=8)
    p.add_argument("--squash-calib", action="store_true",
                   help="calibrate on plain square-resized frames instead of "
                        "letterboxed ones. Deployment letterboxes, so this is "
                        "the WRONG distribution on principle -- it is here "
                        "because the measured VE cosine (0.8515) came out below "
                        "the quantization experiment's 0.869, and its calibration "
                        "was squash-resized. Principle loses to measurement.")
    p.add_argument("--full-int8", action="store_true",
                   help="ablation: quantize the neck too (measured WORSE mAP at "
                        "identical speed -- kept so the claim stays falsifiable)")
    args = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True))
    imgs = build_calibration(files, args.size, args.n_calib, args.squash_calib)
    print(f"[calib] {len(imgs)} "
          f"{'squash-resized' if args.squash_calib else 'letterboxed'} frames at "
          f"{imgs[0].shape}", flush=True)

    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_first(self):
            # Not part of the CalibrationDataReader ABC, but modelopt's
            # GEMV-pattern pass calls it before calibration starts.
            return {"images": imgs[0]}

        def get_next(self):
            if self.i >= len(imgs):
                return None
            d = {"images": imgs[self.i]}
            self.i += 1
            return d

        def rewind(self):
            self.i = 0

    import modelopt.onnx.quantization as moq

    exclude = [] if args.full_int8 else [NECK_PATTERN]
    print(f"[quant] int8 W8A8, exclude={exclude}", flush=True)
    t0 = time.time()
    moq.quantize(
        onnx_path=args.ve, calibration_data_reader=Reader(), quantize_mode="int8",
        calibration_method="max", calibration_eps=["cpu"], high_precision_dtype="fp32",
        disable_mha_qdq=True, mha_accumulation_dtype="fp32",
        nodes_to_exclude=exclude, use_external_data_format=True,
        output_path=args.out)
    print(f"[quant] done in {time.time() - t0:.0f}s -> {args.out} "
          f"({os.path.getsize(args.out) / 2**20:.0f} MiB)", flush=True)

    if args.full_int8:
        import onnx
        m = onnx.load(args.out, load_external_data=False)
        n = sum(1 for x in m.graph.node
                if x.op_type in ("QuantizeLinear", "DequantizeLinear"))
        print(f"    Q/DQ nodes: {n} (neck NOT excluded -- ablation)", flush=True)
    else:
        _assert_neck_untouched(args.out)
    sys.stdout.flush()
    os._exit(0)   # ORT/modelopt teardown likes to SIGABRT; the file is written


if __name__ == "__main__":
    main()
