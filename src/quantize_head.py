#!/usr/bin/env python3
"""INT8 the head (DETR encoder/decoder + scoring + mask decoder).

Why this is worth doing now when the quantization experiment said it was not:
that verdict ("DETR decoder INT8 = 1.10x, and it is only 7% of the pipeline, so
skip it") was measured at ONE concept. This integration runs ten, and the
N-sweep showed the head is 69-77% of the engine at N=10 and costs 7.62 ms per
concept in DETR alone. The same 1.10x now applies to roughly half the total.

Two things differ from the VE's recipe, and both matter:

  * `disable_mha_qdq=False`. The ViT's attention is RoPE-based, so ModelOpt
    reports "Found 0 MHA patterns" and there is nothing to quantize there. The
    DETR attention is standard MHA and IS detected -- this is the one place in
    SAM3 where quantizing attention is even possible.
  * The calibration inputs are FPN feature maps, not images, so they have to be
    produced by actually running the VE (staged calibration). Feeding random
    tensors would set activation ranges that no real image produces.

Memory: the head at N=10 and res 1008 emits pred_masks [10,200,288,288] = 663 MB,
and ModelOpt marks hundreds of activations as graph outputs during calibration.
The cheap route would be to calibrate the N=1 head and transplant, as is done
across resolutions -- but that DOES NOT WORK here and the gate caught it: the
two heads are exported separately, so their auto-generated tensor names
(`onnx::MatMul_7117`...) do not line up, and 108 of the quantized tensors simply
do not exist in the other graph. Structure being identical is not enough; the
resolution transplant works because those graphs come from the same exporter
path with the same node ordering. So each N is calibrated directly.

Usage:
    python3 -m src.quantize_head --res 1008 --calib-n 8
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

HEAD_IN = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2",
           "text_features", "text_mask"]


def collect(res, n_calib, image_dir, lib_dir, concepts, ve_engine, n_max):
    """Run the real VE on real frames to get the head's real input distribution."""
    import cv2
    import text_library as TL

    from src.preprocess import prepare
    from src.run_engine import Engine

    eng = Engine(ve_engine)
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:n_calib]
    tf, tm, _ = TL.assemble(lib_dir, concepts[:n_max], n_max)

    samples = []
    for f in files:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        x, _ = prepare(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), res)
        n = eng.key("images")
        host, dev = eng.buf[n]
        np.copyto(host, x.astype(eng.dtypes[n]).ravel())
        eng.cuda.memcpy_htod_async(dev, host, eng.stream)
        eng.ctx.execute_async_v3(eng.stream.handle)
        d = {}
        for name in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"):
            k = eng.key(name)
            h, dv = eng.buf[k]
            eng.cuda.memcpy_dtoh_async(h, dv, eng.stream)
            d[name] = (h, k)
        eng.stream.synchronize()
        s = {nm: h.reshape(eng.shapes[k]).astype(np.float32).copy()
             for nm, (h, k) in d.items()}
        s["text_features"] = tf.astype(np.float32)
        s["text_mask"] = tm.astype(np.int64)
        samples.append(s)
        print(f"    calib {len(samples)}/{len(files)}", flush=True)
    del eng
    return samples


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, default=1008)
    p.add_argument("--calib-n", type=int, default=8)
    p.add_argument("--image-dir", default="/root/willy/datasets/coco/val2017")
    p.add_argument("--lib-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "concept_library"))
    p.add_argument("--n-max", type=int, default=10,
                   help="must match the head being calibrated -- the text tensors "
                        "are part of its input signature")
    p.add_argument("--concepts", nargs="+", default=[
        "person", "bottle", "cup", "chair", "helmet", "hand", "glove",
        "cardboard box", "plastic bag", "bottle"])
    p.add_argument("--ve-engine", default=None,
                   help="fp16 VE-only engine used to produce calibration features")
    p.add_argument("--head-onnx", default=None, help="N=1 head to calibrate")
    p.add_argument("--out", default=None)
    p.add_argument("--exclude-mask-decoder", action="store_true",
                   help="leave the mask decoder in fp16. Two reasons, and the "
                        "second is the binding one: (1) the question is what "
                        "quantizing DETR buys, and the mask decoder is not DETR; "
                        "(2) calibrating it at N=10 res 1008 OOMs -- its "
                        "activations are [10,200,288,288] = 663 MB each and "
                        "ModelOpt retains hundreds of them as graph outputs. "
                        "Measured: direct N=10 calibration was OOM-killed.")
    p.add_argument("--no-mha", action="store_true",
                   help="ablation: leave the 6 DETR attention BMMs in fp16. The "
                        "quantization experiment measured that quantizing them "
                        "bought ~0 at N=1 because they are memory-bound; at N=10 "
                        "they are 10x larger, so it is worth re-testing rather "
                        "than inheriting the verdict.")
    args = p.parse_args(argv)

    ve_eng = args.ve_engine or f"out/engines/veonly_fp16_rect_{args.res}.engine"
    head = args.head_onnx or (f"out/res={args.res}/head.onnx" if args.n_max == 10
                              else f"out/res={args.res}_n{args.n_max}/head.onnx")
    out = args.out or f"out/head_int8_{args.res}_n{args.n_max}.onnx"
    for f in (ve_eng, head):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")

    print(f"[calib] running VE on {args.calib_n} real frames ...", flush=True)
    samples = collect(args.res, args.calib_n, args.image_dir, args.lib_dir,
                      args.concepts, ve_eng, args.n_max)
    if not samples:
        raise SystemExit("no calibration samples")
    print(f"[calib] {len(samples)} samples; shapes "
          f"{ {k: v.shape for k, v in samples[0].items()} }", flush=True)

    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_first(self):
            return samples[0]

        def get_next(self):
            if self.i >= len(samples):
                return None
            d = samples[self.i]
            self.i += 1
            return d

        def rewind(self):
            self.i = 0

    import modelopt.onnx.quantization as moq

    # ANCHORED AT THE START -- the same re.match trap the VE recipe hit.
    # "/mask_decoder/.*" matches nothing here; the names start "/model/".
    exclude = [".*mask_decoder.*"] if args.exclude_mask_decoder else []
    print(f"[quant] int8 W8A8 on the head, disable_mha_qdq={args.no_mha}, "
          f"exclude={exclude}", flush=True)
    t0 = time.time()
    moq.quantize(
        onnx_path=head, calibration_data_reader=Reader(), quantize_mode="int8",
        calibration_method="max", calibration_eps=["cpu"], high_precision_dtype="fp32",
        disable_mha_qdq=args.no_mha, mha_accumulation_dtype="fp32",
        nodes_to_exclude=exclude, use_external_data_format=True, output_path=out)
    print(f"[quant] done in {time.time() - t0:.0f}s -> {out}", flush=True)

    import onnx
    m = onnx.load(out, load_external_data=False)
    qn = [n for n in m.graph.node if n.op_type == "QuantizeLinear"]
    md = [n for n in qn if "mask_decoder" in n.name]
    print(f"    QuantizeLinear nodes: {len(qn)}  of which in mask_decoder: "
          f"{len(md)}", flush=True)
    if not qn:
        raise SystemExit("FAIL: nothing was quantized")
    if args.exclude_mask_decoder and md:
        raise SystemExit(
            f"FAIL: {len(md)} Q/DQ landed in the mask decoder despite the "
            f"exclusion -- the re.match pattern did not anchor. e.g. "
            f"{[n.name for n in md[:3]]}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
