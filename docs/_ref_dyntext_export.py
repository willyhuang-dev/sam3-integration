#!/usr/bin/env python3
"""Export MultiConceptSam3 to ONNX.

The graph contract (inputs, outputs, shapes) lives in model.py -- read that
first if you want to know what comes out the other end.

Memory
------
ONNX tracing at SIZE=1008 peaks above 13 GB of HOST RAM (measured; the cost is
dominated by the vision backbone, whose token count grows as (SIZE/14)^2, NOT by
N_MAX -- dropping N_MAX from 10 to 4 saves only ~380 MB). Budget accordingly.

The exporter choice matters more than any flag: see README for measured peaks
per (size, exporter, constant-folding) combination. Do not assume
--no-constant-folding lowers peak memory; measure it.

Usage
-----
    python3 export_onnx.py --model-dir /path/to/sam3 --out model.onnx \
        --size 1008 --n-max 10

`--model-dir` is a local checkout of the (gated) `facebook/sam3` HuggingFace
repo. Everything else has a working default.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import text_library as TL  # noqa: E402  (vendored alongside this file)
from model import MASK_UPSCALE, PATCH, MultiConceptSam3, load_model  # noqa: E402


def build_example_text(lib_dir, n_max):
    """Example text tensors, taken from the concept library.

    These are traced into the graph only as SHAPES -- the values are irrelevant
    to the exported model, because text is a runtime input. Using real library
    vectors (rather than zeros) keeps the traced dtypes and value ranges
    realistic, which avoids surprises in downstream shape/precision inference.
    """
    manifest = TL.load_manifest(lib_dir)
    union = list(manifest.keys())[:n_max]
    if not union:
        raise SystemExit(f"concept library at {lib_dir} is empty")
    tf, tm, slots = TL.assemble(lib_dir, union, n_max)
    return (
        torch.from_numpy(tf.astype(np.float32)),
        torch.from_numpy(tm.astype(np.int64)),
        slots,
    )


def build_example_image(size, image_path=None):
    """Example pixel_values.

    Defaults to a synthetic tensor. The real export only needs the SHAPE to be
    right -- pixel content cannot change the traced graph, and requiring a
    specific sample image was one of the things that made the previous script
    impossible to run outside its original directory. --image is kept for the
    case where you want the tracer to see a genuine normalised frame.
    """
    if image_path:
        from PIL import Image
        from transformers import Sam3Processor
        proc = Sam3Processor.from_pretrained(os.environ["SAM3_MODEL_DIR"])
        pv = proc(images=Image.open(image_path).convert("RGB"),
                  text="x", return_tensors="pt")["pixel_values"]
        if pv.shape[-1] != size:
            pv = torch.nn.functional.interpolate(
                pv, size=(size, size), mode="bilinear", align_corners=False)
        return pv
    # SAM3 normalises to roughly [-1, 1]; mid-grey is a safe, deterministic stand-in.
    return torch.zeros(1, 3, size, size, dtype=torch.float32)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=os.environ.get("SAM3_MODEL_DIR"),
                   help="local checkout of the gated facebook/sam3 HF repo "
                        "(or set SAM3_MODEL_DIR)")
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--size", type=int, default=644,
                   help=f"input resolution; must be a multiple of {PATCH} (default: 644)")
    p.add_argument("--n-max", type=int, default=10,
                   help="concept slots baked into the graph (default: 10). This "
                        "IS fixed at export; the concepts themselves are not.")
    p.add_argument("--lib-dir", default=os.path.join(HERE, "concept_library"),
                   help="concept library used only to shape the example text input")
    p.add_argument("--image", default=None,
                   help="optional real image for the tracer; synthetic by default")
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--no-constant-folding", action="store_true",
                   help="disable ONNX constant folding. TensorRT folds constants "
                        "itself, so the engine is unaffected either way. See the "
                        "memory notes in README before assuming this helps.")
    p.add_argument("--dynamo", action="store_true",
                   help="use the torch.export-based ONNX exporter (PyTorch 2.9+ "
                        "default) instead of the legacy TorchScript one. Traces "
                        "differently, so peak memory differs -- worth trying when "
                        "the legacy path runs out of RAM.")
    p.add_argument("--save-eager", action="store_true",
                   help="also dump the eager-mode reference outputs next to the "
                        "ONNX, for numerical comparison. Costs ~700 MB at "
                        "SIZE=1008 and is off by default for that reason.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.model_dir:
        raise SystemExit("--model-dir (or SAM3_MODEL_DIR) is required")
    if args.size % PATCH:
        raise SystemExit(f"--size must be a multiple of {PATCH}; {args.size} is not")
    os.environ.setdefault("SAM3_MODEL_DIR", args.model_dir)

    mask_dim = args.size // PATCH * MASK_UPSCALE
    print(f"[cfg] size={args.size} n_max={args.n_max} -> mask {mask_dim}x{mask_dim}, "
          f"opset={args.opset}, constant_folding={not args.no_constant_folding}")

    print("[load] model ...")
    model = load_model(args.model_dir, args.size)
    tf, tm, slots = build_example_text(args.lib_dir, args.n_max)
    pv = build_example_image(args.size, args.image)
    print(f"[load] example text slots: {slots}")

    wrapper = MultiConceptSam3(model).eval()
    with torch.no_grad():
        b, l, m = wrapper(pv, tf, tm)
    print(f"[eager] pred_boxes{tuple(b.shape)} pred_logits{tuple(l.shape)} "
          f"pred_masks{tuple(m.shape)}")

    if args.save_eager:
        torch.save({"boxes": b, "logits": l, "masks": m, "pixel_values": pv,
                    "text_features": tf, "text_mask": tm, "slots": slots},
                   args.out + ".eager.pt")
        print(f"[eager] reference saved -> {args.out}.eager.pt")

    # Free the eager activations before tracing. At SIZE=1008 the mask tensor
    # alone is ~633 MB and tracing needs headroom; holding both is wasteful.
    del b, l, m
    import gc
    gc.collect()

    print("[onnx] exporting ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    print(f"[onnx] exporter={'dynamo' if args.dynamo else 'torchscript'}")
    # torch.no_grad() around the export matters at large --size. Tracing runs the
    # model, and without this every intermediate keeps an autograd graph alive --
    # a second copy of the activations that is pure waste here, since export
    # never backpropagates. At 1008 the activations are the dominant cost, so
    # this is one of the few levers that acts on the actual bottleneck (unlike
    # --n-max or constant folding, both measured to be near-irrelevant).
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (pv, tf, tm), args.out,
            input_names=["images", "text_features", "text_mask"],
            output_names=["pred_boxes", "pred_logits", "pred_masks"],
            opset_version=args.opset, dynamo=args.dynamo,
            do_constant_folding=not args.no_constant_folding,
        )
    size_gb = os.path.getsize(args.out) / (1024 ** 3)
    print(f"[done] {args.out} ({size_gb:.2f} GB)")
    print("[next] build an fp16 engine:  ./build_engine.sh", args.out, "model.engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
