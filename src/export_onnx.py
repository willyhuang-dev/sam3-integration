#!/usr/bin/env python3
"""Export the integrated model as the VE / head pair of ONNX graphs.

Why a pair and not one graph
----------------------------
Inference wants one engine, and it gets one -- `merge.py` stitches the two
graphs back together after quantization. The split exists purely because
ModelOpt's INT8 calibration builds an ONNX Runtime session that holds ~331
activation tensors as graph outputs; on the monolithic SAM3 that exceeds this
box's 28 GB of host RAM and gets OOM-killed. Calibrating the VE alone fits.

Why the batch axis is dynamic
-----------------------------
The measurement matrix is 5 resolutions x 5 batch sizes. Exporting 25 ONNX
graphs would cost ~25 x 10 minutes and ~1.9 GB each; exporting 5 with a dynamic
batch axis and letting trtexec pin min=opt=max per engine costs 5. The engines
are still single-batch-optimised, which is what `opt_bs` means.

Memory
------
`torch.no_grad()` around the export is the one lever that matters (dyntext
measured 17.3 GB -> 4.3 GB at size 1008). Tracing EXECUTES the model, and
without it every intermediate keeps an autograd graph alive -- a second copy of
exactly the activations that dominate at large sizes.

Usage
-----
    python3 -m src.export_onnx --size 1008 --n-max 10 --out-dir out/res=1008
    python3 -m src.export_onnx --size 1008 --no-prune     # dense baseline
"""
import argparse
import gc
import os
import time

import torch

from .integrated import (
    MASK_UPSCALE, PATCH, IntegratedVE, MultiConceptHead, load_model,
    rect_rows_for,
)

VE_INPUTS = ["images"]
VE_OUTPUTS = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
HEAD_INPUTS = VE_OUTPUTS + ["text_features", "text_mask"]
HEAD_OUTPUTS = ["pred_boxes", "pred_logits", "presence_logits", "pred_masks"]
HEAD_OUTPUTS_NOMASK = HEAD_OUTPUTS[:3]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=os.environ.get("SAM3_MODEL_ID", "facebook/sam3"))
    p.add_argument("--size", type=int, required=True,
                   help=f"input resolution; must be a multiple of {PATCH}")
    p.add_argument("--n-max", type=int, default=10,
                   help="concept slots baked into the graph (default: 10). Fixed "
                        "at export; WHICH concepts fill them is not.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--aspect", default="16:9",
                   help="content aspect ratio the letterbox preserves, W:H "
                        "(default 16:9). Sets how many token rows survive.")
    p.add_argument("--no-prune", action="store_true",
                   help="dense baseline: skip strategy C entirely. This is the "
                        "control every pruned number is compared against.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cpu",
                   help="cpu keeps the 16 GB card free; the traced graph is "
                        "identical either way")
    p.add_argument("--no-masks", action="store_true",
                   help="drop pred_masks. TensorRT then dead-code-eliminates the "
                        "whole mask decoder, so the difference against the "
                        "with-masks engine IS the mask decoder's cost -- and it "
                        "is also the deployment option for detection-only use, "
                        "where pred_masks is what makes large batches OOM.")
    p.add_argument("--topk-masks", type=int, default=None,
                   help="decode masks only for the top-K scoring queries "
                        "instead of all 200. Near-lossless (thresholding keeps "
                        "a handful anyway) and it is what stops pred_masks from "
                        "dominating VRAM: 663 MB -> 66 MB per batch element at "
                        "K=20, res 1008.")
    p.add_argument("--head-only", action="store_true",
                   help="export only head.onnx. The VE graph does NOT depend on "
                        "--n-max -- concept slots live entirely in the head -- so "
                        "sweeping N reuses one VE (and one INT8 calibration) and "
                        "re-exports only the 92 MB head.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.size % PATCH:
        raise SystemExit(f"--size must be a multiple of {PATCH}; {args.size} is not")
    aw, ah = (int(v) for v in args.aspect.split(":"))
    grid = args.size // PATCH
    rect = None if args.no_prune else rect_rows_for(args.size, aw, ah)
    mask = grid * MASK_UPSCALE
    os.makedirs(args.out_dir, exist_ok=True)

    kept = "dense (no pruning)" if rect is None else \
        f"{rect}/{grid} rows = {rect / grid:.1%} of tokens"
    print(f"[cfg] size={args.size} grid={grid}x{grid} n_max={args.n_max} "
          f"mask={mask}x{mask}", flush=True)
    print(f"[cfg] strategy C: {kept}  (aspect {aw}:{ah}, bar at the bottom)",
          flush=True)

    print("[load] model ...", flush=True)
    model = load_model(args.model_id, args.size).to(args.device)
    ve = IntegratedVE(model, rect_rows=rect).eval()
    head = MultiConceptHead(model, n_max=args.n_max,
                            return_masks=not args.no_masks,
                            topk_masks=args.topk_masks).eval()

    img = torch.zeros(1, 3, args.size, args.size, device=args.device)
    # Real-ish values rather than zeros: the traced graph cannot depend on them,
    # but shape/precision inference occasionally does.
    torch.manual_seed(0)
    tf = torch.randn(args.n_max, 32, 256, device=args.device)
    tm = torch.zeros(args.n_max, 32, dtype=torch.int64, device=args.device)
    tm[:, :6] = 1

    print("[check] split == monolithic ...", flush=True)
    with torch.no_grad():
        f = ve(img)
        out = head(*f, tf, tm)
    for n, t in zip(VE_OUTPUTS, f):
        print(f"    {n:14s} {tuple(t.shape)}", flush=True)
    head_outputs = HEAD_OUTPUTS_NOMASK if args.no_masks else HEAD_OUTPUTS
    for n, t in zip(head_outputs, out):
        print(f"    {n:16s} {tuple(t.shape)}", flush=True)

    dyn_b = {n: {0: "batch"} for n in VE_OUTPUTS}
    if args.head_only:
        print("[onnx] VE skipped (--head-only); it does not depend on n_max",
              flush=True)
    else:
        t0 = time.time()
        ve_path = os.path.join(args.out_dir, "ve.onnx")
        print("[onnx] VE ...", flush=True)
        with torch.no_grad():
            torch.onnx.export(
                ve, (img,), ve_path,
                input_names=VE_INPUTS, output_names=VE_OUTPUTS,
                opset_version=args.opset, do_constant_folding=True, dynamo=False,
                dynamic_axes={"images": {0: "batch"}, **dyn_b})
        print(f"    {ve_path} ({os.path.getsize(ve_path) / 2**20:.0f} MiB) "
              f"{time.time() - t0:.0f}s", flush=True)

    del out
    gc.collect()

    t0 = time.time()
    head_path = os.path.join(args.out_dir, "head.onnx")
    print("[onnx] head ...", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            head, (*f, tf, tm), head_path,
            input_names=HEAD_INPUTS, output_names=head_outputs,
            opset_version=args.opset, do_constant_folding=True, dynamo=False,
            dynamic_axes={**dyn_b, **{n: {0: "batch"} for n in head_outputs}})
    print(f"    {head_path} ({os.path.getsize(head_path) / 2**20:.0f} MiB) "
          f"{time.time() - t0:.0f}s", flush=True)

    meta = os.path.join(args.out_dir, "meta.json")
    with open(meta, "w") as fh:
        import json
        json.dump({"size": args.size, "grid": grid, "n_max": args.n_max,
                   "rect_rows": rect, "mask": None if args.no_masks else mask,
                   "masks": not args.no_masks, "topk_masks": args.topk_masks,
                   "aspect": args.aspect,
                   "opset": args.opset}, fh, indent=2)
    print(f"[done] {meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
