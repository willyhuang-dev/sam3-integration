#!/usr/bin/env python3
"""Check the exported ONNX against eager PyTorch, at MORE THAN ONE batch size.

This exists because of a specific, silent failure mode. The head computes
`batch = fpn_feat_2.shape[0]` and uses it to tile the text tensor. The
TorchScript tracer is free to fold that into the constant 1 -- it warns about
exactly this ("Converting a tensor to a Python integer... will be treated as a
constant"), and if it does, the graph still exports, still declares a dynamic
batch axis, and still runs at batch 1. It only produces wrong answers, or a
shape error, at batch > 1. Every engine in the measurement matrix would be
built on that graph.

So: run batch 1 AND batch 2, compare both against eager.

Usage:
    python3 -m src.verify_onnx --dir out/res=644 [--batches 1 2 4]
"""
import argparse
import json
import os

import numpy as np
import torch

from .integrated import IntegratedVE, MultiConceptHead, load_model

HEAD_OUTPUTS = ["pred_boxes", "pred_logits", "presence_logits", "pred_masks"]


def rel(a, b):
    """Relative Frobenius error.

    Not max|Δ|: SAM3's ViT carries massive activations (max ~229 against a
    median of 0.63), so an absolute elementwise threshold measures activation
    magnitude rather than correctness -- a lesson the ROI experiment paid for
    five times.
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = np.linalg.norm(a - b)
    n = np.linalg.norm(a)
    return float(d / n) if n else float(d)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--model-id", default=os.environ.get("SAM3_MODEL_ID", "facebook/sam3"))
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    p.add_argument("--tol", type=float, default=2e-3,
                   help="relative Frobenius tolerance (ORT vs eager, fp32)")
    args = p.parse_args(argv)

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    size, n_max, rect = meta["size"], meta["n_max"], meta["rect_rows"]
    print(f"[cfg] {meta}", flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    ve_s = ort.InferenceSession(os.path.join(args.dir, "ve.onnx"), so,
                                providers=["CPUExecutionProvider"])
    hd_s = ort.InferenceSession(os.path.join(args.dir, "head.onnx"), so,
                                providers=["CPUExecutionProvider"])

    model = load_model(args.model_id, size)
    ve = IntegratedVE(model, rect_rows=rect).eval()
    head = MultiConceptHead(model, n_max=n_max).eval()

    torch.manual_seed(0)
    tf = torch.randn(n_max, 32, 256)
    tm = torch.zeros(n_max, 32, dtype=torch.int64)
    tm[:, :6] = 1

    bad = []
    for b in args.batches:
        g = torch.Generator().manual_seed(100 + b)
        img = torch.randn(b, 3, size, size, generator=g)
        with torch.no_grad():
            ef = ve(img)
            eo = head(*ef, tf, tm)

        try:
            of = ve_s.run(None, {"images": img.numpy()})
            oo = hd_s.run(None, {
                "fpn_feat_0": of[0], "fpn_feat_1": of[1], "fpn_feat_2": of[2],
                "fpn_pos_2": of[3], "text_features": tf.numpy(),
                "text_mask": tm.numpy()})
        except Exception as e:
            print(f"  batch={b}  ORT FAILED: {type(e).__name__}: {e}", flush=True)
            bad.append((b, "ort-error"))
            continue

        print(f"  batch={b}", flush=True)
        for name, e_t, o_a in zip(
                ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"], ef, of):
            r = rel(e_t.numpy(), o_a)
            print(f"    {name:16s} {str(tuple(o_a.shape)):26s} rel={r:.3e}", flush=True)
            if r > args.tol:
                bad.append((b, name, r))
        for name, e_t, o_a in zip(HEAD_OUTPUTS, eo, oo):
            r = rel(e_t.numpy(), o_a)
            print(f"    {name:16s} {str(tuple(o_a.shape)):26s} rel={r:.3e}", flush=True)
            if r > args.tol:
                bad.append((b, name, r))
            if o_a.shape[0] != b:
                bad.append((b, name, f"batch axis froze at {o_a.shape[0]}"))

    if bad:
        print(f"FAIL: {bad}", flush=True)
        return 1
    print(f"OK: ONNX matches eager at batches {args.batches}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
