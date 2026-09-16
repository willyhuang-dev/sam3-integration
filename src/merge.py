#!/usr/bin/env python3
"""Stitch the quantized VE and the fp16 head back into ONE graph.

The split was only ever a workaround for ModelOpt's calibration memory; nothing
about inference wants two engines. Merging is not re-quantization -- it joins
two already-final graphs at the four tensors the VE produces and the head
consumes, so the engine gets one context and the FPN tensors never round-trip
through host memory between stages.

Three mechanical traps, all hit before:
  1. The two graphs can carry different opsets. `ReduceMean` takes `axes` as an
     ATTRIBUTE at opset <= 17 and as an INPUT at >= 18, so relabelling the
     opset passes the checker and then fails; a real version_converter pass is
     required.
  2. IR versions must match.
  3. Initializer names collide, so the head is prefixed.

Usage:
    python3 -m src.merge <ve.onnx> <head.onnx> <out.onnx>
"""
import os
import sys
import time

import onnx
from onnx import compose, version_converter

IO = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
PREFIX = "h/"


def _opset(m):
    return [o for o in m.opset_import if o.domain in ("", "ai.onnx")][0].version


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit(__doc__)
    ve_path, head_path, out = argv
    t0 = time.time()

    ve = onnx.load(ve_path, load_external_data=True)
    head = onnx.load(head_path, load_external_data=True)
    vop, hop = _opset(ve), _opset(head)
    print(f"[merge] VE opset {vop} ir {ve.ir_version} | "
          f"head opset {hop} ir {head.ir_version}", flush=True)

    target = max(vop, hop)
    if vop < target:
        print(f"[merge] converting VE {vop} -> {target}", flush=True)
        ve = version_converter.convert_version(ve, target)
    if hop < target:
        print(f"[merge] converting head {hop} -> {target}", flush=True)
        head = version_converter.convert_version(head, target)
    ve.ir_version = head.ir_version = max(ve.ir_version, head.ir_version)

    head = compose.add_prefix(head, prefix=PREFIX)
    merged = compose.merge_models(
        ve, head, io_map=[(n, PREFIX + n) for n in IO])

    ins = [i.name for i in merged.graph.input]
    outs = [o.name for o in merged.graph.output]
    print(f"[merge] inputs  {ins}", flush=True)
    print(f"[merge] outputs {outs}", flush=True)
    # The four stitched tensors must be GONE from the boundary: if they are
    # still graph inputs the merge silently did not connect and the engine
    # would demand FPN tensors from the caller.
    leaked = [n for n in IO if n in ins or PREFIX + n in ins]
    if leaked:
        raise SystemExit(f"FAIL: {leaked} are still graph inputs -- not merged")

    onnx.save(merged, out, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(out) + "_data", size_threshold=1024)
    print(f"[merge] -> {out} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
