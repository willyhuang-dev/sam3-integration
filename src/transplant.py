#!/usr/bin/env python3
"""Move the Q/DQ pairs calibrated at 644 onto a VE graph of another resolution.

Why not just calibrate at each resolution: ModelOpt marks ~331 activation
tensors as graph outputs so it can collect their maxima. On ViT-H at 1008 that
working set exceeds this box's 28 GB and the calibration process is OOM-killed.
At 644 it fits. The two graphs are structurally identical -- same ops, same
tensor names, only the declared dimensions differ -- so the Q/DQ subgraph can
be lifted across.

What is exact and what is not:
  * WEIGHT scales are per-channel over the weight tensor. Resolution-independent
    -- exact.
  * ACTIVATION scales are per-tensor maxima collected at 644. Carried over as-is
    -- an approximation.

The approximation was measured, not assumed: a properly-calibrated VE at 644
reaches cos 0.879 against fp16, and the transplanted one reaches 0.869 at 1008.
The gap is the inherent cost of per-tensor INT8 on ViT-H's dynamic range, not a
transplant artefact, and box mAP retention held at 98.6-98.8%.

⚠️ The failure mode is silent. If the graphs are NOT structurally identical, the
rewiring simply finds no consumer, writes a file, and you get an unquantized
graph that builds and runs. `_check_structure` refuses instead.

Usage:
    python3 -m src.transplant <base.onnx> <quantized_644.onnx> <out.onnx>
"""
import os
import sys
import time
from collections import defaultdict, deque

import onnx

MAX_CARRY = 16   # see _check_structure; a real mismatch produced 124


def _check_structure(base_g, q_g, qdq):
    """Refuse a transplant that cannot land; return the base nodes + weights to carry.

    Three claims, each catching a different way this goes quietly wrong:
      1. the two graphs have the same node count (different topology entirely)
      2. every tensor a Q/DQ node reads either exists in the base or is an
         initializer the quantizer added and we can carry over
      3. the rewiring found consumers (checked by the caller against 0)
    """
    b_nodes = len(base_g.node)
    q_nodes = sum(1 for n in q_g.node
                  if n.op_type not in ("QuantizeLinear", "DequantizeLinear"))
    if b_nodes != q_nodes:
        raise SystemExit(
            f"FAIL: base has {b_nodes} nodes, quantized graph has {q_nodes} "
            f"non-Q/DQ nodes. These are not the same graph at two resolutions; "
            f"transplanting would silently produce a wrong or unquantized model.")

    produced = {o for n in base_g.node for o in n.output}
    produced |= {i.name for i in base_g.input} | {i.name for i in base_g.initializer}
    q_init = {i.name for i in q_g.initializer}
    missing = [n.input[0] for n in qdq
               if n.op_type == "QuantizeLinear" and n.input[0] not in produced]
    # ModelOpt sometimes DUPLICATES a shared weight so each consumer can carry
    # its own scale (names get a "_<k>" suffix). Those duplicates exist only in
    # the quantized graph, so they show up as "missing" -- but they are weights,
    # which are resolution-independent, so carrying them across is correct. What
    # is NOT acceptable is a missing tensor that the quantized graph does not
    # define either: that means the graphs really are different.
    carry = [m for m in missing if m in q_init]
    hard = [m for m in missing if m not in q_init]
    if hard:
        raise SystemExit(
            f"FAIL: {len(hard)} tensors the Q/DQ nodes quantize exist in NEITHER "
            f"graph as initializers, e.g. {hard[:3]} -- these are not the same "
            f"graph at two resolutions.")
    # A handful of carried weights is normal: the quantizer duplicates a shared
    # weight so each consumer can hold its own scale. MANY of them is not -- it
    # means the exporter gave the two graphs different auto-generated names,
    # which only happens when the graphs really differ.
    #
    # Measured: 644 -> 728/840/924/1008 carry 0. 644 -> 588 carries 124, because
    # 588's rect_rows is exactly window_size so window_partition needs NO
    # padding at all, while 644's 26 rows pad to 48. Different padding, different
    # constant folding, different names. Node counts matched anyway, so the
    # count check alone let that through and the result failed to parse.
    if len(carry) > MAX_CARRY:
        raise SystemExit(
            f"FAIL: {len(carry)} weight initializers would have to be carried "
            f"across (limit {MAX_CARRY}). That many means the exporter named "
            f"these two graphs differently, i.e. they are NOT the same graph at "
            f"two resolutions -- matching node counts is not enough evidence. "
            f"Calibrate at the target resolution instead of transplanting.")
    if carry:
        print(f"[transplant] carrying {len(carry)} duplicated weight initializers "
              f"created by the quantizer (e.g. {carry[:2]})", flush=True)
    return b_nodes, carry


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit(__doc__)
    base_path, q_path, out = argv
    t0 = time.time()

    q = onnx.load(q_path, load_external_data=True)
    b = onnx.load(base_path, load_external_data=True)
    qg, bg = q.graph, b.graph

    qdq = [n for n in qg.node
           if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    n_base, carry = _check_structure(bg, qg, qdq)
    print(f"[transplant] base {n_base} nodes, {len(qdq)} Q/DQ to move", flush=True)

    q_init = {i.name: i for i in qg.initializer}
    q_cons = defaultdict(list)
    for n in qg.node:
        for idx, inp in enumerate(n.input):
            if inp:
                q_cons[inp].append((n.name, idx))

    b_init = {i.name for i in bg.initializer}
    need = {inp for n in qdq for inp in n.input[1:] if inp in q_init}
    need |= set(carry)   # the duplicated weights themselves, not just their scales
    for nm in sorted(need):
        if nm not in b_init:
            bg.initializer.append(q_init[nm])
    for n in qdq:
        bg.node.append(n)

    b_nodes = {n.name: n for n in bg.node}
    rewired = 0
    for n in qdq:
        for (cn, idx) in q_cons.get(n.output[0], []):
            if cn in b_nodes and idx < len(b_nodes[cn].input):
                b_nodes[cn].input[idx] = n.output[0]
                rewired += 1
    if rewired == 0:
        raise SystemExit(
            "FAIL: transplanted Q/DQ nodes but rewired 0 consumers -- the "
            "result would be an unquantized graph carrying dead Q/DQ nodes.")
    print(f"[transplant] rewired {rewired} consumer inputs", flush=True)

    # topological sort: the appended Q/DQ nodes sit at the end but feed nodes
    # that come earlier in the list, which ONNX forbids.
    prod = {o: n for n in bg.node for o in n.output}
    nodes = list(bg.node)
    adj, indeg = defaultdict(list), {}
    for n in nodes:
        d = 0
        for inp in n.input:
            if inp and inp in prod and prod[inp] is not n:
                d += 1
                adj[id(prod[inp])].append(n)
        indeg[id(n)] = d
    ready = deque(n for n in nodes if indeg[id(n)] == 0)
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for m in adj[id(n)]:
            indeg[id(m)] -= 1
            if indeg[id(m)] == 0:
                ready.append(m)
    if len(order) != len(nodes):
        raise SystemExit(f"FAIL: topological sort produced {len(order)} of "
                         f"{len(nodes)} nodes -- the graph has a cycle")
    del bg.node[:]
    bg.node.extend(order)

    onnx.save(b, out, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(out) + "_data", size_threshold=1024)
    print(f"[transplant] -> {out} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
