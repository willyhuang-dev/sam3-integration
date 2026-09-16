#!/usr/bin/env python3
"""The whole measurement, end to end, resumable.

    export -> quantize(644 once) -> transplant -> merge -> build -> bench

What gets measured and why it is a 2x2 and not one line
------------------------------------------------------
The ask is the integrated engine: INT8 + strategy C. But a single column of
numbers cannot say WHERE the speed came from, and the two ingredients have very
different risk profiles -- INT8 costs ~1-3% mAP, strategy C costs ~0.9% det50
but only when the black-bar assumption holds. So every resolution gets the
full factorial at bs=1:

    fp16 dense   the stock reference
    fp16 rect    strategy C alone
    int8 dense   quantization alone
    int8 rect    both -- the deliverable

and the deliverable alone is swept across the batch sizes. That is 4x5 + 5x4
extra = 40 engines. The factorial is what made the ROI experiment's central
result trustworthy (it overturned the previous best configuration), so it is
worth the build time.

Resumability matters: a single engine build is minutes, the matrix is hours,
and this box shares its GPU. Every stage checks for its own output first, and
`results.jsonl` is append-only, so a killed run picks up where it stopped.

Usage:
    python3 scripts/run_matrix.py --stage all
    python3 scripts/run_matrix.py --stage export --res 1008
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

RES = [644, 728, 840, 924, 1008]
BATCHES = [1, 2, 4, 6, 8]
N_SWEEP = list(range(1, 11))   # concept slots baked into the engine
N_SWEEP_RES = (644, 1008)      # both ends of the resolution range
N_DEFAULT = 10
CALIB_RES = 644          # the only resolution whose calibration fits in 28 GB
OUT = os.path.join(HERE, "out")
LOGS = os.path.join(HERE, "logs")
RESULTS = os.path.join(HERE, "results.jsonl")


def sh(cmd, log=None):
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if log:
        with open(os.path.join(LOGS, log), "w") as fh:
            fh.write(out)
    if r.returncode:
        print(f"  !! rc={r.returncode}\n  " + out[-1200:].replace("\n", "\n  "),
              flush=True)
    else:
        print(f"  ok ({time.time() - t0:.0f}s)", flush=True)
    return r.returncode


def d_pruned(res):
    return os.path.join(OUT, f"res={res}")


def d_dense(res):
    return os.path.join(OUT, f"res={res}_dense")


def stage_export(res_list):
    for res in res_list:
        for dirname, extra in ((d_pruned(res), []), (d_dense(res), ["--no-prune"])):
            if os.path.exists(os.path.join(dirname, "meta.json")):
                print(f"[export] {dirname} exists, skip", flush=True)
                continue
            print(f"[export] {dirname}", flush=True)
            sh([sys.executable, "-m", "src.export_onnx", "--size", str(res),
                "--out-dir", dirname] + extra,
               log=f"export_{os.path.basename(dirname)}.log")


def stage_quantize():
    """Calibrate once at 644, for the pruned and the dense graph separately.

    Two calibrations, not one: the pruned and dense VE graphs are structurally
    DIFFERENT (the pruned one carries the crop and the zero-pad back), so their
    Q/DQ subgraphs are not interchangeable. Sharing them would be the same
    silent-wrong-transplant this pipeline guards against everywhere else.
    """
    for tag, src in (("rect", d_pruned(CALIB_RES)), ("dense", d_dense(CALIB_RES))):
        out = os.path.join(OUT, f"ve_{CALIB_RES}_{tag}.mp_neck.onnx")
        if os.path.exists(out):
            print(f"[quant] {out} exists, skip", flush=True)
            continue
        print(f"[quant] {tag} @ {CALIB_RES}", flush=True)
        sh([sys.executable, "-m", "src.quantize",
            "--ve", os.path.join(src, "ve.onnx"), "--out", out,
            "--size", str(CALIB_RES)], log=f"quant_{tag}.log")


CORNER_RES = (644, 1008)   # where the single-ingredient corners are measured


def stage_merge(res_list):
    for res in res_list:
        for tag, srcdir in (("rect", d_pruned(res)), ("dense", d_dense(res))):
            # int8+dense is only ever benchmarked at the corner resolutions, and
            # each unneeded one costs a transplant, a merge and ~2 GB of disk.
            skip_int8 = (tag == "dense" and res not in CORNER_RES)
            # fp16: merge the untouched VE straight onto the head
            fp16 = os.path.join(OUT, f"fp16_{tag}_{res}.onnx")
            if not os.path.exists(fp16):
                sh([sys.executable, "-m", "src.merge",
                    os.path.join(srcdir, "ve.onnx"),
                    os.path.join(srcdir, "head.onnx"), fp16],
                   log=f"merge_fp16_{tag}_{res}.log")
            else:
                print(f"[merge] {fp16} exists, skip", flush=True)

            if skip_int8:
                continue
            # int8: transplant the 644 scales first (identity at 644 itself)
            q644 = os.path.join(OUT, f"ve_{CALIB_RES}_{tag}.mp_neck.onnx")
            ve_q = os.path.join(OUT, f"ve_int8_{tag}_{res}.onnx")
            if res == CALIB_RES:
                ve_q = q644
            elif not os.path.exists(ve_q):
                sh([sys.executable, "-m", "src.transplant",
                    os.path.join(srcdir, "ve.onnx"), q644, ve_q],
                   log=f"transplant_{tag}_{res}.log")
            int8 = os.path.join(OUT, f"int8_{tag}_{res}.onnx")
            if not os.path.exists(int8):
                sh([sys.executable, "-m", "src.merge", ve_q,
                    os.path.join(srcdir, "head.onnx"), int8],
                   log=f"merge_int8_{tag}_{res}.log")
            else:
                print(f"[merge] {int8} exists, skip", flush=True)


def d_nhead(res, n):
    return os.path.join(OUT, f"res={res}_n{n}")


def stage_nsweep_prep(res_list):
    """Heads for N = 1..10, reusing one VE and one calibration per resolution.

    The concept slots live ENTIRELY in the head: the vision encoder never sees
    text, so its graph -- and therefore the INT8 scales calibrated on it -- are
    identical for every N. That is why this sweep costs ten 92 MB head exports
    and ten merges rather than ten full exports and ten calibrations.

    It also means the sweep isolates cleanly: anything that changes with N is
    the head, because nothing else changed.
    """
    for res in [r for r in N_SWEEP_RES if r in res_list]:
        ve_q = os.path.join(OUT, f"ve_int8_rect_{res}.onnx") if res != CALIB_RES \
            else os.path.join(OUT, f"ve_{CALIB_RES}_rect.mp_neck.onnx")
        if not os.path.exists(ve_q):
            print(f"[nsweep] quantized VE for {res} not ready yet, skip", flush=True)
            continue
        for n in N_SWEEP:
            if n == N_DEFAULT:
                continue          # already built by the main matrix
            d = d_nhead(res, n)
            if not os.path.exists(os.path.join(d, "head.onnx")):
                sh([sys.executable, "-m", "src.export_onnx", "--size", str(res),
                    "--n-max", str(n), "--out-dir", d, "--head-only"],
                   log=f"export_n{n}_{res}.log")
            merged = os.path.join(OUT, f"int8_rect_{res}_n{n}.onnx")
            if not os.path.exists(merged):
                sh([sys.executable, "-m", "src.merge", ve_q,
                    os.path.join(d, "head.onnx"), merged],
                   log=f"merge_n{n}_{res}.log")


def done_rows():
    if not os.path.exists(RESULTS):
        return set()
    seen = set()
    for line in open(RESULTS):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "ok" in r or "error" in r:   # attempted, not "succeeded" -- a
                                       # build-ok-but-bench-failed row has
                                       # ok=False and no "error" key, and
                                       # without this it gets rebuilt forever
            seen.add((r["tag"], r["res"], r["bs"]))
    return seen


def stage_bench(res_list, print_plan=False):
    seen = done_rows()
    rect_of = {res: json.load(open(os.path.join(d_pruned(res), "meta.json")))["rect_rows"]
               for res in res_list}

    # Order matters more than it looks. An engine build is ~10 minutes and the
    # matrix is dozens of them, so this run WILL be interrupted at some point.
    # The deliverable -- the 5x5 int8+rect sweep that was actually asked for --
    # therefore runs to completion FIRST; the attribution corners are the part
    # that gets cut short if anything gets cut short.
    plan = [("int8_rect", f"int8_rect_{res}.onnx", res, BATCHES, True,
             rect_of[res], N_DEFAULT) for res in res_list]
    # Second ask: what does one more concept slot cost? N is fixed at export, so
    # each point is its own engine. bs=1 keeps it one variable at a time.
    for res in [r for r in N_SWEEP_RES if r in res_list]:
        for n in N_SWEEP:
            if n == N_DEFAULT:
                continue          # the (res, bs=1) cell above IS N=10
            plan.append((f"int8_rect_n{n}", f"int8_rect_{res}_n{n}.onnx", res,
                         [1], True, rect_of[res], n))
    # Reference for the headline speedup: stock precision, stock token count.
    plan += [("fp16_dense", f"fp16_dense_{res}.onnx", res, [1], False, None,
              N_DEFAULT) for res in res_list]
    # The two single-ingredient corners complete the 2x2. Only at the ends of
    # the resolution range: 644 is the structurally special one (its windowed
    # attention saves nothing, see README) and 1008 is the intended operating
    # point, so the pair brackets the behaviour without 10 more builds.
    for res in [r for r in CORNER_RES if r in res_list]:
        plan.append(("fp16_rect", f"fp16_rect_{res}.onnx", res, [1], False,
                     rect_of[res], N_DEFAULT))
        plan.append(("int8_dense", f"int8_dense_{res}.onnx", res, [1], True,
                     None, N_DEFAULT))

    if print_plan:
        # One JSON object per REMAINING engine, for the host-side driver.
        # It has to run outside the container because the fix for the
        # nvidia-driver host-RAM leak is restarting the container, which a
        # process inside it cannot do to itself.
        for tag, onnx_name, res, batches, int8, rect, n_max in plan:
            onnx_path = os.path.join(OUT, onnx_name)
            if not os.path.exists(onnx_path):
                continue
            for b in batches:
                if (tag, res, b) in seen:
                    continue
                print(json.dumps({"tag": tag, "onnx": onnx_path, "res": res,
                                  "bs": b, "int8": int8, "rect": rect,
                                  "n_max": n_max}))
        return

    for tag, onnx_name, res, batches, int8, rect, n_max in plan:
        onnx_path = os.path.join(OUT, onnx_name)
        todo = [b for b in batches if (tag, res, b) not in seen]
        if not todo:
            print(f"[bench] {tag} r{res}: all done, skip", flush=True)
            continue
        if not os.path.exists(onnx_path):
            print(f"[bench] {onnx_path} missing, skip", flush=True)
            continue
        cmd = [sys.executable, "-m", "src.bench",
               "--onnx", onnx_path, "--res", str(res),
               "--bs", *[str(b) for b in todo], "--tag", tag,
               "--n-max", str(n_max), "--out", RESULTS]
        if int8:
            cmd.append("--int8")
        if rect is not None:
            cmd += ["--rect-rows", str(rect)]
        print(f"[bench] {tag} r{res} bs={todo} n_max={n_max}", flush=True)
        subprocess.run(cmd, cwd=HERE)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all",
                   choices=["all", "export", "quantize", "merge", "nsweep",
                            "bench", "plan"])
    p.add_argument("--res", type=int, nargs="+", default=RES)
    args = p.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    # 644 must be exported and calibrated before anything can be transplanted.
    order = sorted(set(args.res), key=lambda r: (r != CALIB_RES, r))
    if args.stage in ("all", "export"):
        stage_export(order)
    if args.stage in ("all", "quantize"):
        stage_quantize()
    if args.stage in ("all", "merge"):
        stage_merge(order)
    if args.stage in ("all", "nsweep"):
        stage_nsweep_prep(order)
    if args.stage == "plan":
        stage_bench(order, print_plan=True)
        return
    if args.stage in ("all", "bench"):
        stage_bench(order)
    print("=== run_matrix done ===", flush=True)


if __name__ == "__main__":
    main()
