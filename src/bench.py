#!/usr/bin/env python3
"""Build one engine per (resolution, batch) and measure latency + VRAM.

Timing is trtexec's "GPU Compute Time" median, deliberately: every number the
quantization experiment published came from it, so these are comparable to
those rather than to a fresh harness. It is pure device compute -- no H2D/D2H,
no Python.

VRAM is sampled from `nvidia-smi --query-compute-apps` while the benchmark
runs, because that is the number that decides whether a configuration fits on
the card. The deterministic breakdown (weights + activation arena + I/O
buffers) is reported alongside so an OOM can be attributed rather than guessed.

Each engine is built with min = opt = max = the batch under test. That is what
`opt_bs` means: TensorRT picks tactics for exactly that shape, with no profile
compromise, which is the best case a deployment could get.

⚠️ The mask output is the term that decides where this stops fitting:
pred_masks is [B, N_MAX, 200, M, M] float32 -- at 1008 with N_MAX=10 that is
663 MB per batch element. It is reported per row so an OOM is never a mystery.

Usage:
    python3 -m src.bench --plan plan.json --out results.jsonl
    python3 -m src.bench --onnx out/merged_644.onnx --res 644 --bs 1 2 4
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time

TRTEXEC = os.environ.get("TRTEXEC", "trtexec")
GPU_MEM_RE = re.compile(r"GPU Compute Time: .*median = ([0-9.]+) ms")
THROUGHPUT_RE = re.compile(r"Throughput: ([0-9.]+) qps")


class GpuSampler(threading.Thread):
    """Peak GPU memory of any compute process, sampled while a child runs.

    Queries every 100 ms. The card is shared with other containers on this box,
    so the sampler records the TOTAL used and the per-PID used separately --
    reporting only the total would silently attribute a neighbour's 7 GB to us.
    """

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_total = 0
        self.peak_procs = {}
        self._done = threading.Event()   # NOT _stop: Thread._stop is a real method

    def run(self):
        while not self._done.is_set():
            try:
                tot = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.peak_total = max(self.peak_total, int(tot.splitlines()[0]))
                apps = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                for line in apps.splitlines():
                    if not line.strip():
                        continue
                    pid, mem = (v.strip() for v in line.split(","))
                    self.peak_procs[pid] = max(self.peak_procs.get(pid, 0), int(mem))
            except Exception:
                pass
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()
        self.join(timeout=3)


def run(cmd, log_path=None, sample=False):
    sampler = GpuSampler() if sample else None
    if sampler:
        sampler.start()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if sampler:
        sampler.stop()
    out = (r.stdout or "") + (r.stderr or "")
    if log_path:
        with open(log_path, "w") as fh:
            fh.write(" ".join(cmd) + "\n\n" + out)
    return r.returncode, out, wall, sampler


def build(onnx, engine, res, bs, int8, log_dir, tag):
    shape = f"images:{bs}x3x{res}x{res}"
    cmd = [TRTEXEC, f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16",
           f"--minShapes={shape}", f"--optShapes={shape}", f"--maxShapes={shape}",
           "--skipInference"]
    if int8:
        cmd.insert(3, "--int8")
    log = os.path.join(log_dir, f"build_{tag}_r{res}_b{bs}.log")
    rc, out, wall, _ = run(cmd, log)
    ok = rc == 0 and os.path.exists(engine)
    return ok, wall, out, log


def bench(engine, res, bs, iters, log_dir, tag):
    cmd = [TRTEXEC, f"--loadEngine={engine}", f"--shapes=images:{bs}x3x{res}x{res}",
           f"--iterations={iters}", "--warmUp=500", "--avgRuns=10", "--noDataTransfers"]
    log = os.path.join(log_dir, f"bench_{tag}_r{res}_b{bs}.log")
    rc, out, wall, sampler = run(cmd, log, sample=True)
    m = GPU_MEM_RE.search(out)
    q = THROUGHPUT_RE.search(out)
    return {
        "ok": rc == 0 and m is not None,
        "latency_ms": float(m.group(1)) if m else None,
        "qps": float(q.group(1)) if q else None,
        "vram_peak_total_mib": sampler.peak_total if sampler else None,
        "vram_peak_proc_mib": max(sampler.peak_procs.values())
        if sampler and sampler.peak_procs else None,
        "bench_log": log,
        "stderr_tail": None if rc == 0 else out[-1500:],
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--bs", type=int, nargs="+", required=True)
    p.add_argument("--tag", required=True,
                   help="what this engine IS, e.g. int8_rect or fp16_dense; "
                        "goes into the result rows")
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument("--rect-rows", type=int, default=None)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--engine-dir", default="out/engines")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--delete-engines", action="store_true",
                   help="engines are ~0.5-1 GB each and the matrix has dozens, "
                        "but they are KEPT by default: the detection check and "
                        "the layer profile both need a real engine, and "
                        "rebuilding one costs ~5 minutes. Disk is the cheap "
                        "resource here, build time is not.")
    args = p.parse_args(argv)

    os.makedirs(args.engine_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    mask = args.res // 14 * 4

    for bs in args.bs:
        engine = os.path.join(args.engine_dir, f"{args.tag}_r{args.res}_b{bs}.engine")
        row = {"tag": args.tag, "res": args.res, "bs": bs, "n_max": args.n_max,
               "rect_rows": args.rect_rows, "int8": args.int8, "mask": mask,
               "mask_bytes": bs * args.n_max * 200 * mask * mask * 4,
               "ts": time.strftime("%F %T")}
        print(f"=== {args.tag} res={args.res} bs={bs} "
              f"(pred_masks {row['mask_bytes'] / 2**30:.2f} GiB) ===", flush=True)

        ok, wall, out, log = build(args.onnx, engine, args.res, bs,
                                   args.int8, args.log_dir, args.tag)
        row["build_s"] = round(wall, 1)
        row["build_ok"] = ok
        if not ok:
            row["error"] = "build failed"
            row["stderr_tail"] = out[-1500:]
            print(f"    BUILD FAILED after {wall:.0f}s -- see {log}", flush=True)
        else:
            row["engine_mib"] = round(os.path.getsize(engine) / 2**20, 1)
            print(f"    built in {wall:.0f}s, {row['engine_mib']:.0f} MiB", flush=True)
            row.update(bench(engine, args.res, bs, args.iters, args.log_dir, args.tag))
            if row["ok"]:
                print(f"    {row['latency_ms']:.2f} ms  {row['qps']:.1f} qps  "
                      f"VRAM {row['vram_peak_proc_mib']} MiB", flush=True)
            else:
                print("    BENCH FAILED (likely OOM) -- row kept with the error",
                      flush=True)
            if args.delete_engines:
                os.remove(engine)

        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
