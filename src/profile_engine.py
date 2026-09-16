#!/usr/bin/env python3
"""Where does the time actually go inside one merged engine.

The first measurement raised a question the matrix alone cannot answer: at
N = 10 concepts the engine costs 2.2x what the single-concept reference did,
even though the vision encoder -- 93% of the compute at N = 1 -- runs exactly
once and is both pruned and quantized. That means the head has become a large
share, and the head is the part that was deliberately left in fp16 because,
at N = 1, quantizing it bought 1.10x on 7% of the pipeline.

If the head is now the majority, that decision is worth revisiting, and this
script is what decides it rather than arguing about it.

Buckets come from the merged graph's own naming: VE layers keep their
`/vision_encoder/...` names and the head was prefixed `h/` at merge time, so
the split is exact rather than heuristic.

Usage:
    python3 -m src.profile_engine --engine out/engines/int8_rect_r644_b1.engine \
        --res 644 --bs 1
"""
import argparse
import json
import os
import re
import subprocess
from collections import defaultdict

TRTEXEC = os.environ.get("TRTEXEC", "trtexec")

# Ordered: the first pattern that matches wins, so put the specific ones first.
BUCKETS = [
    ("mask decoder", re.compile(r"h/.*mask_decoder")),
    ("DETR decoder", re.compile(r"h/.*detr_decoder")),
    ("DETR encoder", re.compile(r"h/.*detr_encoder")),
    ("head other", re.compile(r"^h/")),
    ("VE neck (fp16)", re.compile(r"/vision_encoder/neck")),
    ("VE backbone", re.compile(r"/vision_encoder/backbone")),
    ("VE other", re.compile(r"/vision_encoder")),
]


def bucket_of(name):
    for label, pat in BUCKETS:
        if pat.search(name):
            return label
    return "unattributed"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default=None, help="write the buckets as JSON here")
    args = p.parse_args(argv)

    cmd = [TRTEXEC, f"--loadEngine={args.engine}",
           f"--shapes=images:{args.bs}x3x{args.res}x{args.res}",
           f"--iterations={args.iters}", "--warmUp=500",
           "--dumpProfile", "--profilingVerbosity=detailed",
           "--separateProfileRun", "--noDataTransfers"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode:
        print(out[-2000:])
        raise SystemExit(f"trtexec failed rc={r.returncode}")

    # Layer rows look like:  name  Runtime, %  Invocations  Average time (ms)
    rows = []
    for line in out.splitlines():
        m = re.match(r"\[.*\]\s+(.*?)\s+([0-9.]+)\s+([0-9]+)\s+([0-9.]+)\s*$", line)
        if m and not m.group(1).startswith("Total"):
            rows.append((m.group(1).strip(), float(m.group(4))))
    if not rows:
        # fall back to the simpler two-column layout some TRT versions print
        for line in out.splitlines():
            m = re.match(r"\s*(\S.*?)\s{2,}([0-9.]+)\s*$", line)
            if m:
                rows.append((m.group(1).strip(), float(m.group(2))))
    if not rows:
        raise SystemExit("could not parse any layer rows from the profile; "
                         "the trtexec output format has changed -- inspect it "
                         "rather than trusting a zeroed table")

    agg = defaultdict(float)
    cnt = defaultdict(int)
    for name, ms in rows:
        agg[bucket_of(name)] += ms
        cnt[bucket_of(name)] += 1
    total = sum(agg.values())

    print(f"\n{args.engine}  res={args.res} bs={args.bs}  "
          f"total {total:.2f} ms over {len(rows)} layers\n")
    print(f"{'bucket':18s} {'ms':>9s} {'%':>7s} {'layers':>7s}")
    order = [b for b, _ in BUCKETS] + ["unattributed"]
    for b in order:
        if b in agg:
            print(f"{b:18s} {agg[b]:9.2f} {100 * agg[b] / total:6.1f}% {cnt[b]:7d}")
    ve = sum(v for k, v in agg.items() if k.startswith("VE"))
    hd = sum(v for k, v in agg.items() if not k.startswith("VE") and k != "unattributed")
    print(f"\n  VE   {ve:7.2f} ms ({100 * ve / total:.1f}%)")
    print(f"  head {hd:7.2f} ms ({100 * hd / total:.1f}%)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"engine": args.engine, "res": args.res, "bs": args.bs,
                       "total_ms": total, "buckets": dict(agg),
                       "ve_ms": ve, "head_ms": hd}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
