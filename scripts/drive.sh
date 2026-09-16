#!/usr/bin/env bash
# Drive the engine builds ONE AT A TIME from the VM706 HOST, restarting the
# container whenever host RAM runs low.
#
# Why this exists
# ---------------
# Building 26 engines leaked ~21 GB of host RAM into the NVIDIA kernel driver:
# not in any /proc/meminfo counter, not in any process's RSS, and not released
# when the process exits. Available RAM went 26 GB -> 0.9 GB over one run. The
# 1008 builds then died -- and they died SILENTLY: trtexec was OOM-killed, so
# its log just stops, with no error line and no "engine built" line. The whole
# 1008 row came back looking like the model could not fit on the card, which is
# a completely wrong reading of what happened.
#
# `docker restart tensorrt` reclaims the part of it that is per-container, so
# this loop does that whenever headroom drops below the threshold. It has to run
# on the host: a process inside the container cannot restart the container it is
# running in.
#
# Usage (on the VM706 host):
#   bash scripts/drive.sh [min_avail_mb]

set -u
REPO_HOST=/home/ubuntu/Documents/willy/models/pretrained_weights/sam3_huggingface/exp/sam3-integration
REPO_CTR=/root/willy/models/pretrained_weights/sam3_huggingface/exp/sam3-integration
MIN_AVAIL_MB=${1:-9000}   # a 1008 build peaked at ~6.8 GB RSS; leave real margin

avail() { free -m | awk 'NR==2{print $7}'; }

restart_container() {
    echo "[drive] restarting container (avail $(avail) MB < ${MIN_AVAIL_MB})"
    docker restart tensorrt >/dev/null
    sleep 8
    echo "[drive] after restart: avail $(avail) MB"
}

in_ctr() { docker exec -w "$REPO_CTR" tensorrt bash -lc "$1"; }

docker start tensorrt >/dev/null 2>&1
sleep 3

while : ; do
    PLAN=$(in_ctr "python3 scripts/run_matrix.py --stage plan" 2>/dev/null)
    [ -z "$PLAN" ] && { echo "[drive] nothing left to build"; break; }

    ROW=$(printf '%s\n' "$PLAN" | head -1)
    read -r TAG ONNX RES BS INT8 RECT NMAX <<<"$(printf '%s' "$ROW" | python3 -c '
import sys, json
r = json.load(sys.stdin)
print(r["tag"], r["onnx"], r["res"], r["bs"],
      int(r["int8"]), r["rect"] if r["rect"] is not None else -1, r["n_max"])')"
    LEFT=$(printf '%s\n' "$PLAN" | wc -l)

    [ "$(avail)" -lt "$MIN_AVAIL_MB" ] && restart_container
    if [ "$(avail)" -lt "$MIN_AVAIL_MB" ]; then
        echo "[drive] STILL only $(avail) MB after a restart -- the driver leak is"
        echo "[drive] past what a container restart can reclaim. Stopping rather"
        echo "[drive] than recording OOM-killed builds as if they were results."
        break
    fi

    echo "[drive] $(date +%H:%M) ${TAG} res=${RES} bs=${BS} N=${NMAX} (${LEFT} left, avail $(avail) MB)"
    CMD="python3 -m src.bench --onnx ${ONNX} --res ${RES} --bs ${BS} --tag ${TAG} --n-max ${NMAX} --out results.jsonl"
    [ "$INT8" = "1" ] && CMD="$CMD --int8"
    [ "$RECT" != "-1" ] && CMD="$CMD --rect-rows ${RECT}"
    in_ctr "$CMD" 2>&1 | grep -E '^(    |=== )' | tail -4

    # A row is always appended (success or failure), so the plan shrinks and the
    # loop cannot spin forever on one entry.
done
echo "[drive] done: $(in_ctr 'wc -l < results.jsonl') rows"
