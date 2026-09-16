#!/usr/bin/env python3
"""Turn results.jsonl into the tables in docs/report.md.

Kept separate from the benchmark so the report can be regenerated from the raw
rows without re-running anything, and so the raw rows stay the source of truth.
That ordering was learnt the hard way: the ROI experiment's report once carried
a stale table for weeks while its JSON was correct the whole time.

Rows that FAILED are printed, not dropped. An OOM at (1008, bs=8) is a result --
silently omitting it would read as "not measured".

Usage:
    python3 scripts/make_report.py [--results results.jsonl] [--out docs/report.md]
"""
import argparse
import json
import os
from collections import defaultdict

RES = [644, 728, 840, 924, 1008]
BATCHES = [1, 2, 4, 6, 8]


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    # later rows win, so a re-run overwrites an earlier failure
    latest = {}
    for r in rows:
        latest[(r["tag"], r["res"], r["bs"])] = r
    return latest


def cell(r, field, fmt="{:.1f}"):
    if r is None:
        return "—"
    if not r.get("build_ok", False):
        return "**build 失敗**"
    if not r.get("ok", False):
        return "**OOM/失敗**"
    v = r.get(field)
    return fmt.format(v) if v is not None else "—"


def matrix(rows, tag, field, fmt="{:.1f}"):
    out = ["| res \\ bs | " + " | ".join(str(b) for b in BATCHES) + " |",
           "|---" * (len(BATCHES) + 1) + "|"]
    for res in RES:
        cells = [cell(rows.get((tag, res, b)), field, fmt) for b in BATCHES]
        out.append(f"| **{res}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.jsonl")
    p.add_argument("--out", default="docs/report.md")
    a = p.parse_args()
    rows = load(a.results)
    if not rows:
        raise SystemExit("no rows")

    any_row = next(iter(rows.values()))
    L = []
    L.append("# 量測結果：整合後的 SAM3 INT8 engine")
    L.append("")
    L.append("RTX 5060 Ti 16 GB（sm_120）· TensorRT 10.9 · 每顆 engine "
             "`min=opt=max=bs` · 延遲＝trtexec `GPU Compute Time` median"
             f"（{any_row.get('n_max')} 個概念插槽，單次 forward 全出）")
    L.append("")

    L.append("## 主矩陣：INT8 + 策略 C（交付配置）")
    L.append("")
    L.append("### 延遲（ms，每次 forward 處理 bs 張影像 × N 個概念）")
    L.append("")
    L.append(matrix(rows, "int8_rect", "latency_ms", "{:.1f}"))
    L.append("")
    L.append("### 吞吐（qps）")
    L.append("")
    L.append(matrix(rows, "int8_rect", "qps", "{:.1f}"))
    L.append("")
    L.append("### VRAM 峰值（MiB，行程實際佔用）")
    L.append("")
    L.append(matrix(rows, "int8_rect", "vram_peak_proc_mib", "{:.0f}"))
    L.append("")

    # per-image latency makes the batching question answerable
    L.append("### 每張影像攤提延遲（ms / image）")
    L.append("")
    # Failed rows are carried through, not filtered out: dropping them would
    # render an OOM cell as "—", which reads as "not measured" rather than
    # "measured, and it does not fit".
    per = {}
    for (tag, res, bs), r in rows.items():
        if tag != "int8_rect":
            continue
        per[(tag, res, bs)] = dict(r, per_img=r["latency_ms"] / bs) \
            if r.get("ok") else r
    L.append(matrix(per, "int8_rect", "per_img", "{:.1f}"))
    L.append("")

    L.append("## 概念數（num_classes）掃描，bs=1")
    L.append("")
    L.append("N 是匯出期固定的插槽數，每個 N 是一顆獨立 engine。**VE 完全不隨 N 改變**"
             "（vision encoder 看不到文字），所以這條曲線量到的全部是 head 的成本。")
    L.append("")
    for res in (644, 1008):
        got = [(n, rows.get((f"int8_rect_n{n}" if n != 10 else "int8_rect", res, 1)))
               for n in range(1, 11)]
        got = [(n, r) for n, r in got if r is not None]
        if not got:
            continue
        L.append(f"### res = {res}")
        L.append("")
        L.append("| N | 延遲 (ms) | vs N=1 | 每概念邊際 (ms) | VRAM (MiB) | "
                 "pred_masks | engine (MiB) |")
        L.append("|---|---|---|---|---|---|---|")
        base = None
        prev = None
        for n, r in got:
            lat = r.get("latency_ms") if r.get("ok") else None
            if n == 1 and lat:
                base = lat
            ratio = f"{lat / base:.2f}×" if (lat and base) else "—"
            marg = f"{lat - prev:+.1f}" if (lat and prev) else "—"
            L.append(f"| {n} | {cell(r, 'latency_ms', '{:.1f}')} | {ratio} | {marg} | "
                     f"{cell(r, 'vram_peak_proc_mib', '{:.0f}')} | "
                     f"{r['mask_bytes'] / 2**20:.0f} MiB | "
                     f"{cell(r, 'engine_mib', '{:.0f}')} |")
            if lat:
                prev = lat
        L.append("")

    L.append("## 歸因：加速從哪裡來（bs=1）")
    L.append("")
    L.append("| res | fp16 dense | fp16 + 策略C | INT8 dense | **INT8 + 策略C** "
             "| 策略C 單獨 | INT8 單獨 | 合計 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for res in RES:
        g = {t: rows.get((t, res, 1)) for t in
             ("fp16_dense", "fp16_rect", "int8_dense", "int8_rect")}
        def lat(t):
            r = g[t]
            return r["latency_ms"] if r and r.get("ok") else None
        base = lat("fp16_dense")

        def spd(t):
            v, b = lat(t), base
            return f"{b / v:.2f}×" if (v and b) else "—"
        vals = [f"{lat(t):.1f}" if lat(t) else "—" for t in
                ("fp16_dense", "fp16_rect", "int8_dense", "int8_rect")]
        L.append(f"| {res} | " + " | ".join(vals) +
                 f" | {spd('fp16_rect')} | {spd('int8_dense')} | {spd('int8_rect')} |")
    L.append("")

    L.append("## Engine 體積（MiB）")
    L.append("")
    L.append("| res | INT8+策略C | fp16 dense |")
    L.append("|---|---|---|")
    for res in RES:
        a_ = rows.get(("int8_rect", res, 1))
        b_ = rows.get(("fp16_dense", res, 1))
        L.append(f"| {res} | {cell(a_, 'engine_mib', '{:.0f}')} | "
                 f"{cell(b_, 'engine_mib', '{:.0f}')} |")
    L.append("")

    bad = [r for r in rows.values() if not r.get("ok")]
    L.append("## 沒跑成的格子")
    L.append("")
    if not bad:
        L.append("（無）")
    else:
        L.append("| tag | res | bs | pred_masks 大小 | 情況 | engine 建起來了嗎 | "
                 "實際錯誤 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(bad, key=lambda r: (r["tag"], r["res"], r["bs"])):
            why = "build 失敗" if not r.get("build_ok") else "執行時失敗"
            built = "—" if not r.get("build_ok") else f"是（{r.get('engine_mib')} MiB）"
            # Quote the engine's own words rather than labelling it "OOM".
            # A build that is OOM-KILLED leaves no error line at all -- its log
            # just stops -- so "no error line" is itself the diagnosis, and it
            # means the harness died, not the model.
            tail = r.get("stderr_tail") or ""
            hit = [ln.strip() for ln in tail.splitlines()
                   if "out of memory" in ln.lower() or "cuda failure" in ln.lower()]
            err = hit[-1][:110] if hit else "（log 無錯誤行 → 行程被 OOM killer 殺掉）"
            L.append(f"| {r['tag']} | {r['res']} | {r['bs']} | "
                     f"{r['mask_bytes'] / 2**30:.2f} GiB | {why} | {built} | `{err}` |")
    L.append("")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {a.out} from {len(rows)} rows ({len(bad)} failed)")


if __name__ == "__main__":
    main()
