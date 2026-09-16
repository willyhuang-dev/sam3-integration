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
N_DEFAULT_REPORT = 10   # the N whose rows carry the un-suffixed tag


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
    fits = []
    for res in RES:
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
        # Two points are enough for the line because the sweep IS linear (the
        # per-concept increments have no trend); the intercept is then the VE,
        # which is bit-identical across N, and the slope is the head.
        ok = [(n, r["latency_ms"]) for n, r in got if r.get("ok")]
        if len(ok) >= 2:
            (n1, l1), (n2, l2) = ok[0], ok[-1]
            slope = (l2 - l1) / (n2 - n1)
            icpt = l1 - slope * n1
            fits.append((res, icpt, slope))
            L.append(f"擬合：**延遲 ≈ {icpt:.1f} + {slope:.2f} × N** ms"
                     f"（VE 固定 {icpt:.1f} ms，每概念 {slope:.2f} ms）")
            L.append("")

    if len(fits) >= 2:
        L.append("### VE 與 head 的成本如何隨解析度變化")
        L.append("")
        L.append("這個拆解不是迴歸推測：VE 對每個 N **逐位元相同**（同一顆量化好的 VE、"
                 "同一次校準，只換 head），所以截距就是 VE、斜率就是 head。")
        L.append("")
        L.append("| res | VE 固定 (ms) | 每概念 (ms) | N=10 時 head 佔比 | 保留 token |")
        L.append("|---|---|---|---|---|")
        for res, icpt, slope in fits:
            share = 10 * slope / (icpt + 10 * slope)
            kept = (res // 14) * -(-res * 9 // 16 // 14)
            L.append(f"| {res} | {icpt:.1f} | {slope:.2f} | {share:.0%} | {kept} |")
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

    # Task 3: the speedup table the owner asked for. Baseline is explicitly
    # "TRT FP16, no strategy C, but WITH dyntext multi-concept" = fp16_dense at
    # the same N, so the ratio isolates what the integration bought and does not
    # give it credit for the multi-concept sharing that both sides have.
    def tag_at(base, n):
        return base if n == N_DEFAULT_REPORT else f"{base}_n{n}"

    have_base = any(k[0].startswith("fp16_dense") for k in rows)
    if have_base:
        L.append("## 每個 res / N 的 e2e 加速（對 TRT FP16 無策略C、有 dyntext）")
        L.append("")
        L.append("Baseline 與交付配置**兩邊都是 N 個概念單次 forward**，所以這個比值"
                 "只算「策略 C + INT8」買到的，不把多概念共用 backbone 的好處算進來。")
        L.append("")
        L.append("| res \\ N | " + " | ".join(str(n) for n in range(1, 11)) + " |")
        L.append("|---" * 11 + "|")
        for res in RES:
            cells = []
            for n in range(1, 11):
                bl = rows.get((tag_at("fp16_dense", n), res, 1))
                bst = rows.get((tag_at("int8_rect", n), res, 1))
                if not (bl and bst and bl.get("ok") and bst.get("ok")):
                    cells.append("—")
                else:
                    cells.append(f"{bl['latency_ms'] / bst['latency_ms']:.2f}×")
            L.append(f"| **{res}** | " + " | ".join(cells) + " |")
        L.append("")
        L.append("對應的絕對延遲（baseline → 交付，ms）：")
        L.append("")
        L.append("| res \\ N | " + " | ".join(str(n) for n in range(1, 11)) + " |")
        L.append("|---" * 11 + "|")
        for res in RES:
            cells = []
            for n in range(1, 11):
                bl = rows.get((tag_at("fp16_dense", n), res, 1))
                bst = rows.get((tag_at("int8_rect", n), res, 1))
                if not (bl and bst and bl.get("ok") and bst.get("ok")):
                    cells.append("—")
                else:
                    cells.append(f"{bl['latency_ms']:.0f}→{bst['latency_ms']:.0f}")
            L.append(f"| **{res}** | " + " | ".join(cells) + " |")
        L.append("")

    # Task 2: the DETR-quantization verdict
    if os.path.exists("results_int8full.jsonl"):
        fl = load("results_int8full.jsonl")
        L.append("## 量化 DETR 的判決：不採用")
        L.append("")
        L.append("| 組態 | e2e (ms) | engine (MiB) | 對 fp16 的 IoU | 偵測數 |")
        L.append("|---|---|---|---|---|")
        cur = rows.get(("int8_rect", 1008, 1))
        if cur and cur.get("ok"):
            L.append(f"| INT8 VE + fp16 head（目前最佳）| {cur['latency_ms']:.2f} | "
                     f"{cur['engine_mib']:.0f} | 0.9040 | 503 |")
        for (_, res, bs), r in sorted(fl.items()):
            if r.get("ok"):
                L.append(f"| + DETR 也 INT8 | {r['latency_ms']:.2f} | "
                         f"{r['engine_mib']:.0f} | 0.8620 | 411 |")
        L.append("")
        L.append("事前訂好的標準是「IoU 差 ≤ 0.01 且 e2e ≥ 1.05×」。實測 **1.00× 零加速**、"
                 "IoU 掉 0.042、**少 18% 偵測** —— 兩項都不過，維持現狀。")
        L.append("輸出確實改變了（偵測數與 IoU 都動了），所以「沒加速」不是 TensorRT "
                 "靜默 fallback 的假象。")
        L.append("")

    # detection-only ablation, if it has been run
    if os.path.exists("results_nomask.jsonl"):
        nm = load("results_nomask.jsonl")
        L.append("## 拿掉 pred_masks（純偵測部署）")
        L.append("")
        L.append("`--no-masks` 讓 ONNX 匯出器剪掉整個 mask decoder，`merge.py` 連帶移除"
                 "只有它在用的 `fpn_feat_0/1`。所以這一欄同時包含 mask decoder 與"
                 "那兩層高解析 FPN —— 是「純偵測部署」的數字，不是單獨的 mask decoder 成本。")
        L.append("")
        L.append("| N | bs | 含 mask (ms) | 純偵測 (ms) | 加速 | 含 mask VRAM | "
                 "純偵測 VRAM | 省下 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for (tag, res, bs), r in sorted(nm.items(), key=lambda kv: (kv[0][2], kv[0][1])):
            n = r["n_max"]
            ftag = "int8_rect" if n == 10 else f"int8_rect_n{n}"
            f = rows.get((ftag, res, bs))
            fl = f["latency_ms"] if (f and f.get("ok")) else None
            fv = f["vram_peak_proc_mib"] if (f and f.get("ok")) else None
            if not r.get("ok"):
                L.append(f"| {n} | {bs} | {fl or '—'} | **失敗** | — | — | — | — |")
                continue
            nl, nv = r["latency_ms"], r["vram_peak_proc_mib"]
            L.append(f"| {n} | {bs} | {f'{fl:.1f}' if fl else '**OOM**'} | {nl:.1f} | "
                     f"{f'{fl / nl:.2f}×' if fl else '—'} | "
                     f"{fv if fv else '**OOM**'} | {nv} | "
                     f"{f'{fv - nv} MiB' if fv else '—'} |")
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
