# sam3-integration

三個獨立的 SAM3 開發事項合成一顆 TensorRT INT8 engine，並在 **RTX 5060 Ti
(sm_120)** 上量測 5 種解析度 × 5 種 batch size 的延遲與 VRAM。

| 來源 | 帶進來的東西 | 落在哪一段 |
|---|---|---|
| `sam3-dyntext-export` | 一次 forward 出 N 個概念；文字是**執行期輸入張量** | head |
| `sam3-roi-token-pruning` | 策略 C：把 letterbox 黑邊的 token 整列裁掉（**DETR 保持 dense**）| ViT |
| `sam3_quantization` | INT8 W8A8 + `mp_neck` 混精度（FPN neck 留 fp16）| VE |

```
frame (16:9)
  │  letterbox 到方形畫布，黑邊在下   ← src/preprocess.py（策略 C 的前提）
  ▼
images [B,3,S,S] ─┐
                  │  ViT 只跑前 rect_rows 列 token  ┐
                  │  FPN neck 留 fp16、其餘 INT8    ├─ VE（93% 算力）
                  ▼                                 ┘
        fpn_feat_0/1/2 + fpn_pos_2
                  │  repeat_interleave(N) × text tile(B)
                  ▼                                  ┐
text_features [N,32,256] ──► DETR enc/dec + mask head ├─ head（fp16）
text_mask     [N,32]     ──►                          ┘
                  ▼
   pred_boxes [B,N,200,4] · pred_logits [B,N,200]
   presence_logits [B,N] · pred_masks [B,N,200,M,M]
```

**換概念不需要重新匯出、也不需要重建 engine**——只換兩個輸入張量。只有 `--n-max`
（有幾個插槽）與 `--size` 是匯出期固定的。

---

## 為什麼是這個組合，而不是別的

三件事都有「看起來更好但實測更差」的變體，這裡採用的都是被量測選出來的那一個：

- **只剃 ViT、DETR 保持 dense。** 兩邊都剃是 1.66× 但掉 6.1% det50；只剃 ViT 是
  1.62× 只掉 0.9%。性價比差兩個數量級 —— 剃 DETR 幾乎不省時間卻付掉幾乎全部代價。
- **策略 C 而不是策略 A。** ROI 是「第 0 列起、整寬」的矩形時不需要 gather 也不需要
  遮罩，直接把 token grid 裁短、沿用官方 `window_partition`。**傳任何非 None 的
  `attn_mask` 會讓 PyTorch SDPA 掉出 FlashAttention**，策略 A 因此多付固定成本。
- **INT8 而不是 FP8 / NVFP4。** 消費級 Blackwell 的 FP8 tensor core 被 FP32-accumulate
  節流，實測只有 1.05×；INT8 沒被節流，1.49×。NVFP4 編得過但落在 INT8 同一個數字。
- **`mp_neck` 混精度。** 只把 2 個 FPN neck conv 留 fp16，**零速度成本**，shopee mAP
  保留率 97.3% → 99.4%。純 PTQ 的其他槓桿（SmoothQuant、entropy 校準、加校準資料）
  三個都實測無效。
- **letterbox 而不是 squash。** dyntext 原本是直接壓成正方形；策略 C 需要黑邊才有東西
  可剃。等比 letterbox vs 壓扁只差 −0.2% det50 —— SAM3 對長寬比失真比對解析度損失更敏感。

---

## 為什麼要切成 VE / head 兩張圖

推論要的是一顆 engine，最後也確實縫回一顆。切分純粹是因為 **ModelOpt 的 INT8 校準
會把約 331 個 activation 張量掛成 graph output 來收集極值**，在 ViT-H @1008 上這個
工作集超過本機 28 GB 主機記憶體、直接被 OOM killer 砍掉。只校準 VE 塞得下。

而且只有 VE 需要量化：它是 93% 的算力，decoder 量化只有 1.10×（BMM 太小、是
memory-bound，「能量化 ≠ 會變快」）。

各解析度不重新校準，而是**把 644 校準好的 Q/DQ 移植過去**：兩張圖結構完全相同，
只有宣告的維度不同。weight scale 與解析度無關（精確），activation scale 是 644 的值
（近似）—— 這個近似被量過：正規校準的 644 是 cos 0.879，移植到 1008 是 0.869，
box mAP 保留率 98.6–98.8%。

---

## 跑法

全部在 VM706 的 `tensorrt` 容器裡（`docker restart tensorrt` + `docker exec`，
**不要**另外 `docker run` 新容器）。

```bash
cd /root/willy/models/pretrained_weights/sam3_huggingface/exp/sam3-integration

python3 -m pytest tests/ -q                 # 10 個測試，含兩個負控制

python3 scripts/run_matrix.py --stage all   # export → 量化 → 移植 → 縫合 → 建 engine → 測
```

`run_matrix.py` 每一階段都會先檢查自己的產物、`results.jsonl` 是 append-only，
所以被砍掉可以直接重跑接續。

單獨一步：

```bash
python3 -m src.export_onnx --size 1008 --n-max 10 --out-dir out/res=1008
python3 -m src.verify_onnx --dir out/res=1008 --batches 1 2   # 動態 batch 沒被凍結
python3 -m src.quantize   --ve out/res=644/ve.onnx --out out/ve_644_rect.mp_neck.onnx --size 644
python3 -m src.transplant out/res=1008/ve.onnx out/ve_644_rect.mp_neck.onnx out/ve_int8_rect_1008.onnx
python3 -m src.merge      out/ve_int8_rect_1008.onnx out/res=1008/head.onnx out/int8_rect_1008.onnx
python3 -m src.bench      --onnx out/int8_rect_1008.onnx --res 1008 --bs 1 2 4 6 8 --tag int8_rect --int8
```

實際偵測（速度以外唯一能證明整合沒壞掉的東西）：

```bash
python3 -m src.run_engine --engine out/engines/int8_rect_r1008_b1.engine \
    --compare out/engines/fp16_rect_r1008_b1.engine \
    --image images/demo.jpg person "cardboard box"
```

---

## 每個解析度剃掉多少

16:9 內容、黑邊在下。`rect_rows = ceil(S·9/16 / 14)`。

| res | token grid | rect_rows | 保留 token | windowed 視窗列 |
|---|---|---|---|---|
| 644 | 46×46 | 26 | 56.5% | 2 / 2（**沒省到**）|
| 728 | 52×52 | 30 | 57.7% | 2 / 3 |
| 840 | 60×60 | 34 | 56.7% | 2 / 3 |
| 924 | 66×66 | 38 | 57.6% | 2 / 3 |
| 1008 | 72×72 | 41 | 56.9% | 2 / 3 |

⚠️ **644 是例外**：`window_size=24`，26 列與 46 列都要 pad 到 2 個視窗列，所以
28 層 windowed attention 一點都沒省，只有 4 層 global 與所有 ∝N 的 linear/MLP 省到。
其餘解析度 windowed 從 3 列降到 2 列。這是表裡唯一一個結構性的不連續，
解讀 644 的加速比時要記得。

---

## 測量結果

完整表格見 [`docs/report.md`](docs/report.md)（81 格，RTX 5060 Ti 16 GB）。摘要：

### 交付配置（INT8 + 策略 C、N=10 概念、bs=1）

| res | 延遲 | vs fp16 dense | VRAM | engine |
|---|---|---|---|---|
| 644 | 65.3 ms | 1.24× | 1604 MiB | 520 MiB（0.56×）|
| 728 | 86.8 ms | 1.37× | 1854 MiB | 520 MiB |
| 840 | 112.2 ms | 1.37× | 2242 MiB | 520 MiB |
| 924 | 136.1 ms | 1.35× | 2570 MiB | 519 MiB |
| 1008 | 162.5 ms | 1.34× | 2936 MiB | 520 MiB |

### ⭐ 整合買到多少：對 TRT FP16（無策略 C、有 dyntext）的 e2e 加速

兩邊都是 N 個概念單次 forward，所以這個比值只算「策略 C + INT8」買到的，
不把多概念共用 backbone 的好處算進來。50 格全部實測。

| res \ N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 644 | 1.64× | 1.54× | 1.46× | 1.41× | 1.36× | 1.31× | 1.29× | 1.27× | 1.25× | 1.24× |
| 728 | 1.97× | 1.83× | 1.72× | 1.64× | 1.58× | 1.50× | 1.46× | 1.44× | 1.40× | 1.37× |
| 840 | 2.09× | 1.90× | 1.76× | 1.66× | 1.59× | 1.53× | 1.48× | 1.44× | 1.40× | 1.37× |
| 924 | 2.08× | 1.87× | 1.74× | 1.64× | 1.57× | 1.50× | 1.45× | 1.41× | 1.38× | 1.35× |
| 1008 | **2.11×** | 1.88× | 1.74× | 1.64× | 1.57× | 1.50× | 1.45× | 1.41× | 1.37× | **1.34×** |

**加速隨 N 單調下降，從 ~2.1× 掉到 ~1.34×。** 這是 Amdahl 的直接體現：整合只加速 VE，
N 越大則完全沒被優化的 head 佔比越高。⇒ **概念數少時整合的價值最大**；
要在大 N 保住倍率，得動 head（見下面的拆解）。

⚠️ **644 整列最低，而且它的 baseline 從 644→728 是 39→66 ms（1.69×，遠超 token 比
1.28×）。兩者同源**——`window_size=24` 造成的量化效應：

| res | dense 視窗數 | 策略C 視窗數 | windowed attention 省下 |
|---|---|---|---|
| 644 | ⌈46/24⌉² = **4** | 2×2 = **4** | **0** |
| 728 | ⌈52/24⌉² = **9** | 2×3 = 6 | 1/3 |

644 的 dense 與 rect 都只需 4 個視窗，所以 28 層 windowed attention 一點都沒省；
728 的 dense 卻要 9 個。這同時解釋了「644 加速最低」與「baseline 在 644→728 超線性跳」。

### 四個會影響部署決策的結論

**1. batch 在這張卡上只買到 VRAM，買不到吞吐。** 每張影像的攤提延遲跨五個解析度、
五個 batch 全部持平（644 是 65.3→67.1 ms，1008 是 162.5→162.7 ms）。36 個 SM 在
bs=1 就已經飽和——N=10 個概念在 head 內部本來就等於 batch 10。要提高吞吐只能降
解析度或減概念數。

**2. 三個組件都兌現了各自宣稱的加速，而且幾乎完美相乘——但被 Amdahl 定律吃掉。**
head 在 fp16 基準與交付配置裡是**同一份沒被動過的 fp16 圖**，所以用 N 掃描量到的
head 成本一減，就能把 VE 分離出來（兩組獨立量測相減，不是迴歸）：

| res 1008 | 總延遲 | 減掉 head 125.4 ms | VE 加速 | 來源實驗宣稱 |
|---|---|---|---|---|
| fp16 dense | 218.0 | 92.6 ms | 1.00× | — |
| fp16 + 策略C | 178.6 | 53.2 ms | **1.74×** | ROI 實驗 1.62× |
| INT8 dense | 188.3 | 62.9 ms | **1.47×** | 量化實驗 1.53× |
| **INT8 + 策略C** | 162.5 | 37.1 ms | **2.50×** | 相乘 2.56× |

**VE 只佔 fp16 基準的 42%，所以 2.50× 的 VE 加速攤到端到端只剩 1.34×。**

**3. 超過 3~5 個概念，head 就比整顆（已剃已量化的）VE 還貴。** N 掃描在五個解析度
都是乾淨線性，`VE 固定成本 / 每概念成本` 就是損益平衡的概念數：

| res | VE 固定 | 每概念 | N=10 時 head 佔比 | 損益平衡 N |
|---|---|---|---|---|
| 644 | 19.5 ms | 4.63 ms | 70% | 4.2 |
| 728 | 27.3 | 5.98 | 69% | 4.6 |
| 840 | 30.5 | 8.16 | 73% | 3.7 |
| 924 | 33.6 | 10.22 | 75% | 3.3 |
| 1008 | 37.5 | 12.44 | 77% | 3.0 |

**4. 唯一到不了的格子（1008/bs=8）是 `pred_masks` 造成的，不是模型放不下。**
engine 只有 520 MiB 且 build 成功；執行時 TRT context 要 12064 MiB，再加
`pred_masks` 那顆 4.94 GiB 輸出就爆掉。拿掉 mask 之後同一格跑得起來：
**893.7 ms / 6908 MiB**。

### 下一步該優化哪裡（用消融量出來的，不是猜的）

`--no-masks` 讓匯出器剪掉整個 mask decoder，兩條線性擬合相減就得到 head 的內部拆解：

| 項目 | 成本 @1008 | 性質 |
|---|---|---|
| VE（含 fpn_feat_2） | 35.6 ms | 固定，N 個概念共用 |
| FPN 高解析層 0/1 | 1.6 ms | 固定，**只有 mask decoder 用** |
| **DETR enc/dec + scoring** | **7.62 ms / 概念** | 每概念成本的 **61%** |
| mask decoder | 4.91 ms / 概念 | 每概念成本的 39% |

⇒ **head 裡真正的大宗是 DETR，不是 mask decoder。** 量化實驗量過 DETR decoder
INT8 是 1.10×，當時 N=1、DETR 只佔 7%，所以判定不值得；現在 DETR 佔 N=10 端到端的
47%，同樣的 1.10× 就值得做了。

### 正確性驗證（速度證明不了整合沒壞掉）

`src/run_engine.py` 在真實照片上比對兩顆 engine（res 1008、概念 person/bottle/cup/chair）：
INT8 與 fp16 各偵測到 6 個人、**6/6 全部配對成功、平均 IoU 0.930、分數 MAE 0.071**。
文字作為執行期輸入、letterbox 座標還原、INT8 量化三者串起來都是對的。
⚠️ IoU 0.930 低於量化實驗當年的 0.995；我沒有隔離原因，但 letterbox 本身會放大 y
誤差（內容只佔畫布高度 56.25%，還原時 y 除以 0.5625 = 誤差乘 1.78 倍）——這是方案
固有的、與量化無關，但**尚未證實**是主因。

---

## 佈署時要知道的兩件事

- **不要把整個 `pred_masks` 抓回主機。** `[B,N,200,M,M]` float32，在 1008 / N=10 是
  **每個 batch element 663 MB**。先在小的 boxes/logits 上做門檻與 NMS，再只把活下來的
  `(slot, query)` mask 區塊拷回來。這個張量也是 batch 一大就撞上 16 GB 的主因。
- **不要對 mask 做 sigmoid。** 機率 ≥ 0.5 等價於 logit ≥ 0（逐位元驗證過），
  省掉整張 mask 的 `exp()`。

## 來源

`docs/_ref_*` 是三個來源專案的原始腳本，留著做對照與出處：
`sam3-dyntext-export`、`sam3_roi_token_pruning`、`sam3_quantization`
（後兩者為 CrystalCoreAI private repo）。權重 `facebook/sam3` 是 gated，本 repo 不含
任何權重、engine 或資料集。
