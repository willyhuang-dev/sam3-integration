# I/O specification

Everything you need to drive the exported graph from your own code. `SIZE` is
whatever you passed to `--size`; `N_MAX` likewise. Only two derived numbers
matter:

```
M      = SIZE // 14 * 4        # mask side length: 644 -> 184, 1008 -> 288
QUERY  = 200                   # detection queries per concept slot (fixed)
```

## Inputs

| name | shape | dtype | notes |
|---|---|---|---|
| `images` | `[1, 3, SIZE, SIZE]` | float32 | RGB, normalised to ~[-1, 1] (`x/127.5 - 1`). **Plain resize, no letterbox** — see coordinate mapping below. |
| `text_features` | `[N_MAX, 32, 256]` | float32 | one 32×256 block per concept slot |
| `text_mask` | `[N_MAX, 32]` | int64 | 1 = real token, 0 = padding |

**Shapes are fixed at export.** There are no dynamic axes, so a graph exported
at 644 cannot accept 1008. Re-export to change either number.

**Unused slots must be zero.** If you have 3 concepts and `N_MAX=10`, fill slots
0–2 and leave 3–9 as zeros in *both* tensors. Zeroed slots emit nothing.

## Outputs

| name | shape | dtype |
|---|---|---|
| `pred_boxes` | `[1, N_MAX, QUERY, 4]` | float32 |
| `pred_logits` | `[1, N_MAX, QUERY]` | float32 |
| `pred_masks` | `[1, N_MAX, QUERY, M, M]` | float32 |

All three are **pre-sigmoid**. Index `[0, slot, q]` is the q-th candidate for
the concept you loaded into `slot`.

`pred_boxes` are **normalised xyxy in [0,1]**, relative to the input crop.

## Where text vectors come from

Two options; they produce identical vectors.

**Pre-computed (`concept_library/`)** — 21 concepts ship with this project as
`.npz` files, each holding `text_features[1,32,256]` and `text_mask[1,32]`.
Loading one is a file read: no model, no GPU, no 6.5 GB of weights.

```python
import text_library as TL
tf, tm, slots = TL.assemble("concept_library", ["person", "cardboard box"], n_max=10)
# tf: (10,32,256) float32   tm: (10,32) int64   slots: names per index
```

**On demand** — for a word not in the library, run SAM3's text tower. This needs
the full weights, but it is **CPU-only and takes about a second**; it never
touches the GPU:

```python
from transformers import Sam3Processor, Sam3Model
proc  = Sam3Processor.from_pretrained(MODEL_DIR)
model = Sam3Model.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
tin = proc(text=["forklift"], return_tensors="pt", padding="max_length", max_length=32)
with torch.no_grad():
    out = model.get_text_features(input_ids=tin["input_ids"],
                                  attention_mask=tin["attention_mask"], return_dict=True)
tf_one = out.pooler_output.numpy().astype("float32")   # (1,32,256)
tm_one = tin["attention_mask"].numpy().astype("int64") # (1,32)
```

Cache the result — encoding the same word twice is pure waste.

## Post-processing

### 1. Scores — and one free optimisation

```python
scores = sigmoid(pred_logits[0, slot])        # (QUERY,)
keep   = scores >= threshold                  # 0.35–0.7 depending on your scene
```

For **masks**, do NOT compute sigmoid. Thresholding a mask at probability ≥ 0.5
is exactly equivalent to thresholding its logit at ≥ 0, because sigmoid is
monotonic and `sigmoid(0) = 0.5`:

```python
mask_bool = mask_logits >= 0.0        # identical result, no exp() at all
```

This was verified bit-for-bit over 50 random tensors. On a 288×288 mask it turns
a ~0.06 ms operation into ~0.001 ms, per detection.

### 2. NMS
Boxes are per-slot; run NMS **within** each slot (IoU ≈ 0.85). Different
concepts legitimately overlap — a hand on a box is not a duplicate — so do not
NMS across slots.

### 3. Masks — the one thing that will wreck your throughput

`pred_masks` in full is **`N_MAX × QUERY × M × M × 4` bytes**:

| SIZE | M | full tensor |
|---|---|---|
| 644 | 184 | **270 MB** |
| 1008 | 288 | **663 MB** |

Copying that to host every frame is not viable. **Do not fetch the whole
tensor.** Threshold and NMS first using the small `pred_boxes` / `pred_logits`,
then copy back only the `(slot, query)` blocks that survived — typically 20–50
of the 2000, at 135 KB (644) or 331 KB (1008) each.

The block for `(slot, q)` starts at element offset:

```
((slot * QUERY) + q) * M * M
```

With PyCUDA that is a direct offset on the device pointer:

```python
cuda.memcpy_dtoh(host_MxM, int(device_ptr) + offset_elems * 4)
```

`server/runner.py` in the demo repo is a working reference for this.

### 4. Mask → frame coordinates

`preprocess` is a **plain squash-resize with no letterbox**, so mask space, box
space and the source frame share one linear map. Given a crop at `(x0, y0)` of
size `(cw, ch)`:

```python
sx, sy = cw / M, ch / M
frame_x = mask_x * sx + x0
frame_y = mask_y * sy + y0
```

Scale the **contour points**, not the mask image — extract contours at M×M
(~34 k or ~83 k px) and map the handful of resulting points. Upsampling every
mask to frame resolution first is orders of magnitude more expensive for an
identical outcome.

⚠️ If you use a minimum-area filter when extracting contours, **scale it**.
Areas found at M×M are in low-res px²; one such pixel covers `sx*sy` frame px²
(~60 on a 1920×1080 frame at 644). An unscaled threshold silently discards small
but genuine detections.

## Cost model

Fixed at export:
- `N_MAX` — concept slots. Raising it costs little: the mask head is a small
  fraction of total compute.
- `SIZE` — **this is the expensive dial.** Token count grows as `(SIZE/14)²`,
  and attention is quadratic in tokens: 644 → 1008 is 2.45× the pixels but
  roughly 2.5–6× the compute.

Free at runtime:
- **which** concepts, and how many of the `N_MAX` slots you fill. Text is an
  input tensor, not a baked constant. Swapping concepts needs no re-export and
  no engine rebuild — that is the entire point of this export.
