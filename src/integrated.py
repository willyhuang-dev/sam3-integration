#!/usr/bin/env python3
"""SAM3 with all three integrations applied at once.

Three independent pieces of work meet here:

  1. **dyntext** -- one forward, N concepts. Text arrives as a runtime input
     tensor, the vision backbone runs ONCE and its FPN features are shared
     across every concept slot.
  2. **ROI token pruning, strategy C + DETR dense** -- the letterbox black bar
     carries no information, so the ViT runs on only the top `rect_rows` token
     rows. No gather, no attention mask: the grid is simply shorter, so the
     official `window_partition` / global attention run unchanged and
     FlashAttention survives. The DETR encoder deliberately stays DENSE --
     pruning it costs 6.3% det50 for 1.02x, while pruning only the ViT costs
     0.9% for 1.62x (roi_token_pruning stage 11).
  3. **INT8 quantization** -- handled downstream by the ONNX/TensorRT scripts,
     but it is why this module is SPLIT into `IntegratedVE` and
     `MultiConceptHead`: monolithic ModelOpt calibration OOMs on 28 GB of host
     RAM, so the VE is calibrated alone and the two graphs are merged back into
     one engine afterwards.

The split boundary is the same one the quantization experiment used
(`encode_image`), so the VE graph stays structurally identical across
resolutions and the 644 calibration scales can be transplanted onto any of
them.

Graph contract
--------------
    IntegratedVE
        images        [B, 3, S, S]                 float32
          ->
        fpn_feat_0    [B, 256, 4g, 4g]             g = S // 14
        fpn_feat_1    [B, 256, 2g, 2g]
        fpn_feat_2    [B, 256,  g,  g]
        fpn_pos_2     [B, 256,  g,  g]

    MultiConceptHead
        fpn_feat_0/1/2, fpn_pos_2                  (as above)
        text_features [N_MAX, 32, 256]             float32  <- runtime input
        text_mask     [N_MAX, 32]                  int64    <- runtime input
          ->
        pred_boxes    [B, N_MAX, 200, 4]           normalised xyxy
        pred_logits   [B, N_MAX, 200]              pre-sigmoid
        presence_logits [B, N_MAX]                 pre-sigmoid
        pred_masks    [B, N_MAX, 200, M, M]        pre-sigmoid, M = g * 4

`presence_logits` is returned even though the dyntext export dropped it: the
verified SAM3 confidence is `sigmoid(pred_logits) * sigmoid(presence_logits)`,
and the tensor is [B, N] -- free to carry, impossible to reconstruct later.
"""
import math

import torch
from torch import nn

PATCH = 14        # ViT patch size; S must be a multiple of it
MASK_UPSCALE = 4  # mask side is (S // PATCH) * this
WINDOW = 24       # backbone_config.window_size, for the cost commentary only


def rect_rows_for(size, aspect_w=16, aspect_h=9):
    """Token rows covering the letterboxed content, black bar at the BOTTOM.

    Strategy C is only valid when the kept region is "row 0 upward, full
    width" -- that is exactly what an aspect-preserving letterbox of a wide
    frame into a square canvas gives you, provided the bar is at the bottom.
    Anything else (bars top AND bottom, pillarboxing) needs strategy A.

    Returns the FULL grid when the content already fills the canvas, which
    makes the pruning a no-op rather than an error.
    """
    grid = size // PATCH
    content_px = size * aspect_h / aspect_w
    return min(grid, int(math.ceil(content_px / PATCH)))


def load_model(model_id_or_dir, size, dtype=torch.float32):
    """Load the strategy-C fork of Sam3Model, reconfigured for `size`.

    The feature-map sizes are derived from image_size and must be set BEFORE
    from_pretrained, or the positional embeddings are built for the wrong grid
    -- which does not raise, it just produces wrong output (dyntext's note).

    Uses the vendored `modeling_sam3_sparse` fork, NOT `transformers.Sam3Model`:
    the stock class has no `rect_rows`. The fork also constructs its own
    `Sam3ViTModel` instead of going through `AutoModel.from_config`, because
    AutoModel resolves to the OFFICIAL ViT and the pruning path would then
    silently never run.
    """
    from transformers import Sam3Config

    from .modeling_sam3_sparse import Sam3Model

    cfg = Sam3Config.from_pretrained(model_id_or_dir)
    grid = size // PATCH
    cfg.vision_config.image_size = size
    cfg.vision_config.backbone_config.image_size = size
    cfg.vision_config.backbone_feature_sizes = [
        [grid * 4, grid * 4], [grid * 2, grid * 2], [grid, grid],
    ]
    model = Sam3Model.from_pretrained(model_id_or_dir, config=cfg, dtype=dtype).eval()
    assert type(model.vision_encoder.backbone).__module__.endswith(
        "modeling_sam3_sparse"), (
        "backbone is not the fork -- rect_rows would be silently ignored and "
        "every reconciliation test would pass while running the dense path")
    return model


class IntegratedVE(nn.Module):
    """images -> the three FPN levels the head needs, with strategy C applied.

    `rect_rows=None` is the dense path, kept as the honest baseline: the whole
    point of measuring is to compare against it, and the fork guarantees
    `rect_rows=None` is element-wise identical to stock SAM3.
    """

    def __init__(self, model, rect_rows=None):
        super().__init__()
        self.model = model
        self.rect_rows = rect_rows

    def forward(self, images):
        out = self.model.vision_encoder(
            images, roi=None, rect_rows=self.rect_rows)
        f = out.fpn_hidden_states
        p = out.fpn_position_encoding
        # Levels 0/1/2 only: Sam3Model.forward itself drops the last level
        # (`fpn_hidden_states[:-1]`), and only level 2's position encoding is
        # ever consumed (the DETR encoder's). Exporting the rest would bake
        # dead tensors into the engine.
        return f[0], f[1], f[2], p[2]


class MultiConceptHead(nn.Module):
    """FPN features + N concept vectors -> per-concept boxes/scores/masks.

    DO NOT "simplify" the repeat/tile pair below. `repeat_interleave` on the
    features and `repeat` (tile) on the text are what pair image b with concept
    n at flat row `b * N + n`; swapping either for the other silently pairs the
    wrong image with the wrong concept -- shapes stay valid, outputs are
    garbage, and no test that only checks shapes would notice.
    """

    def __init__(self, model, n_max):
        super().__init__()
        self.model = model
        self.n_max = n_max

    def forward(self, fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2,
                text_features, text_mask):
        from .modeling_sam3_sparse import Sam3VisionEncoderOutput

        n = self.n_max
        batch = fpn_feat_2.shape[0]

        f0 = fpn_feat_0.repeat_interleave(n, dim=0)
        f1 = fpn_feat_1.repeat_interleave(n, dim=0)
        f2 = fpn_feat_2.repeat_interleave(n, dim=0)
        p2 = fpn_pos_2.repeat_interleave(n, dim=0)
        tf = text_features.repeat(batch, 1, 1)
        tm = text_mask.repeat(batch, 1)

        # Sam3Model.forward takes fpn_hidden_states[:-1] and
        # fpn_position_encoding[:-1][-1], i.e. it reads levels 0/1/2 of the
        # features and ONLY level 2 of the position encoding. The 4th entry and
        # the unused position levels are placeholders, never read; passing p2
        # for them avoids allocating tensors that would show up in the graph.
        vision_out = Sam3VisionEncoderOutput(
            last_hidden_state=None,
            fpn_hidden_states=(f0, f1, f2, f2),
            fpn_position_encoding=(p2, p2, p2, p2),
        )
        out = self.model(
            vision_embeds=vision_out, input_ids=None, text_embeds=tf,
            attention_mask=tm, roi=None)

        # [B*N, ...] -> [B, N, ...]. -1 keeps the batch axis dynamic.
        boxes = out.pred_boxes.reshape(-1, n, *out.pred_boxes.shape[1:])
        logits = out.pred_logits.reshape(-1, n, *out.pred_logits.shape[1:])
        presence = out.presence_logits.reshape(-1, n)
        masks = out.pred_masks.reshape(-1, n, *out.pred_masks.shape[1:])
        return boxes, logits, presence, masks


class IntegratedSam3(nn.Module):
    """VE + head in one module -- the reference the split must reproduce.

    Only used to verify that splitting changed nothing (and for eager
    experiments). The exported artefact is always the split pair, because
    that is what the INT8 calibration needs.
    """

    def __init__(self, model, n_max, rect_rows=None):
        super().__init__()
        self.ve = IntegratedVE(model, rect_rows=rect_rows)
        self.head = MultiConceptHead(model, n_max)

    def forward(self, images, text_features, text_mask):
        f0, f1, f2, p2 = self.ve(images)
        return self.head(f0, f1, f2, p2, text_features, text_mask)
