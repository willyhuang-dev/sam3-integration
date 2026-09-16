#!/usr/bin/env python3
"""SAM3 as a single-forward, multi-concept detector+segmenter.

Graph contract
--------------
    images        [1, 3, SIZE, SIZE]      float32
    text_features [N_MAX, 32, 256]        float32   <- runtime input, not baked
    text_mask     [N_MAX, 32]             int64     <- runtime input, not baked
      ->
    pred_boxes    [1, N_MAX, 200, 4]      float32   normalised cxcywh-free xyxy
    pred_logits   [1, N_MAX, 200]         float32   pre-sigmoid scores
    pred_masks    [1, N_MAX, 200, M, M]   float32   pre-sigmoid mask logits,
                                                    M = SIZE // 14 * 4

Two properties matter and are easy to lose if you re-derive this:

1. TEXT IS A RUNTIME INPUT, NOT A BAKED CONSTANT. Changing which concepts you
   detect costs nothing -- you swap two input tensors. No re-export, no engine
   rebuild. That is the entire point of the "dyntext" name.

2. MASKS COME FROM THE SAME FORWARD. SAM3 already computes mask logits during
   concept detection; the older export simply discarded them. Returning them
   costs no extra compute and removes the need for a second model/engine.

Both are covered by tests/test_model.py -- cheap enough to run on every change.
"""
import dataclasses
from types import SimpleNamespace

import torch

PATCH = 14        # ViT patch size; SIZE must be a multiple of it
MASK_UPSCALE = 4  # mask resolution is (SIZE // PATCH) * this


class MultiConceptSam3(torch.nn.Module):
    """Wraps Sam3Model so text arrives as tensors and masks are returned.

    DO NOT "simplify" this to a plain `self.model(pixel_values=..., text=...)`
    call. The structure below is what makes multi-concept detection cheap, and
    it is not obvious from the outside:

      * The vision encoder runs ONCE. Its FPN outputs are then repeated N_MAX
        times so a single image forward feeds every concept's detection head.
        Calling the model per concept instead would multiply the backbone cost
        by N_MAX -- the backbone is the overwhelming majority of the compute,
        so that is the difference between "N concepts are nearly free" and
        "N concepts cost N times as much".
      * Text enters as `text_embeds` carrying a pre-computed `pooler_output`,
        bypassing the tokeniser entirely. That is what lets text be a runtime
        INPUT TENSOR rather than something baked in at export.

    (Verified the hard way: a rewritten version that dropped the repeat trick
    and passed pixel_values directly failed with "You must specify exactly one
    of input_ids or text_embeds" -- and had it not failed, it would have
    silently exported a graph with completely different cost characteristics.)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, text_features, text_mask):
        vision_out = self.model.vision_encoder(pixel_values)
        n = text_features.shape[0]
        fpn_hidden = [t.repeat(n, *([1] * (t.dim() - 1)))
                      for t in vision_out.fpn_hidden_states]
        fpn_pos = [t.repeat(n, *([1] * (t.dim() - 1)))
                   for t in vision_out.fpn_position_encoding]
        vision_out = dataclasses.replace(
            vision_out, fpn_hidden_states=fpn_hidden, fpn_position_encoding=fpn_pos)
        text_out = SimpleNamespace(pooler_output=text_features)
        out = self.model(vision_embeds=vision_out, text_embeds=text_out,
                         attention_mask=text_mask)
        # unsqueeze(0): the model emits [N_MAX, ...]; the exported contract
        # carries an explicit batch dim so downstream code never has to guess
        # whether the leading axis is batch or concept.
        return (
            out.pred_boxes.unsqueeze(0),
            out.pred_logits.unsqueeze(0),
            out.pred_masks.unsqueeze(0),
        )


def load_model(model_dir, size):
    """Load Sam3Model reconfigured for `size`.

    The vision backbone's feature-map sizes are derived from image_size, and
    they must be set BEFORE from_pretrained or the positional embeddings are
    built for the wrong grid.

    This lives beside MultiConceptSam3 rather than in the exporter on purpose:
    the two together are what constitutes a usable model. Calling
    Sam3Model.from_pretrained() yourself and wrapping the result gets you a
    model whose positional embeddings are built for the wrong grid -- which
    does not raise, it just produces wrong output.
    """
    from transformers import Sam3Config, Sam3Model
    cfg = Sam3Config.from_pretrained(model_dir)
    grid = size // PATCH
    cfg.vision_config.image_size = size
    cfg.vision_config.backbone_config.image_size = size
    cfg.vision_config.backbone_feature_sizes = [
        [grid * 4, grid * 4], [grid * 2, grid * 2], [grid, grid],
    ]
    return Sam3Model.from_pretrained(model_dir, config=cfg, dtype=torch.float32).eval()
