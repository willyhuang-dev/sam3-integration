#!/usr/bin/env python3
"""The one definition of how a frame becomes `images`.

Strategy C is only correct if the preprocessing actually puts the black bar
where the pruning assumes it is. That makes this file part of the model, not a
convenience: calibration, the runner and any evaluation must all go through it,
or the engine is measured on one distribution and calibrated on another.

Contract: aspect-preserving resize to the canvas WIDTH, pasted at the TOP-LEFT,
bottom padded with black. Not centred -- centring would put a bar above the
content and strategy C only prunes from the bottom.

This differs from the dyntext export, which squash-resized to a square. Squashing
leaves no bar to prune, and the ROI experiment measured that letterboxing is
close to free anyway: aspect-preserving vs squashed cost only -0.2% det50, and
SAM3 turned out to be MORE sensitive to aspect distortion than to the
resolution lost to the bar.

Normalisation is SAM3's own: mean = std = 0.5 on [0,1], i.e. x/127.5 - 1
(read off Sam3Processor rather than assumed).
"""
import numpy as np

MEAN = 0.5
STD = 0.5


def letterbox(rgb, size, aspect_w=16, aspect_h=9):
    """uint8 HxWx3 RGB -> (uint8 size x size x 3 canvas, content_rows_px).

    The frame is first centre-cropped to `aspect_w:aspect_h` so that a source
    of any shape lands on the same content/bar split the engine was built for.
    A 1920x1080 frame is already 16:9 and is cropped by nothing.
    """
    h, w = rgb.shape[:2]
    want = aspect_w / aspect_h
    if w / h > want:                      # too wide -> trim the sides
        new_w = int(round(h * want))
        x0 = (w - new_w) // 2
        rgb = rgb[:, x0:x0 + new_w]
    elif w / h < want:                    # too tall -> trim top and bottom
        new_h = int(round(w / want))
        y0 = (h - new_h) // 2
        rgb = rgb[y0:y0 + new_h]

    import cv2
    content_h = int(round(size * aspect_h / aspect_w))
    resized = cv2.resize(rgb, (size, content_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:content_h] = resized
    return canvas, content_h


def to_tensor(canvas):
    """uint8 HxWx3 -> float32 [1,3,H,W] normalised the way SAM3 expects."""
    x = canvas.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def prepare(rgb, size, aspect_w=16, aspect_h=9):
    canvas, content_h = letterbox(rgb, size, aspect_w, aspect_h)
    return to_tensor(canvas), content_h
