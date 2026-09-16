"""Gates on the three properties the integration can silently lose.

Every one of these has a NEGATIVE CONTROL in mind -- a plausible wrong
implementation that keeps every shape valid and would pass a shape-only test:

  * rect_rows ignored          -> the pruning never runs, everything "passes"
    while the dense path is measured. (`test_rect_full_is_dense` pins the
    no-op end; `test_rect_crop_changes_output` pins that it is NOT a no-op
    when it should bite.)
  * repeat_interleave/repeat swapped in the head -> image b gets concept
    (b+n) mod N. Shapes identical, outputs garbage.
    (`test_concept_image_pairing`.)
  * backbone re-run per concept -> N concepts cost Nx, which is the whole
    point of dyntext. (`test_backbone_runs_once`.)
  * split != monolithic        -> the merged engine silently drifts from the
    reference. (`test_split_equals_monolithic`.)

Run inside the tensorrt container:
    python3 -m pytest tests/ -x -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrated import (  # noqa: E402
    PATCH, IntegratedSam3, IntegratedVE, MultiConceptHead, load_model,
    rect_rows_for,
)

MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SIZE = 644          # grid 46; smallest legal size that still exercises windows
N_MAX = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# pure arithmetic -- no model, runs anywhere
# --------------------------------------------------------------------------
def test_rect_rows_for_matches_the_published_table():
    # The table in README/docs; a change here changes every measured number,
    # so it is pinned rather than recomputed.
    assert [rect_rows_for(s) for s in (644, 728, 840, 924, 1008)] == [26, 30, 34, 38, 41]


def test_rect_rows_never_exceeds_the_grid():
    for s in (644, 728, 840, 924, 1008):
        assert rect_rows_for(s, aspect_w=1, aspect_h=1) == s // PATCH


# --------------------------------------------------------------------------
# model-backed
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_ID, SIZE).to(DEV)


@pytest.fixture(scope="module")
def text():
    torch.manual_seed(0)
    tf = torch.randn(N_MAX, 32, 256, device=DEV)
    tm = torch.zeros(N_MAX, 32, dtype=torch.int64, device=DEV)
    tm[:, :6] = 1
    return tf, tm


def _img(n=1, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, 3, SIZE, SIZE, generator=g).to(DEV)


def test_backbone_runs_once_regardless_of_concept_count(model, text):
    """N concepts must cost ONE backbone pass, not N."""
    tf, tm = text
    calls = []
    real = model.vision_encoder.forward

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    model.vision_encoder.forward = spy
    try:
        for n in (1, N_MAX):
            calls.clear()
            net = IntegratedSam3(model, n_max=n).eval()
            with torch.inference_mode():
                net(_img(), tf[:n], tm[:n])
            assert len(calls) == 1, f"n_max={n} ran the backbone {len(calls)}x"
    finally:
        model.vision_encoder.forward = real


def test_rect_full_is_dense(model):
    """rect_rows == grid must be a bit-exact no-op.

    If it is not, every pruned measurement is contaminated by an unrelated
    numerical change and the speedup cannot be attributed to pruning.
    """
    grid = SIZE // PATCH
    img = _img()
    with torch.inference_mode():
        dense = IntegratedVE(model, rect_rows=None).eval()(img)
        full = IntegratedVE(model, rect_rows=grid).eval()(img)
    for a, b in zip(dense, full):
        assert torch.equal(a, b), (a - b).abs().max().item()


def test_rect_crop_changes_output(model):
    """The negative control for the test above: cropping must actually bite.

    A `rect_rows` that is silently dropped would make `test_rect_full_is_dense`
    pass trivially -- this is what makes that test discriminating.
    """
    img = _img()
    with torch.inference_mode():
        dense = IntegratedVE(model, rect_rows=None).eval()(img)
        crop = IntegratedVE(model, rect_rows=rect_rows_for(SIZE)).eval()(img)
    assert not torch.equal(dense[2], crop[2])


def test_rect_crop_zeroes_the_black_bar_rows(model):
    """Strategy C pads the absent rows with ZEROS at the ViT exit.

    Verified in roi_token_pruning stage 9 to be the right choice (FPN convs use
    padding_mode='zeros', so zero is the signal they already see at every real
    image border). Pinning it here because `rect_fill` could be changed
    without any shape breaking.
    """
    k = rect_rows_for(SIZE)
    with torch.inference_mode():
        out = model.vision_encoder(_img(), roi=None, rect_rows=k)
    # last_hidden_state is [B, g*g, C] in row-major token order
    g = SIZE // PATCH
    lhs = out.last_hidden_state.view(1, g, g, -1)
    assert torch.count_nonzero(lhs[:, k:]) == 0
    assert torch.count_nonzero(lhs[:, :k]) > 0


def test_concept_image_pairing(model, text):
    """Image b must be paired with concept n at flat row b*N+n.

    This is the test that a swapped repeat_interleave/repeat fails. Two
    DIFFERENT images and two DIFFERENT concepts: run them batched, then run
    each (image, concept) alone.

    The threshold is NOT a fixed tolerance. Batching changes which cuBLAS
    tiling runs, so the same arithmetic drifts ~1e-4 -- a fixed atol either
    fails on that noise or is so loose it stops discriminating. Instead the
    test carries its own negative control: the MISMATCHED pairing is scored
    too, and the claim is that the right pairing is orders of magnitude
    closer. That ratio is what a swapped repeat/interleave destroys.
    """
    tf, tm = text
    imgs = torch.cat([_img(seed=1), _img(seed=2)])
    ve = IntegratedVE(model, rect_rows=None).eval()
    head2 = MultiConceptHead(model, n_max=2).eval()
    head1 = MultiConceptHead(model, n_max=1).eval()
    with torch.inference_mode():
        boxes, logits, presence, _ = head2(*ve(imgs), tf[:2], tm[:2])
        single = [[head1(*ve(imgs[b:b + 1]), tf[n:n + 1], tm[n:n + 1])
                   for n in range(2)] for b in range(2)]

    def err(b, n, sb, sn):
        s = single[sb][sn]
        return max((boxes[b, n] - s[0][0, 0]).abs().max().item(),
                   (logits[b, n] - s[1][0, 0]).abs().max().item(),
                   (presence[b, n] - s[2][0, 0]).abs().max().item())

    for b in range(2):
        for n in range(2):
            right = err(b, n, b, n)
            wrong = min(err(b, n, sb, sn) for sb in range(2) for sn in range(2)
                        if (sb, sn) != (b, n))
            assert right < 1e-2, f"image {b} x concept {n}: {right:.2e}"
            assert wrong > 20 * right, (
                f"image {b} x concept {n}: correct pairing {right:.2e} is not "
                f"distinguishable from the best wrong one {wrong:.2e} -- this "
                f"test has lost its discriminating power")


def test_text_is_a_runtime_input(model, text):
    """Swapping the text tensor must change the answer with no reload."""
    tf, tm = text
    net = IntegratedSam3(model, n_max=1).eval()
    with torch.inference_mode():
        a = net(_img(), tf[0:1], tm[0:1])[1]
        b = net(_img(), tf[1:2], tm[1:2])[1]
    assert not torch.allclose(a, b)


def test_split_equals_monolithic(model, text):
    """VE -> head chained must equal the single module, bit for bit."""
    tf, tm = text
    k = rect_rows_for(SIZE)
    net = IntegratedSam3(model, n_max=N_MAX, rect_rows=k).eval()
    ve = IntegratedVE(model, rect_rows=k).eval()
    head = MultiConceptHead(model, n_max=N_MAX).eval()
    img = _img()
    with torch.inference_mode():
        mono = net(img, tf, tm)
        split = head(*ve(img), tf, tm)
    for name, a, b in zip(("boxes", "logits", "presence", "masks"), mono, split):
        assert torch.equal(a, b), f"{name}: {(a - b).abs().max().item()}"


def test_output_shapes(model, text):
    tf, tm = text
    g, m = SIZE // PATCH, SIZE // PATCH * 4
    net = IntegratedSam3(model, n_max=N_MAX, rect_rows=rect_rows_for(SIZE)).eval()
    with torch.inference_mode():
        boxes, logits, presence, masks = net(_img(2), tf, tm)
    assert boxes.shape == (2, N_MAX, 200, 4)
    assert logits.shape == (2, N_MAX, 200)
    assert presence.shape == (2, N_MAX)
    assert masks.shape == (2, N_MAX, 200, m, m)
    assert g == 46
