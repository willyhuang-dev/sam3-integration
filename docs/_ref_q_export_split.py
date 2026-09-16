"""
Split SAM3 (no_text_no_geometry_no_seg) into VE + Decoder ONNX, and validate
the chained VE->Decoder output matches the monolithic wrapper numerically.

Reuses Sam3WithoutTextAndGeometryWrapper from the existing monolithic export script.
"""
import argparse, sys, os
from pathlib import Path
import numpy as np
import torch

HERE = "/root/willy/models/pretrained_weights/sam3_huggingface"
sys.path.insert(0, HERE)
from export_onnx_no_text_no_geometry_no_seg import (
    load_sam3_with_resolution, Sam3WithoutTextAndGeometryWrapper,
)

class VEWrapper(torch.nn.Module):
    """images -> fpn_feat_2, fpn_pos_2   (no_seg only needs level-2 features)"""
    def __init__(self, mono):
        super().__init__()
        self.mono = mono
    def forward(self, images):
        f0, f1, f2, pos2 = self.mono.encode_image(images)
        return f2, pos2

class DecoderWrapper(torch.nn.Module):
    """fpn_feat_2, fpn_pos_2, text_features, text_mask -> pred_boxes, pred_logits, presence_logits"""
    def __init__(self, mono):
        super().__init__()
        self.mono = mono
    def forward(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        m = self.mono
        bs = fpn_feat_2.shape[0]
        text_feats = text_features.repeat(bs, 1, 1)
        text_mask_b = text_mask.repeat(bs, 1) == 1
        enc = m.detr_encoder(vision_features=[fpn_feat_2], text_features=text_feats,
                             vision_pos_embeds=[fpn_pos_2], text_mask=text_mask_b)
        dec = m.detr_decoder(vision_features=enc.last_hidden_state, text_features=enc.text_features,
                             vision_pos_encoding=enc.pos_embeds_flattened, text_mask=text_mask_b,
                             spatial_shapes=enc.spatial_shapes)
        all_box_offsets = m.box_head(dec.intermediate_hidden_states)
        ref_inv = m._inverse_sigmoid(dec.reference_boxes)
        all_pred_boxes = m._box_cxcywh_to_xyxy((ref_inv + all_box_offsets).sigmoid())
        all_pred_logits = m.dot_product_scoring(
            decoder_hidden_states=dec.intermediate_hidden_states,
            text_features=enc.text_features, text_mask=text_mask_b).squeeze(-1)
        return all_pred_boxes[-1], all_pred_logits[-1], dec.presence_logits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=1008)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=f"{HERE}/ckpts/onnx_opset=16/res=1008")
    args = ap.parse_args()
    dev = args.device
    S = args.image_size
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ve_path = out / "sam3_ve.onnx"
    dec_path = out / "sam3_decoder.onnx"

    model, _ = load_sam3_with_resolution(target_resolution=S)
    model.eval().to(dev)
    mono = Sam3WithoutTextAndGeometryWrapper(model, image_size=S).to(dev).eval()
    ve = VEWrapper(mono).to(dev).eval()
    dec = DecoderWrapper(mono).to(dev).eval()

    # real text features
    tf = torch.from_numpy(np.load(f"{HERE}/ckpts/text/gun/text_features.npy")).to(dev)   # [1,32,256]
    tm = torch.from_numpy(np.load(f"{HERE}/ckpts/text/gun/attention_mask.npy")).float().to(dev)  # [1,32]
    img = torch.randn(1, 3, S, S, device=dev)

    # ---- numerical validation: monolithic vs chained ----
    print("Validating split vs monolithic ...", flush=True)
    with torch.inference_mode():
        mb, ml, mp = mono(img, tf, tm)
        f2, pos2 = ve(img)
        db, dl, dp = dec(f2, pos2, tf, tm)
    for name, a, b in [("pred_boxes", mb, db), ("pred_logits", ml, dl), ("presence_logits", mp, dp)]:
        diff = (a.float() - b.float()).abs().max().item()
        print(f"  {name:16s} shape={tuple(a.shape)}  max|Δ|={diff:.3e}", flush=True)
    print(f"  VE outputs: fpn_feat_2={tuple(f2.shape)}  fpn_pos_2={tuple(pos2.shape)}", flush=True)

    # ---- export VE ----
    print("\nExporting VE ...", flush=True)
    torch.onnx.export(
        ve, (img,), str(ve_path),
        input_names=["images"], output_names=["fpn_feat_2", "fpn_pos_2"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
        dynamic_axes={"images": {0: "batch"}, "fpn_feat_2": {0: "batch"}, "fpn_pos_2": {0: "batch"}},
    )
    print(f"  saved {ve_path} ({ve_path.stat().st_size/2**20:.0f} MiB)", flush=True)

    # ---- export Decoder ----
    print("Exporting Decoder ...", flush=True)
    torch.onnx.export(
        dec, (f2, pos2, tf, tm), str(dec_path),
        input_names=["fpn_feat_2", "fpn_pos_2", "text_features", "text_mask"],
        output_names=["pred_boxes", "pred_logits", "presence_logits"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
        dynamic_axes={"fpn_feat_2": {0: "batch"}, "fpn_pos_2": {0: "batch"},
                      "pred_boxes": {0: "batch"}, "pred_logits": {0: "batch"}, "presence_logits": {0: "batch"}},
    )
    print(f"  saved {dec_path} ({dec_path.stat().st_size/2**20:.0f} MiB)", flush=True)
    print("\nSPLIT EXPORT DONE", flush=True)

if __name__ == "__main__":
    main()
