"""Split the SEG SAM3 into VE_seg (images -> 3 fpn levels + pos) and Head_seg
(fpn + text -> pred_masks, pred_boxes, pred_logits, presence_logits), validate chained==monolithic,
export both ONNX. Foundation for a single quantized engine that outputs box & mask."""
import argparse, sys, os
from pathlib import Path
import numpy as np, torch
HERE = "/root/willy/models/pretrained_weights/sam3_huggingface"
sys.path.insert(0, HERE)
from export_onnx_no_text_no_geometry import load_sam3_with_resolution, Sam3WithoutTextAndGeometryWrapper

class VEWrapperSeg(torch.nn.Module):
    """images -> fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2 (seg needs all 3 levels for masks)"""
    def __init__(self, mono): super().__init__(); self.mono = mono
    def forward(self, images):
        f0, f1, f2, pos2 = self.mono.encode_image(images)
        return f0, f1, f2, pos2

class HeadWrapperSeg(torch.nn.Module):
    """fpn_feat_0/1/2, fpn_pos_2, text -> pred_masks, pred_boxes, pred_logits, presence_logits"""
    def __init__(self, mono): super().__init__(); self.mono = mono
    def forward(self, fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2, text_features, text_mask):
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
        all_pred_logits = m.dot_product_scoring(decoder_hidden_states=dec.intermediate_hidden_states,
                             text_features=enc.text_features, text_mask=text_mask_b).squeeze(-1)
        pred_logits = all_pred_logits[-1]; pred_boxes = all_pred_boxes[-1]
        dhs = dec.intermediate_hidden_states[-1]; presence = dec.presence_logits[-1]
        mo = m.mask_decoder(decoder_queries=dhs, backbone_features=[fpn_feat_0, fpn_feat_1, fpn_feat_2],
                            encoder_hidden_states=enc.last_hidden_state, prompt_features=text_feats, prompt_mask=text_mask_b)
        return mo.pred_masks, pred_boxes, pred_logits, presence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=1008)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=f"{HERE}/ckpts/onnx_seg/res=1008")
    args = ap.parse_args()
    dev, S = args.device, args.image_size
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, _ = load_sam3_with_resolution(target_resolution=S); model.eval().to(dev)
    mono = Sam3WithoutTextAndGeometryWrapper(model, image_size=S).to(dev).eval()
    ve = VEWrapperSeg(mono).to(dev).eval(); head = HeadWrapperSeg(mono).to(dev).eval()
    tf = torch.from_numpy(np.load(f"{HERE}/ckpts/text/gun/text_features.npy")).to(dev)
    tm = torch.from_numpy(np.load(f"{HERE}/ckpts/text/gun/attention_mask.npy")).float().to(dev)
    img = torch.randn(1, 3, S, S, device=dev)
    print("validating split==monolithic ...", flush=True)
    with torch.inference_mode():
        mm, mb, ml, mp = mono(img, tf, tm)
        f0, f1, f2, pos2 = ve(img)
        hm, hb, hl, hpp = head(f0, f1, f2, pos2, tf, tm)
    for n, a, b in [("pred_masks", mm, hm), ("pred_boxes", mb, hb), ("pred_logits", ml, hl), ("presence", mp, hpp)]:
        print(f"  {n:14s} shape={tuple(a.shape)} max|Δ|={(a.float()-b.float()).abs().max().item():.3e}", flush=True)
    print(f"  VE outs: f0={tuple(f0.shape)} f1={tuple(f1.shape)} f2={tuple(f2.shape)} pos2={tuple(pos2.shape)}", flush=True)
    print("exporting VE_seg ...", flush=True)
    torch.onnx.export(ve, (img,), str(out/"sam3_ve_seg.onnx"),
        input_names=["images"], output_names=["fpn_feat_0","fpn_feat_1","fpn_feat_2","fpn_pos_2"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
        dynamic_axes={"images":{0:"batch"},"fpn_feat_0":{0:"batch"},"fpn_feat_1":{0:"batch"},"fpn_feat_2":{0:"batch"},"fpn_pos_2":{0:"batch"}})
    print(f"  saved sam3_ve_seg.onnx ({(out/'sam3_ve_seg.onnx').stat().st_size/2**20:.0f} MiB)", flush=True)
    print("exporting Head_seg ...", flush=True)
    torch.onnx.export(head, (f0,f1,f2,pos2,tf,tm), str(out/"sam3_head_seg.onnx"),
        input_names=["fpn_feat_0","fpn_feat_1","fpn_feat_2","fpn_pos_2","text_features","text_mask"],
        output_names=["pred_masks","pred_boxes","pred_logits","presence_logits"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
        dynamic_axes={"fpn_feat_0":{0:"batch"},"fpn_feat_1":{0:"batch"},"fpn_feat_2":{0:"batch"},"fpn_pos_2":{0:"batch"},
                      "pred_masks":{0:"batch"},"pred_boxes":{0:"batch"},"pred_logits":{0:"batch"},"presence_logits":{0:"batch"}})
    print(f"  saved sam3_head_seg.onnx ({(out/'sam3_head_seg.onnx').stat().st_size/2**20:.0f} MiB)", flush=True)
    print("SPLIT SEG EXPORT DONE", flush=True)

if __name__ == "__main__": main()
