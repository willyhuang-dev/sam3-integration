"""Merge VE_seg + Head_seg into one Q/DQ ONNX. VE_seg outputs fpn_feat_0/1/2 + fpn_pos_2 ->
Head_seg inputs of same name (4 connections). Handles opset align + IR + name-collision prefix.
Usage: merge_seg.py <ve.onnx> <head.onnx> <out.onnx>"""
import onnx, sys, time, os
from onnx import compose, version_converter
VE, HEAD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
t0=time.time()
ve=onnx.load(VE, load_external_data=True); head=onnx.load(HEAD, load_external_data=True)
def op(m): return [o for o in m.opset_import if o.domain in ("","ai.onnx")][0].version
vop, hop = op(ve), op(head)
print(f"VE opset {vop} ir {ve.ir_version} ; HEAD opset {hop} ir {head.ir_version}", flush=True)
target=max(vop,hop)
if vop<target: print(f"convert VE {vop}->{target}",flush=True); ve=version_converter.convert_version(ve,target)
if hop<target: print(f"convert HEAD {hop}->{target}",flush=True); head=version_converter.convert_version(head,target)
head.ir_version=ve.ir_version=max(ve.ir_version,head.ir_version)
head2=compose.add_prefix(head, prefix="h/")
io=[(f"fpn_feat_{i}",f"h/fpn_feat_{i}") for i in (0,1,2)]+[("fpn_pos_2","h/fpn_pos_2")]
merged=compose.merge_models(ve, head2, io_map=io)
print("merge OK; inputs:", [i.name for i in merged.graph.input], flush=True)
print("outputs:", [o.name for o in merged.graph.output], flush=True)
onnx.save(merged, OUT, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(OUT)+"_data", size_threshold=1024)
print(f"MERGED -> {OUT} in {time.time()-t0:.1f}s", flush=True)
