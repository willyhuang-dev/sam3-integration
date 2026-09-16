"""Merge a quantized VE ONNX + quantized decoder ONNX into ONE Q/DQ ONNX (argv: ve dec out).
Handles: opset align (version_converter to decoder's opset), IR align, name-collision prefix."""
import onnx, sys, time, os
from onnx import compose, version_converter
VE, DEC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
t0=time.time()
ve=onnx.load(VE, load_external_data=True); dec=onnx.load(DEC, load_external_data=True)
vop=[o for o in ve.opset_import if o.domain in ("","ai.onnx")][0].version
dop=[o for o in dec.opset_import if o.domain in ("","ai.onnx")][0].version
print(f"VE opset {vop} ir {ve.ir_version} ; DEC opset {dop} ir {dec.ir_version}", flush=True)
if vop < dop:
    print(f"version_converter VE {vop} -> {dop} ...", flush=True)
    ve=version_converter.convert_version(ve, dop)
ve.ir_version=dec.ir_version
dec2=compose.add_prefix(dec, prefix="dec/")
merged=compose.merge_models(ve, dec2, io_map=[("fpn_feat_2","dec/fpn_feat_2"),("fpn_pos_2","dec/fpn_pos_2")])
print("merge OK; inputs:", [i.name for i in merged.graph.input], "outputs:", [o.name for o in merged.graph.output], flush=True)
onnx.save(merged, OUT, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(OUT)+"_data", size_threshold=1024)
print(f"MERGED -> {OUT} in {time.time()-t0:.1f}s", flush=True)
