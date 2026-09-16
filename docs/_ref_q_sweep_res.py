"""End-to-end mp_neck INT8 sweep for ONE resolution R (stride-14 valid).
Per R: export box+seg fp32 @R -> transplant 644 mp_neck scales -> quantize decoder @R ->
merge box(VE+dec) & seg(VE+head) -> build 4 engines (VE / decoder / merged-box / merged-seg) ->
benchmark GPU-compute median -> eval box+mask mAP (tomato) -> delete engines+big onnx.
Results appended as one JSON line to sweep_results.jsonl. Usage: sweep_res.py <R>"""
import os, sys, glob, time, subprocess, json, re, gc
import numpy as np, torch
os.environ.setdefault("TRANSFORMERS_VERBOSITY","error"); os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
HERE="/root/willy/models/pretrained_weights/sam3_huggingface"; CD=f"{HERE}/exp/quantization/int8fp8"
sys.path.insert(0, HERE)
R=int(sys.argv[1]); assert R%14==0, "res must be multiple of stride 14"
WORK=f"{CD}/sweep_work/res{R}"; os.makedirs(WORK, exist_ok=True)
ENGDIR=f"{HERE}/ckpts/tensorrt/sweep/res{R}"; os.makedirs(ENGDIR, exist_ok=True)
LOG=f"{CD}/sweep_results.jsonl"
def sh(cmd, timeout=3600):
    print(f"  $ {cmd}", flush=True)
    r=subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or ""), (r.stderr or "")
def logtail(rc,out,err,n=350):
    t=(out[-n:]+err[-n:]).strip()
    print("  "+t.replace("\n","\n  "), flush=True)
    return rc
def onnx_shapes(path):
    import onnx
    m=onnx.load(path, load_external_data=False)
    inits={i.name for i in m.graph.initializer}
    s={}
    for i in m.graph.input:
        if i.name in inits: continue
        dims=[]
        for d in i.type.tensor_type.shape.dim:
            dims.append(d.dim_value if (d.dim_value and d.dim_value>0) else 1)  # pin dynamic->1
        s[i.name]="x".join(str(x) for x in dims)
    return s
def shape_flags(path):
    s=onnx_shapes(path); spec=",".join(f"{k}:{v}" for k,v in s.items())
    return f"--minShapes={spec} --optShapes={spec} --maxShapes={spec}"
def build(onnx_path, eng_path):
    fl=shape_flags(onnx_path)
    rc,out,err=sh(f"trtexec --int8 --fp16 --onnx={onnx_path} {fl} --saveEngine={eng_path} --skipInference 2>&1")
    ok=os.path.exists(eng_path)
    print(f"  build {'OK' if ok else 'FAIL'} {os.path.basename(eng_path)} sz={os.path.getsize(eng_path)//(1024*1024) if ok else 0}MB", flush=True)
    if not ok: logtail(rc,out,err,600)
    return ok
def bench(eng_path):
    rc,out,err=sh(f"trtexec --loadEngine={eng_path} --iterations=200 --warmUp=300 --avgRuns=200 2>&1")
    m=re.search(r"GPU Compute Time:.*?median\s*=\s*([\d.]+)", out+err)
    return float(m.group(1)) if m else None

res={"res":R, "t_start":time.strftime("%H:%M:%S")}
t0=time.time()

# ---- 1) fp32 exports @R ----
box_ve_fp32=f"{HERE}/ckpts/onnx_opset=16/res={R}/sam3_ve.onnx"
dec_fp32   =f"{HERE}/ckpts/onnx_opset=16/res={R}/sam3_decoder.onnx"
seg_ve_fp32=f"{HERE}/ckpts/onnx_seg/res={R}/sam3_ve_seg.onnx"
seg_head   =f"{HERE}/ckpts/onnx_seg/res={R}/sam3_head_seg.onnx"
print(f"=== [{R}] export fp32 box+seg ===", flush=True)
if not os.path.exists(box_ve_fp32):
    logtail(*sh(f"python3 {CD}/export_split.py --image-size {R} --out-dir {HERE}/ckpts/onnx_opset=16/res={R}"))
if not os.path.exists(seg_ve_fp32):
    logtail(*sh(f"python3 {CD}/export_split_seg.py --image-size {R} --out-dir {HERE}/ckpts/onnx_seg/res={R}"))
assert os.path.exists(box_ve_fp32) and os.path.exists(seg_ve_fp32), "export failed"

# ---- 2) transplant 644 mp_neck scales onto R (box + seg) ----
box_ve_q=f"{WORK}/ve_box_{R}.mp_neck.onnx"
seg_ve_q=f"{WORK}/ve_seg_{R}.mp_neck.onnx"
print(f"=== [{R}] transplant mp_neck 644->{R} ===", flush=True)
logtail(*sh(f"python3 {CD}/transplant_seg.py {box_ve_fp32} {CD}/sam3_ve_644.mp_neck.onnx {box_ve_q}"))
logtail(*sh(f"python3 {CD}/transplant_seg.py {seg_ve_fp32} {CD}/sam3_ve_seg_644.mp_neck.onnx {seg_ve_q}"))
assert os.path.exists(box_ve_q) and os.path.exists(seg_ve_q), "transplant failed"

# ---- 3) quantize decoder @R (staged calib: PyTorch VE @R on real images) ----
dec_q=f"{WORK}/decoder_{R}.int8.onnx"
print(f"=== [{R}] quantize decoder @{R} ===", flush=True)
from export_onnx_no_text_no_geometry_no_seg import load_sam3_with_resolution, Sam3WithoutTextAndGeometryWrapper
from export_split import VEWrapper
from transformers import Sam3Processor
from PIL import Image
mdl,_=load_sam3_with_resolution(R); mdl.eval().to("cpu")
mono=Sam3WithoutTextAndGeometryWrapper(mdl, image_size=R).to("cpu").eval()
ve=VEWrapper(mono).to("cpu").eval()
proc=Sam3Processor.from_pretrained("facebook/sam3")
files=sorted(glob.glob(f"{HERE}/images/**/*.jpg", recursive=True))[:4]
tf=np.load(f"{HERE}/ckpts/text/tomato/text_features.npy").astype(np.float32)
tm=np.load(f"{HERE}/ckpts/text/tomato/attention_mask.npy").astype(np.float32)
feats=[]
import torch.nn.functional as Fn
with torch.inference_mode():
    for f in files:
        px=proc(images=Image.open(f).convert("RGB"),text="person",return_tensors="pt")["pixel_values"]
        px=Fn.interpolate(px, size=(R,R), mode="bilinear", align_corners=False)
        f2,pos2=ve(px); feats.append((f2.numpy().astype(np.float32),pos2.numpy().astype(np.float32)))
del mdl,mono,ve; gc.collect()
from onnxruntime.quantization import CalibrationDataReader
class DR(CalibrationDataReader):
    def __init__(s): s.i=0
    def _s(s,i):
        f2,pos2=feats[i]; return {"fpn_feat_2":f2,"fpn_pos_2":pos2,"text_features":tf,"text_mask":tm}
    def get_first(s): return s._s(0)
    def get_next(s):
        if s.i>=len(feats): return None
        d=s._s(s.i); s.i+=1; return d
    def rewind(s): s.i=0
import modelopt.onnx.quantization as moq
moq.quantize(onnx_path=dec_fp32, calibration_data_reader=DR(), quantize_mode="int8",
    calibration_method="max", calibration_eps=["cpu"], high_precision_dtype="fp32",
    disable_mha_qdq=True, mha_accumulation_dtype="fp32", use_external_data_format=True, output_path=dec_q)
assert os.path.exists(dec_q), "decoder quant failed"
print(f"  decoder quantized -> {dec_q}", flush=True)

# ---- 4) merge box (VE+dec) & seg (VE+head) ----
merged_box=f"{WORK}/merged_box_{R}.mp_neck.onnx"
merged_seg=f"{WORK}/merged_seg_{R}.mp_neck.onnx"
print(f"=== [{R}] merge box & seg ===", flush=True)
logtail(*sh(f"python3 {CD}/merge_v3.py {box_ve_q} {dec_q} {merged_box}"))
logtail(*sh(f"python3 {CD}/merge_seg.py {seg_ve_q} {seg_head} {merged_seg}"))
assert os.path.exists(merged_box) and os.path.exists(merged_seg), "merge failed"

# ---- 5) build 4 engines ----
print(f"=== [{R}] build engines ===", flush=True)
e_ve=f"{ENGDIR}/ve.engine"; e_dec=f"{ENGDIR}/dec.engine"
e_box=f"{ENGDIR}/merged_box.engine"; e_seg=f"{ENGDIR}/merged_seg.engine"
ok_ve=build(box_ve_q, e_ve)
ok_dec=build(dec_q, e_dec)
ok_box=build(merged_box, e_box)
ok_seg=build(merged_seg, e_seg)

# ---- 6) benchmark ----
print(f"=== [{R}] benchmark ===", flush=True)
res["ve_ms"]=bench(e_ve) if ok_ve else None
res["dec_ms"]=bench(e_dec) if ok_dec else None
res["e2e_box_ms"]=bench(e_box) if ok_box else None
res["e2e_seg_ms"]=bench(e_seg) if ok_seg else None
print(f"  VE={res['ve_ms']} dec={res['dec_ms']} e2e_box={res['e2e_box_ms']} e2e_seg={res['e2e_seg_ms']}", flush=True)

# ---- 7) eval box+mask mAP on the seg engine ----
print(f"=== [{R}] eval box+mask mAP ===", flush=True)
if ok_seg:
    env=dict(os.environ, ENG=e_seg, NAME=f"seg{R}")
    r=subprocess.run(f"python3 {CD}/eval_seg_res.py", shell=True, capture_output=True, text=True, env=env, timeout=1800)
    out=(r.stdout or "")+(r.stderr or "")
    print("  "+out[-800:].replace("\n","\n  "), flush=True)
    mb=re.search(r"BOX\s+AP@\.5=([\d.]+)\s+mAP@\[\.5:\.95\]=([\d.]+)", out)
    mm=re.search(r"MASK\s+AP@\.5=([\d.]+)\s+mAP@\[\.5:\.95\]=([\d.]+)", out)
    res["box_ap50"]=float(mb.group(1)) if mb else None
    res["box_map"]=float(mb.group(2)) if mb else None
    res["mask_ap50"]=float(mm.group(1)) if mm else None
    res["mask_map"]=float(mm.group(2)) if mm else None

res["mins"]=round((time.time()-t0)/60,1)
with open(LOG,"a") as f: f.write(json.dumps(res)+"\n")
print(f"=== [{R}] DONE {res['mins']}min ; results appended ===", flush=True)

# ---- 8) cleanup engines + big onnx for this res (keep 644 refs & jsonl) ----
sh(f"rm -rf {ENGDIR} {WORK} {HERE}/ckpts/onnx_opset=16/res={R} {HERE}/ckpts/onnx_seg/res={R}")
print(f"=== [{R}] cleaned engines+onnx ===", flush=True)
sys.stdout.flush(); os._exit(0)
