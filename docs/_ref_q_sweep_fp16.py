"""fp16 e2e baseline sweep for ONE res: export box+seg fp32 -> merge fp16 (box:VE+dec, seg:VE+head)
-> build --fp16 engines -> benchmark GPU-compute median + record engine size -> delete.
Appends one JSON line to sweep_fp16.jsonl. Usage: sweep_fp16.py <R>"""
import os, sys, time, subprocess, json, re
HERE="/root/willy/models/pretrained_weights/sam3_huggingface"; CD=f"{HERE}/exp/quantization/int8fp8"
R=int(sys.argv[1]); assert R%14==0
WORK=f"{CD}/fp16_work/res{R}"; os.makedirs(WORK, exist_ok=True)
ENGDIR=f"{HERE}/ckpts/tensorrt/fp16sweep/res{R}"; os.makedirs(ENGDIR, exist_ok=True)
def sh(cmd,t=3600):
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=t); return r.returncode,(r.stdout or ""),(r.stderr or "")
def tail(rc,out,err,n=300): print("  "+(out[-n:]+err[-n:]).strip().replace("\n","\n  "),flush=True); return rc
def onnx_shapes(path):
    import onnx
    m=onnx.load(path,load_external_data=False); inits={i.name for i in m.graph.initializer}; s={}
    for i in m.graph.input:
        if i.name in inits: continue
        dims=[d.dim_value if (d.dim_value and d.dim_value>0) else 1 for d in i.type.tensor_type.shape.dim]
        s[i.name]="x".join(map(str,dims))
    return s
def flags(path):
    spec=",".join(f"{k}:{v}" for k,v in onnx_shapes(path).items()); return f"--minShapes={spec} --optShapes={spec} --maxShapes={spec}"
def build_fp16(onnx_path,eng):
    rc,out,err=sh(f"trtexec --fp16 --onnx={onnx_path} {flags(onnx_path)} --saveEngine={eng} --skipInference 2>&1")
    ok=os.path.exists(eng); print(f"  build {'OK' if ok else 'FAIL'} {os.path.basename(eng)} {os.path.getsize(eng)//(1024*1024) if ok else 0}MB",flush=True)
    if not ok: tail(rc,out,err,600)
    return ok
def bench(eng):
    rc,out,err=sh(f"trtexec --loadEngine={eng} --iterations=200 --warmUp=300 --avgRuns=200 2>&1")
    m=re.search(r"GPU Compute Time:.*?median\s*=\s*([\d.]+)",out+err); return float(m.group(1)) if m else None

res={"res":R}; t0=time.time()
box_ve=f"{HERE}/ckpts/onnx_opset=16/res={R}/sam3_ve.onnx"; dec=f"{HERE}/ckpts/onnx_opset=16/res={R}/sam3_decoder.onnx"
seg_ve=f"{HERE}/ckpts/onnx_seg/res={R}/sam3_ve_seg.onnx"; head=f"{HERE}/ckpts/onnx_seg/res={R}/sam3_head_seg.onnx"
print(f"=== [{R}] export fp32 ===",flush=True)
if not os.path.exists(box_ve): tail(*sh(f"python3 {CD}/export_split.py --image-size {R} --out-dir {HERE}/ckpts/onnx_opset=16/res={R}"))
if not os.path.exists(seg_ve): tail(*sh(f"python3 {CD}/export_split_seg.py --image-size {R} --out-dir {HERE}/ckpts/onnx_seg/res={R}"))
assert os.path.exists(box_ve) and os.path.exists(seg_ve)
mbox=f"{WORK}/merged_box_fp16.onnx"; mseg=f"{WORK}/merged_seg_fp16.onnx"
print(f"=== [{R}] merge fp32 (box & seg) ===",flush=True)
tail(*sh(f"python3 {CD}/merge_v3.py {box_ve} {dec} {mbox}"))
tail(*sh(f"python3 {CD}/merge_seg.py {seg_ve} {head} {mseg}"))
assert os.path.exists(mbox) and os.path.exists(mseg)
print(f"=== [{R}] build fp16 engines ===",flush=True)
ebox=f"{ENGDIR}/merged_box_fp16.engine"; eseg=f"{ENGDIR}/merged_seg_fp16.engine"
okb=build_fp16(mbox,ebox); oks=build_fp16(mseg,eseg)
res["e2e_box_fp16_ms"]=bench(ebox) if okb else None
res["e2e_seg_fp16_ms"]=bench(eseg) if oks else None
res["box_fp16_MB"]=os.path.getsize(ebox)//(1024*1024) if okb else None
res["seg_fp16_MB"]=os.path.getsize(eseg)//(1024*1024) if oks else None
res["mins"]=round((time.time()-t0)/60,1)
print(f"  box_fp16={res['e2e_box_fp16_ms']}ms seg_fp16={res['e2e_seg_fp16_ms']}ms sizes {res['box_fp16_MB']}/{res['seg_fp16_MB']}MB",flush=True)
with open(f"{CD}/sweep_fp16.jsonl","a") as f: f.write(json.dumps(res)+"\n")
sh(f"rm -rf {ENGDIR} {WORK} {HERE}/ckpts/onnx_opset=16/res={R} {HERE}/ckpts/onnx_seg/res={R}")
print(f"=== [{R}] fp16 DONE {res['mins']}min ; cleaned ===",flush=True)
sys.stdout.flush(); os._exit(0)
