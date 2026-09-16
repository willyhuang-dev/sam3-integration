"""Rebuild the 644 mp_neck transplant references (box VE + seg VE), neck excluded (=deployed config).
Steps: export fp32 box VE+decoder @644, export fp32 seg VE+head @644, then INT8 mp_neck quantize
both VE graphs @644 (calibration_method=max, disable_mha_qdq, nodes_to_exclude=/neck/.*)."""
import os, sys, glob, time, subprocess
import numpy as np, torch, torch.nn.functional as Fn
os.environ.setdefault("TRANSFORMERS_VERBOSITY","error"); os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
HERE="/root/willy/models/pretrained_weights/sam3_huggingface"; CD=f"{HERE}/exp/quantization/int8fp8"
sys.path.insert(0, HERE)

def run(cmd):
    print("  $ "+" ".join(cmd), flush=True)
    r=subprocess.run(cmd, capture_output=True, text=True)
    tail=(r.stdout or "")[-400:]+(r.stderr or "")[-400:]
    print("  "+tail.replace("\n","\n  "), flush=True)
    if r.returncode!=0: print(f"  !! rc={r.returncode}", flush=True)
    return r.returncode

t0=time.time()
# 1) fp32 exports @644 (skip if present)
box_ve=f"{HERE}/ckpts/onnx_opset=16/res=644/sam3_ve.onnx"
seg_ve=f"{HERE}/ckpts/onnx_seg/res=644/sam3_ve_seg.onnx"
if not os.path.exists(box_ve):
    print("=== export box VE+decoder @644 ===", flush=True)
    run(["python3", f"{CD}/export_split.py", "--image-size","644","--out-dir",f"{HERE}/ckpts/onnx_opset=16/res=644"])
else: print("box VE @644 exists, skip", flush=True)
if not os.path.exists(seg_ve):
    print("=== export seg VE+head @644 ===", flush=True)
    run(["python3", f"{CD}/export_split_seg.py", "--image-size","644","--out-dir",f"{HERE}/ckpts/onnx_seg/res=644"])
else: print("seg VE @644 exists, skip", flush=True)

# 2) calibration images (same recipe as prior scripts: real images -> 644)
from transformers import Sam3Processor
from PIL import Image
proc=Sam3Processor.from_pretrained("facebook/sam3")
files=sorted(glob.glob(f"{HERE}/images/**/*.jpg", recursive=True))[:4]
imgs=[Fn.interpolate(torch.from_numpy(proc(images=Image.open(f).convert("RGB"),text="person",return_tensors="pt")["pixel_values"].numpy().astype(np.float32)),size=(644,644),mode="bilinear",align_corners=False).numpy() for f in files]
print(f"calib {len(imgs)} x {imgs[0].shape}", flush=True)
from onnxruntime.quantization import CalibrationDataReader
class R(CalibrationDataReader):
    def __init__(s): s.i=0
    def get_first(s): return {"images": imgs[0]}
    def get_next(s):
        if s.i>=len(imgs): return None
        d={"images": imgs[s.i]}; s.i+=1; return d
    def rewind(s): s.i=0
import modelopt.onnx.quantization as moq, onnx
def qcount(p): return sum(1 for n in onnx.load(p,load_external_data=False).graph.node if n.op_type=="QuantizeLinear")

# 3) quantize box VE @644 mp_neck
box_out=f"{CD}/sam3_ve_644.mp_neck.onnx"
print("=== quantize BOX VE @644 mp_neck (neck excluded) ===", flush=True)
tq=time.time()
moq.quantize(onnx_path=box_ve, calibration_data_reader=R(), quantize_mode="int8",
    calibration_method="max", calibration_eps=["cpu"], high_precision_dtype="fp32",
    disable_mha_qdq=True, mha_accumulation_dtype="fp32", nodes_to_exclude=["/neck/.*"],
    use_external_data_format=True, output_path=box_out)
print(f"  box mp_neck done {time.time()-tq:.0f}s QuantizeLinear={qcount(box_out)} -> {box_out}", flush=True)

# 4) quantize seg VE @644 mp_neck
seg_out=f"{CD}/sam3_ve_seg_644.mp_neck.onnx"
print("=== quantize SEG VE @644 mp_neck (neck excluded) ===", flush=True)
tq=time.time()
moq.quantize(onnx_path=seg_ve, calibration_data_reader=R(), quantize_mode="int8",
    calibration_method="max", calibration_eps=["cpu"], high_precision_dtype="fp32",
    disable_mha_qdq=True, mha_accumulation_dtype="fp32", nodes_to_exclude=["/neck/.*"],
    use_external_data_format=True, output_path=seg_out)
print(f"  seg mp_neck done {time.time()-tq:.0f}s QuantizeLinear={qcount(seg_out)} -> {seg_out}", flush=True)
print(f"=== BASE DONE {time.time()-t0:.0f}s ===", flush=True)
sys.stdout.flush(); os._exit(0)
