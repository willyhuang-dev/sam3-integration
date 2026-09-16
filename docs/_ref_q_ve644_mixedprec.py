"""Mixed-precision INT8: keep the most quantization-sensitive layers in fp16 (nodes_to_exclude),
rest INT8. Stays mostly INT8 for speed. Test 3 hypotheses (res 644, modelopt.onnx, deployable):
  io          : patch_embed conv (raw-pixel input) + neck convs (produce fpn_feat_2) fp16
  io_fc2      : + every block's mlp/fc2 (post-GELU down-proj = classic sensitive ViT layer)
  io_fc2_ends : + first/last 2 blocks
calibration_method=max (entropy==max per §18). N=8 real imgs. disable_mha_qdq (attention already fp16)."""
import os, sys, glob, time
import numpy as np, torch, torch.nn.functional as Fn
os.environ.setdefault("TRANSFORMERS_VERBOSITY","error"); os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
HERE="/root/willy/models/pretrained_weights/sam3_huggingface"; CD=f"{HERE}/exp/quantization/int8fp8"
sys.path.insert(0, HERE)
from transformers import Sam3Processor
from PIL import Image
N=8
proc=Sam3Processor.from_pretrained("facebook/sam3")
files=sorted(glob.glob(f"{HERE}/images/**/*.jpg", recursive=True))[:N]
imgs=[Fn.interpolate(torch.from_numpy(proc(images=Image.open(f).convert("RGB"), text="person", return_tensors="pt")["pixel_values"].numpy().astype(np.float32)), size=(644,644), mode="bilinear", align_corners=False).numpy() for f in files]
print(f"calib {len(imgs)} x {imgs[0].shape}", flush=True)
from onnxruntime.quantization import CalibrationDataReader
def mkr():
    class R(CalibrationDataReader):
        def __init__(s): s.i=0
        def get_first(s): return {"images": imgs[0]}
        def get_next(s):
            if s.i>=len(imgs): return None
            d={"images": imgs[s.i]}; s.i+=1; return d
        def rewind(s): s.i=0
    return R()
import modelopt.onnx.quantization as moq
IN=f"{HERE}/ckpts/onnx_opset=16/res=644/sam3_ve.onnx"
configs={
 "io":          ["patch_embeddings", "/neck/"],
 "io_fc2":      ["patch_embeddings", "/neck/", "/mlp/fc2/"],
 "io_fc2_ends": ["patch_embeddings", "/neck/", "/mlp/fc2/", r"/layers\.(0|1|30|31)/"],
}
for name, excl in configs.items():
    out=f"{CD}/sam3_ve_644.mp_{name}.onnx"
    print(f"=== quantize mp_{name} exclude={excl} {time.strftime('%T')} ===", flush=True)
    t0=time.time()
    try:
        moq.quantize(onnx_path=IN, calibration_data_reader=mkr(), quantize_mode="int8",
            calibration_method="max", calibration_eps=["cpu"], high_precision_dtype="fp32",
            disable_mha_qdq=True, mha_accumulation_dtype="fp32", nodes_to_exclude=excl,
            use_external_data_format=True, output_path=out)
        print(f"=== mp_{name} DONE {time.time()-t0:.0f}s -> {out} ===", flush=True)
    except Exception as e:
        print(f"mp_{name} FAILED: {type(e).__name__}: {e}", flush=True)
print("ALL MP QUANT DONE", flush=True)
sys.stdout.flush(); os._exit(0)
