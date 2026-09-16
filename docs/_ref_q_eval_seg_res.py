"""Box & mask accuracy of a merged seg engine (images+text -> masks+boxes) on tomato (real polygon GT).
Reports box mAP + mask mAP vs GT. Run with ENG=<engine> and NAME=<label>. os._exit to dodge pycuda teardown."""
import os, sys, json, glob, numpy as np, cv2
os.environ.setdefault("TRANSFORMERS_VERBOSITY","error"); os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
os.environ.setdefault("HF_HOME","/root/willy/hf_cache") if os.path.exists("/root/willy/hf_cache") else None
import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit
from transformers import Sam3Processor
from PIL import Image
D="/root/willy/models/pretrained_weights/sam3_huggingface"; DS="/root/willy/datasets/tomato_test"
SCORE_MIN=0.05; MAXD=50
L=trt.Logger(trt.Logger.ERROR)
ENG=os.environ["ENG"]; NAME=os.environ.get("NAME","eng")
class E:
    """Persistent buffers (static batch=1): allocate once, reuse -> stable with Myelin seg engine."""
    def __init__(s,p):
        with open(p,"rb") as f: s.e=trt.Runtime(L).deserialize_cuda_engine(f.read())
        s.c=s.e.create_execution_context(); s.n=[s.e.get_tensor_name(i) for i in range(s.e.num_io_tensors)]
        s.isin={n:(s.e.get_tensor_mode(n)==trt.TensorIOMode.INPUT) for n in s.n}
        s.dev={}; s.host={}; s.stream=cuda.Stream(); s.ready=False
    def _setup(s,feeds):
        for n in s.n:
            if s.isin[n]: s.c.set_input_shape(n,feeds[n if n in feeds else n.split('/')[-1]].shape)
        for n in s.n:
            dt=trt.nptype(s.e.get_tensor_dtype(n)); shp=tuple(s.c.get_tensor_shape(n))
            s.host[n]=np.empty(shp,dtype=dt); s.dev[n]=cuda.mem_alloc(s.host[n].nbytes)
            s.c.set_tensor_address(n,int(s.dev[n]))
        s.ready=True
    def __call__(s,feeds):
        if not s.ready: s._setup(feeds)
        for n in s.n:
            if s.isin[n]:
                np.copyto(s.host[n], np.ascontiguousarray(feeds[n if n in feeds else n.split('/')[-1]].astype(s.host[n].dtype)))
                cuda.memcpy_htod_async(s.dev[n], s.host[n], s.stream)
        s.c.execute_async_v3(s.stream.handle)
        for n in s.n:
            if not s.isin[n]: cuda.memcpy_dtoh_async(s.host[n], s.dev[n], s.stream)
        s.stream.synchronize()
        return {n.split("/")[-1]:s.host[n].copy() for n in s.n if not s.isin[n]}
def sig(x): return 1/(1+np.exp(-x.astype(np.float32)))
def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    iw=max(0.,x2-x1);ih=max(0.,y2-y1);inter=iw*ih
    ua=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
    return inter/(ua+1e-9)
def mask_iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return inter/(union+1e-9)
proc=Sam3Processor.from_pretrained("facebook/sam3")
eng=E(ENG)
S=[tuple(eng.e.get_tensor_shape(n)) for n in eng.n if "images" in n][0][-1]
MR=[tuple(eng.e.get_tensor_shape(n)) for n in eng.n if "pred_masks" in n][0][-1]
print(f"res S={S} mask MR={MR}",flush=True)
tf=np.load(f"{D}/ckpts/text/tomato/text_features.npy").astype(np.float32); tm=np.load(f"{D}/ckpts/text/tomato/attention_mask.npy").astype(np.float32)
coco=json.load(open(f"{DS}/_annotations.coco.json"))
gt_boxes={}; gt_masks={}
for a in coco["annotations"]:
    im=next(i for i in coco["images"] if i["id"]==a["image_id"]); W,H=im["width"],im["height"]
    x,y,w,h=a["bbox"]; gt_boxes.setdefault(a["image_id"],[]).append([x,y,x+w,y+h])
    m=np.zeros((MR,MR),np.uint8)
    for poly in a["segmentation"]:
        pts=np.array(poly,np.float64).reshape(-1,2)
        pts[:,0]*=MR/W; pts[:,1]*=MR/H
        cv2.fillPoly(m,[pts.round().astype(np.int32)],1)
    gt_masks.setdefault(a["image_id"],[]).append(m.astype(bool))
import os as _os
NLIM=int(_os.environ.get("NLIM","0"))
det_box=[]; det_mask=[]   # (img_id, score, box) and (img_id, score, mask_bool)
for _k,im in enumerate(coco["images"]):
    if NLIM and _k>=NLIM: break
    if _k%40==0: print("  img",_k,flush=True)
    p=f"{DS}/{im['file_name']}"
    if not os.path.exists(p): continue
    W,H=im["width"],im["height"]
    pv=proc(images=Image.open(p).convert("RGB"),text="tomato",return_tensors="pt")["pixel_values"].numpy()[0]  # [3,1008,1008]
    pv=cv2.resize(np.transpose(pv,(1,2,0)),(S,S),interpolation=cv2.INTER_LINEAR)  # [644,644,3]
    px=np.ascontiguousarray(np.transpose(pv,(2,0,1))[None].astype(np.float32))
    r=eng({"images":px,"text_features":tf,"text_mask":tm})
    _ncnt=_ncnt+1 if "_ncnt" in dir() else 1
    pb=r["pred_boxes"][0]; pl=r["pred_logits"].reshape(-1); pr=r["presence_logits"].reshape(-1)
    pm=r["pred_masks"][0]   # [200,184,184] logits
    score=sig(pl)*float(sig(pr[0]))
    keep=np.where(score>=SCORE_MIN)[0]; order=keep[np.argsort(-score[keep])][:MAXD]
    for q in order:
        b=[pb[q,0]*W,pb[q,1]*H,pb[q,2]*W,pb[q,3]*H]
        det_box.append((im["id"],float(score[q]),b))
        det_mask.append((im["id"],float(score[q]),pm[q]>0.0))
print(f"collected: box_dets={len(det_box)} mask_dets={len(det_mask)}",flush=True)
def ap(dets,gts,iou_fn,thr):
    npos=sum(len(v) for v in gts.values())
    if npos==0: return float('nan')
    dets=sorted(dets,key=lambda x:-x[1]); matched={k:[False]*len(v) for k,v in gts.items()}
    tp=np.zeros(len(dets));fp=np.zeros(len(dets))
    for i,(img,sc,d) in enumerate(dets):
        gg=gts.get(img,[]);best=0.;bi=-1
        for j,g in enumerate(gg):
            v=iou_fn(d,g)
            if v>best:best=v;bi=j
        if best>=thr and not matched[img][bi]: tp[i]=1;matched[img][bi]=True
        else: fp[i]=1
    tp=np.cumsum(tp);fp=np.cumsum(fp);rec=tp/npos;prec=tp/np.maximum(tp+fp,1e-9)
    mrec=np.concatenate(([0],rec,[1]));mpre=np.concatenate(([0],prec,[0]))
    for i in range(len(mpre)-1,0,-1): mpre[i-1]=max(mpre[i-1],mpre[i])
    idx=np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))
thrs=[round(0.5+0.05*i,2) for i in range(10)]
print("computing box AP...",flush=True)
bmap=float(np.mean([ap(det_box,gt_boxes,box_iou,t) for t in thrs])); b50=ap(det_box,gt_boxes,box_iou,0.5)
print("computing mask AP...",flush=True)
mmap=float(np.mean([ap(det_mask,gt_masks,mask_iou,t) for t in thrs])); m50=ap(det_mask,gt_masks,mask_iou,0.5)
print(f"[{NAME}] BOX  AP@.5={b50:.4f} mAP@[.5:.95]={bmap:.4f}")
print(f"[{NAME}] MASK AP@.5={m50:.4f} mAP@[.5:.95]={mmap:.4f}")
sys.stdout.flush(); os._exit(0)
