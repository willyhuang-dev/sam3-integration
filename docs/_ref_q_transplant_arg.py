import onnx, sys, time
from collections import defaultdict
D="/root/willy/models/pretrained_weights/sam3_huggingface"
Q_PATH=sys.argv[1]; OUT=sys.argv[2]
B_PATH=f"{D}/ckpts/onnx_opset=16/res=1008/sam3_ve.onnx"
t0=time.time()
q=onnx.load(Q_PATH, load_external_data=True); b=onnx.load(B_PATH, load_external_data=True)
qg,bg=q.graph,b.graph
q_init={i.name:i for i in qg.initializer}
q_cons=defaultdict(list)
for n in qg.node:
    for idx,inp in enumerate(n.input):
        if inp: q_cons[inp].append((n.name,idx))
qdq=[n for n in qg.node if n.op_type in ("QuantizeLinear","DequantizeLinear")]
b_init={i.name for i in bg.initializer}
need=set()
for n in qdq:
    for inp in n.input[1:]:
        if inp in q_init: need.add(inp)
for nm in need:
    if nm not in b_init: bg.initializer.append(q_init[nm])
for n in qdq: bg.node.append(n)
b_nodes={n.name:n for n in bg.node}
for n in qdq:
    for (cn,idx) in q_cons.get(n.output[0],[]):
        if cn in b_nodes and idx<len(b_nodes[cn].input): b_nodes[cn].input[idx]=n.output[0]
# topo sort
producers={}
for n in bg.node:
    for o in n.output: producers[o]=n
nodes=list(bg.node); adj=defaultdict(list); indeg={}
for n in nodes:
    d=0
    for inp in n.input:
        if inp and inp in producers and producers[inp] is not n: d+=1; adj[id(producers[inp])].append(n)
    indeg[id(n)]=d
from collections import deque
ready=deque(n for n in nodes if indeg[id(n)]==0); order=[]
while ready:
    n=ready.popleft(); order.append(n)
    for m in adj[id(n)]:
        indeg[id(m)]-=1
        if indeg[id(m)]==0: ready.append(m)
assert len(order)==len(nodes)
del bg.node[:]; bg.node.extend(order)
import os
loc=os.path.basename(OUT)+"_data"
onnx.save(b, OUT, save_as_external_data=True, all_tensors_to_one_file=True, location=loc, size_threshold=1024)
print(f"transplanted {len(qdq)} Q/DQ -> {OUT} in {time.time()-t0:.1f}s", flush=True)
