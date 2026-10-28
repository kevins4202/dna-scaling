"""Diagnose recurrent versus full-prefix numerical error without changing probe limits."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate as g
import torch
p=g.rows(g.ORIGINAL/'prompts.jsonl')[20]
checkpoint=g.read(g.OUT/'download-manifest.json')['checkpoint']
m=g.Evo2('evo2_7b_base',local_path=str(g.ROOT/checkpoint),use_kernels=False).model.eval().requires_grad_(False)
allowed=torch.tensor([ord(c) for c in 'ACGT'],device='cuda')
prefix=torch.tensor([[ord(c) for c in p['prefix']]],device='cuda')
teacher=torch.tensor([[ord(c) for c in p['natural'][:31]]],device='cuda')
cache=m.initialize_inference_params(max_seqlen=2304)
results=[]
with torch.inference_mode():
    chunk,_=m(torch.cat((prefix,teacher),1),inference_params_dict=None)
    for i in range(32):
        if i:g.advance(cache,256)
        cached,cache=g.logits_last(m,prefix if i==0 else teacher[:,i-1:i],cache)
        exact,_=m(torch.cat((prefix,teacher[:,:i]),1),inference_params_dict=None)
        exact=exact[:,-1].float();long=chunk[:,255+i].float()
        results.append({'i':i,'cached_exact_logit':(cached[:,allowed]-exact[:,allowed]).abs().max().item(),'cached_exact_l1':(cached[:,allowed].softmax(-1)-exact[:,allowed].softmax(-1)).abs().sum().item(),'full_shapes_l1':(long[:,allowed].softmax(-1)-exact[:,allowed].softmax(-1)).abs().sum().item()})
        print(results[-1],flush=True)
g.write(g.OUT/'cache-diagnosis.json',{'id':20,'positions':results})
