"""Official Evo 2 BF16 reload and full-prefix generation with registered ACGT sampling."""
import argparse,collections,os,sys,time,gc
sys.modules['transformer_engine']=None
from common import ROOT,OUT,DATA,ORIGINAL,guard,read,write,rows,sha
import torch
from evo2 import Evo2
from vortex.model.model import StripedHyena
from compat import install
ATTENTION_COMPAT=install()

def checked_load(self,state_dict,strict=True):
    expected=self.state_dict();missing=set(expected)-set(state_dict)
    extra=set(state_dict)-set(expected)
    assert not missing,missing
    tied_output_verified=False
    if 'unembed.weight' in extra:
        assert self.config.tie_embeddings,'Unexpected untied output embedding'
        assert torch.equal(state_dict['unembed.weight'],state_dict['embedding_layer.weight']),'Tied checkpoint embeddings differ'
        extra.remove('unembed.weight');tied_output_verified=True
    assert all(k.endswith('._extra_state') for k in extra),extra
    for k,v in expected.items():
        if torch.is_tensor(v):assert v.shape==state_dict[k].shape,k
    checked_load.evidence={'strict':True,'model_state_keys':len(expected),'ignored_fp8_metadata_keys':sorted(extra),'duplicate_tied_output_verified':tied_output_verified}
    return checked_load.original(self,state_dict,strict=True)
checked_load.original=StripedHyena.custom_load_state_dict
StripedHyena.custom_load_state_dict=checked_load

def advance(cache,prefix_length):
    offset=cache['mha'].seqlen_offset
    offset=prefix_length if offset==0 else offset+1
    for name in ('mha','hcl','hcm','hcs'):cache[name].seqlen_offset=offset

def logits_last(model,x,cache):
    logits,cache=model(x,inference_params_dict=cache)
    last=logits[:,-1].float()
    assert torch.isfinite(last).all(),'Nonfinite logits'
    return last,cache

@torch.inference_mode()
def probe(model,prompt,allowed):
    prefix=torch.tensor([[ord(c) for c in prompt['prefix']]],device='cuda')
    teacher=torch.tensor([[ord(c) for c in prompt['natural'][:31]]],device='cuda')
    maximum=0.0
    for i in range(32):
        x=torch.cat((prefix,teacher[:,:i]),dim=1)
        first,_=logits_last(model,x,None)
        replay,_=logits_last(model,x.clone(),None)
        maximum=max(maximum,(first[:,allowed]-replay[:,allowed]).abs().max().item())
    assert maximum==0.0,maximum
    return {'id':prompt['id'],'positions':32,'finite':True,'cached_used':False,'mode':'full_prefix_replay','max_replay_logit_difference':maximum}

@torch.inference_mode()
def generate(model,prompt,allowed,n):
    x=torch.tensor([[ord(c) for c in prompt['prefix']]],device='cuda')
    rng=torch.Generator(device='cuda').manual_seed(prompt['seed']);output=[]
    for i in range(n):
        last,_=logits_last(model,x,None)
        probabilities=last[:,allowed].softmax(-1)
        new=allowed[torch.multinomial(probabilities,1,generator=rng)]
        output.append(chr(new.item()));x=torch.cat((x,new),dim=1)
    return ''.join(output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    assert torch.cuda.device_count()==1
    protocol=guard();release=protocol['checkpoint']
    assert sha(ROOT/release['checkpoint'])==release['checkpoint_sha256']
    prompts=rows(ORIGINAL/'prompts.jsonl');assert len(prompts)==100
    torch.manual_seed(6200);torch.cuda.manual_seed_all(6200)
    model_wrapper=Evo2('evo2_7b_base',local_path=str(ROOT/release['checkpoint']),use_kernels=False)
    model=model_wrapper.model.eval().requires_grad_(False)
    assert not model.config.use_fp8_input_projections
    assert all(p.device.type=='cuda' and p.device.index==0 for p in model.parameters())
    assert model_wrapper.tokenizer.tokenize(prompts[0]['prefix'])==[ord(c) for c in prompts[0]['prefix']]
    allowed=torch.tensor([ord(c) for c in 'ACGT'],device='cuda')
    metadata={'attention_compatibility':ATTENTION_COMPAT,'model':'evo2_7b_base','parameters':sum(p.numel() for p in model.parameters()),'architecture':dict(model.config),'checkpoint_sha256':release['checkpoint_sha256'],'protocol_sha256':sha(OUT/'protocol.json'),'strict_reload':checked_load.evidence,'dtype_counts':dict(collections.Counter(str(p.dtype) for p in model.parameters())),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda_visible_devices':'7'}
    metadata['architecture']={k:str(v) if not isinstance(v,(str,int,float,bool,list,type(None))) else v for k,v in metadata['architecture'].items()}
    if args.smoke:
        metadata['probes']=[probe(model,prompts[i],allowed) for i in (0,20,40,60,80)]
        a=generate(model,prompts[0],allowed,32);b=generate(model,prompts[0],allowed,32)
        assert a==b and len(a)==32 and set(a)<=set('ACGT')
        metadata.update(status='verified',generation_32_bases=a,deterministic_replay=True)
        write(OUT/'smoke-verification.json',metadata);print('SMOKE_VERIFIED',metadata['parameters'],flush=True);return
    smoke=read(OUT/'smoke-verification.json');assert smoke['status']=='verified' and smoke['protocol_sha256']==metadata['protocol_sha256']
    path=DATA/'generated-evo2_7b_base.jsonl'
    existing=rows(path) if path.exists() else []
    assert len({r['id'] for r in existing})==len(existing)
    by_id={p['id']:p for p in prompts};complete={}
    for r in existing:
        p=by_id[r['id']]
        assert r['seed']==p['seed'] and r['prefix']==p['prefix'] and r['category']==p['category'] and r['group_id']==p['group_id']
        assert len(r['generated'])==2048 and set(r['generated'])<=set('ACGT')
        assert r['checkpoint_sha256']==metadata['checkpoint_sha256'] and r['protocol_sha256']==metadata['protocol_sha256']
        complete[r['id']]=r
    start=time.time()
    for p in prompts:
        if p['id'] in complete:continue
        generated=generate(model,p,allowed,2048)
        record={k:p[k] for k in ('id','group_id','category','seed','prefix')};record.update(generated=generated,checkpoint_sha256=metadata['checkpoint_sha256'],protocol_sha256=metadata['protocol_sha256'])
        assert len(generated)==2048 and set(generated)<=set('ACGT')
        complete[p['id']]=record
        temporary=path.with_suffix('.tmp')
        with temporary.open('w') as f:
            import json
            for q in prompts:
                if q['id'] in complete:f.write(json.dumps(complete[q['id']])+'\n')
            f.flush();os.fsync(f.fileno())
        temporary.replace(path)
        print('COMPLETED',len(complete),'/ 100','elapsed_seconds',round(time.time()-start,1),flush=True)
        gc.collect()
    metadata.update(status='complete',count=100,generated_bases=2048,output_sha256=sha(path),smoke_sha256=sha(OUT/'smoke-verification.json'))
    write(DATA/'generation-evo2_7b_base.json',metadata);guard()
if __name__=='__main__':main()
