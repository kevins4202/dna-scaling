"""Generate matched, resumable continuations through the pinned Savanna launcher."""
import hashlib
import json
import os
import pathlib
import time

import torch
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.training import get_model

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/os.environ.get('DISTANCE_DATA_DIR','data/distance-2048-v1')
if OUT.resolve() not in {(ROOT/'data/distance-2048-v1').resolve(),(ROOT/'data/distance-2048-expanded-v1').resolve()}:
    raise ValueError('Unknown registered evaluation directory')
EXPANDED=OUT.name=='distance-2048-expanded-v1'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def set_length(model, config, length):
    config.seq_length=length
    for module in model.modules():
        if hasattr(module,'seq_length'): module.seq_length=length

def forward(model,tokens):
    positions=torch.arange(tokens.shape[1],device='cuda')[None,:].expand(tokens.shape[0],-1)
    output=model((tokens,None,positions))
    return output if isinstance(output,torch.Tensor) else output[0]

def completed_rows(path, prompts, size):
    if not path.exists(): return {}
    raw=path.read_bytes(); lines=raw.splitlines(keepends=True); valid=[]; result={}
    for i,line in enumerate(lines):
        try: row=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith(b'\n'): raise
            break
        identity=row['id']
        if identity in result or identity not in prompts: raise ValueError('Invalid/duplicate generation id')
        if row['model']!=size or row['group_id']!=prompts[identity]['group_id'] or row['seed']!=prompts[identity]['seed']: raise ValueError('Generation provenance mismatch')
        if len(row['generated'])!=2048 or not set(row['generated'])<=set('ACGT'): raise ValueError('Invalid generated DNA')
        result[identity]=row; valid.append(line if line.endswith(b'\n') else line+b'\n')
    repaired=b''.join(valid)
    if repaired!=raw:
        temp=path.with_suffix('.repair');temp.write_bytes(repaired);temp.replace(path)
    return result

def main():
    before=time.time(); manifest=json.loads((OUT/'manifest.json').read_text())
    for name,expected in manifest['inputs_sha256'].items():
        if digest(ROOT/name)!=expected: raise ValueError('Registered input changed: '+name)
    c=GlobalConfig.consume_global_config();c.configure_distributed_args();c.build_tokenizer();initialize_megatron(global_config=c)
    size=pathlib.Path(c.save).name
    if size not in ('5m','10m','25m','50m','100m') or os.environ.get('CUDA_VISIBLE_DEVICES')!='7': raise ValueError('Only authorized models on physical GPU 7')
    if EXPANDED:
        from distance.expanded_data import verify as verify_expanded
        verify_expanded()
    if not EXPANDED and size=='25m':
        from distance.register_25m import verify
        verify()
    if not EXPANDED and size in ('50m','100m'):
        from distance.family import verify as verify_family
        verify_family(size)
    model=get_model(global_config=c,use_cache=False)
    checkpoint=pathlib.Path(c.save)/'global_step3051/mp_rank_00_model_states.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['iteration']!=3051: raise ValueError('Wrong checkpoint iteration')
    model.load_state_dict(state['module'],strict=True);del state
    parameter_count=sum(p.numel() for p in model.parameters())
    if parameter_count!={'5m':5413780,'10m':10331804,'25m':24886444,'50m':51927104,'100m':97352024}[size]: raise ValueError('Parameter count mismatch')
    model=model.cuda().to(torch.bfloat16).eval()
    for module in model.modules():
        if type(module).__name__=='RowParallelLinear':
            module.register_forward_pre_hook(lambda m,args:(args[0].contiguous(),)+args[1:])
    import savanna.mpu.layers as layers
    layers.print_rank_0=lambda *a,**kw:None
    rows=[json.loads(line) for line in (OUT/'prompts.jsonl').read_text().splitlines()]
    prompts={r['id']:r for r in rows};path=OUT/f'generated-{size}.jsonl'
    completed=completed_rows(path,prompts,size);allowed=torch.tensor(list(b'ACGT'),device='cuda')
    checks=[];batch_size=int(os.environ.get('DISTANCE_BATCH_SIZE','20'))
    with torch.inference_mode():
        for length in (256,768,1280,1792,2303):
            probe=torch.tensor([[ord(x) for x in (r['prefix']+r['natural'])[:length]] for r in rows[:2]],device='cuda')
            set_length(model,c,4096)
            reference=forward(model,torch.nn.functional.pad(probe,(0,4096-length),value=0))[:,length-1].float().clone()
            set_length(model,c,length);dynamic=forward(model,probe)[:,-1].float()
            delta=(reference-dynamic).abs().max().item()
            if not torch.isfinite(dynamic).all() or delta>=.15: raise ValueError(('Dynamic inference validation failed',length,delta))
            checks.append({'length':length,'max_absolute_logit_difference':delta,'tolerance':.15})
        for start in range(0,len(rows),batch_size):
            batch=[r for r in rows[start:start+batch_size] if r['id'] not in completed]
            if not batch: continue
            tokens=torch.tensor([list(r['prefix'].encode()) for r in batch],device='cuda');new=[];tick=time.time()
            generators=[torch.Generator(device='cuda').manual_seed(r['seed']) for r in batch]
            mass=0.0
            for step in range(2048):
                set_length(model,c,tokens.shape[1]);logits=forward(model,tokens)[:,-1].float()
                if not torch.isfinite(logits).all(): raise ValueError('Nonfinite generation logits')
                mass+=torch.softmax(logits,-1)[:,allowed].sum().item()/len(batch)
                probabilities=torch.softmax(logits[:,allowed],-1)
                indices=torch.stack([torch.multinomial(probabilities[i],1,generator=generators[i]) for i in range(len(batch))]).reshape(-1)
                nxt=allowed[indices];new.append(nxt);tokens=torch.cat((tokens,nxt[:,None]),dim=1)
                if (step+1)%256==0: print('PROGRESS',size,'batch',start//batch_size,'bases',step+1,'seconds',round(time.time()-tick,1),flush=True)
            sequences=torch.stack(new,dim=1).cpu().tolist()
            with path.open('a') as output:
                for row,sequence in zip(batch,sequences):
                    result={'id':row['id'],'group_id':row['group_id'],'model':size,'seed':row['seed'],
                            'generated':''.join(map(chr,sequence)),'mean_acgt_probability_mass':mass/2048}
                    output.write(json.dumps(result)+'\n')
                output.flush();os.fsync(output.fileno())
            print('SAVED',size,start+len(batch),'of',len(rows),flush=True)
    final=completed_rows(path,prompts,size)
    if len(final)!=manifest['pairs']: raise ValueError('Incomplete generation')
    metadata={'status':'complete','model':size,'parameters':parameter_count,'count':len(final),
        'continuation_bases':2048,'batch_size':batch_size,'checkpoint_sha256':digest(checkpoint),
        'manifest_sha256':digest(OUT/'manifest.json'),'output_sha256':digest(path),
        'dynamic_inference_checks':checks,'seconds_this_invocation':time.time()-before,
        'sampling':manifest['sampling'],'device':torch.cuda.get_device_name(0),'torch':torch.__version__}
    metadata['evaluation_version']=manifest['version']
    if not EXPANDED and size in ('25m','50m','100m'): metadata['extension_sha256']=digest(OUT/f'extension-{size}.json')
    (OUT/f'generation-{size}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
