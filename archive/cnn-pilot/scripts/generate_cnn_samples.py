import contextlib,hashlib,json,os,pathlib,time
import torch
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.training import get_model
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'data/cnn-pilot'

def set_length(model,config,length):
 config.seq_length=length
 for module in model.modules():
  if hasattr(module,'seq_length'):module.seq_length=length

def forward(model,tokens):
 positions=torch.arange(tokens.shape[1],device='cuda')[None,:].expand(tokens.shape[0],-1)
 output=model((tokens,None,positions))
 return output if isinstance(output,torch.Tensor) else output[0]

if __name__=='__main__':
 c=GlobalConfig.consume_global_config();c.configure_distributed_args();c.build_tokenizer();initialize_megatron(global_config=c)
 model=get_model(global_config=c,use_cache=False);checkpoint=pathlib.Path(c.save)/f'global_step{c.train_iters}/mp_rank_00_model_states.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert state['iteration']==3051;model.load_state_dict(state['module'],strict=True);model=model.cuda().to(torch.bfloat16).eval()
 for module in model.modules():
  if type(module).__name__ == 'RowParallelLinear':
   module.register_forward_pre_hook(lambda m,args:(args[0].contiguous(),)+args[1:])
 import savanna.mpu.layers as layers
 layers.print_rank_0=lambda *a,**kw:None
 rows=list(map(json.loads,(OUT/'prompts.jsonl').open()));size=pathlib.Path(c.save).name;path=OUT/f'generated-{size}.jsonl'
 completed={r['id'] for r in map(json.loads,path.open())} if path.exists() else set()
 allowed=torch.tensor([ord(x) for x in 'ACGT'],device='cuda');equality={};batch_size=10
 with torch.inference_mode():
  probe=torch.tensor([[ord(x) for x in row['prefix']] for row in rows[:2]],device='cuda')
  set_length(model,c,4096);padded=torch.nn.functional.pad(probe,(0,4096-probe.shape[1]),value=0)
  reference=forward(model,padded)[:,probe.shape[1]-1].float()
  set_length(model,c,probe.shape[1]);dynamic=forward(model,probe)[:,-1].float()
  delta=(reference-dynamic).abs().max().item();assert delta<.15,('Dynamic-length logits differ',delta)
  equality={'fixed_vs_dynamic_max_absolute_logit_difference':delta,'tolerance':.15,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
  pending=[r for r in rows if r['id'] not in completed]
  for start in range(0,len(pending),batch_size):
   batch=pending[start:start+batch_size];tokens=torch.tensor([[ord(x) for x in row['prefix']] for row in batch],device='cuda');new=[];mass=0.0
   generators=[torch.Generator(device='cuda').manual_seed(6202+row['id']) for row in batch]
   before=time.time()
   for step in range(512):
    set_length(model,c,tokens.shape[1]);logits=forward(model,tokens)[:,-1].float();assert torch.isfinite(logits).all()
    mass+=torch.softmax(logits,dim=-1)[:,allowed].sum().item()/len(batch)
    probabilities=torch.softmax(logits[:,allowed],dim=-1)
    indices=torch.stack([torch.multinomial(probabilities[i],1,generator=generators[i]) for i in range(len(batch))]).reshape(-1)
    nxt=allowed[indices];new.append(nxt);tokens=torch.cat([tokens,nxt[:,None]],dim=1)
   generated=torch.stack(new,dim=1).cpu().tolist()
   with path.open('a') as f:
    for row,sequence in zip(batch,generated):f.write(json.dumps({'id':row['id'],'group_id':row['group_id'],'fold':row['fold'],'model':size,'generated':''.join(map(chr,sequence))})+'\n')
   print('GENERATED',size,start+len(batch),'/',len(pending),'seconds',time.time()-before,flush=True)
   equality['mean_acgt_probability_mass_last_batch']=mass/512
 (OUT/f'generation-{size}.json').write_text(json.dumps({'status':'complete','model':size,'count':len(rows),'prefix_length':256,'new_bases':512,'sampling':'ACGT-constrained categorical sampling, temperature 1, no top-k/p or repetition penalty; same prompts and per-prompt seeds for both model sizes',**equality},indent=2)+'\n')
 torch.distributed.destroy_process_group()
