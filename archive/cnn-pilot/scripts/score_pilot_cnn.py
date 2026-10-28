import argparse,csv,json,pathlib
import numpy as np
import torch
import train_cnn_evaluator as evaluator

if __name__=='__main__':
 p=argparse.ArgumentParser(description='Score 512-base ACGT continuations using the frozen CNN ensemble. Higher synthetic score means easier to classify as generated.')
 p.add_argument('--input',type=pathlib.Path,required=True,help='JSONL records with text or generated field; prompts must already be excluded')
 p.add_argument('--output',type=pathlib.Path,required=True)
 p.add_argument('--device',default='cuda')
 p.add_argument('--checkpoints',type=pathlib.Path,nargs='+',default=[evaluator.OUT/f'main-seed{s}.pt' for s in (6204,6205,6206)])
 args=p.parse_args();torch.set_num_threads(4);rows=list(map(json.loads,args.input.open()))
 tensors=torch.stack([evaluator.onehot(r.get('text',r.get('generated'))) for r in rows]).to(args.device);logits=[]
 for checkpoint in args.checkpoints:
  model=evaluator.upstream.DilatedResNet1D().to(args.device).eval();saved=torch.load(checkpoint,map_location=args.device,weights_only=False);model.load_state_dict(saved['model'],strict=True)
  scores=[]
  with torch.inference_mode():
   for batch in tensors.split(32):scores.extend(((model(batch)+model(batch.flip((1,2))))/2).cpu().tolist())
  logits.append(scores)
 mean=np.mean(logits,axis=0);assert np.isfinite(mean).all();values=1/(1+np.exp(-np.clip(mean,-40,40)))
 args.output.parent.mkdir(exist_ok=True,parents=True)
 with args.output.open('w') as f:
  w=csv.DictWriter(f,fieldnames=['id','synthetic_score','natural_score']);w.writeheader()
  for i,(r,value) in enumerate(zip(rows,values)):w.writerow({'id':r.get('id',i),'synthetic_score':float(value),'natural_score':float(1-value)})
 print('SCORED',len(rows))
