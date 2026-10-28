import collections,csv,hashlib,importlib.util,json,math,pathlib,random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'data/cnn-pilot';OUT=ROOT/'reports/cnn-pilot';OUT.mkdir(exist_ok=True,parents=True)
SOURCE=ROOT/'vendor/synthetic_genomes_benchmark/scripts/classifier/deep_detector_dilated_resnet1d.py'
spec=importlib.util.spec_from_file_location('published_detector',SOURCE);upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
ENC=np.full(256,-1,dtype=np.int64)
for i,c in enumerate(b'ACGT'):ENC[c]=i

def onehot(text):
 a=ENC[np.frombuffer(text.encode(),dtype=np.uint8)];assert a.min()>=0 and len(a)==512
 return torch.nn.functional.one_hot(torch.tensor(a),4).float().T

def dataset(prompts,generated,kind):
 rows=[]
 for r in prompts:
  common={'id':r['id'],'group_id':r['group_id'],'fold':r['fold'],'category':r['category']}
  if kind=='main':
   rows.append({**common,'text':r['natural'],'label':0,'source':'natural'})
   for size in ('5m','10m'):rows.append({**common,'text':generated[size][r['id']]['generated'],'label':1,'source':size})
  elif kind=='natural_null':
   a,b=r['natural'],r['null_natural']
   if random.Random('natural-null/'+str(r['id'])).random()<.5:a,b=b,a
   rows.extend([{**common,'text':a,'label':0,'source':'natural_a'},{**common,'text':b,'label':1,'source':'natural_b'}])
  else:
   shuffled=list(r['natural']);random.Random('positive-control/'+str(r['id'])).shuffle(shuffled)
   rows.extend([{**common,'text':r['natural'],'label':0,'source':'natural'},{**common,'text':''.join(shuffled),'label':1,'source':'shuffled'}])
 return rows

def predict(model,x):
 model.eval()
 with torch.inference_mode():
  return torch.cat([((model(batch)+model(batch.flip((1,2))))/2).cpu() for batch in x.split(32)]).numpy()

def fit(rows,seed,name):
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 model=upstream.DilatedResNet1D().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 groups={fold:[r for r in rows if r['fold']==fold] for fold in ('train','calibration','test')}
 tensors={fold:torch.stack([onehot(r['text']) for r in records]).cuda() for fold,records in groups.items()}
 labels={fold:np.array([r['label'] for r in records]) for fold,records in groups.items()}
 weight=(labels['train']==0).sum()/max(1,(labels['train']==1).sum());criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device='cuda'))
 y=torch.tensor(labels['train'],device='cuda',dtype=torch.float32);best=-1;stale=0;history=[]
 checkpoint=OUT/f'{name}-seed{seed}.pt'
 for epoch in range(1,61):
  model.train();order=torch.randperm(len(y),device='cuda');loss_sum=0
  for indices in order.split(32):
   x=tensors['train'][indices].clone();mask=torch.rand(len(indices),device='cuda')<.5;x[mask]=x[mask].flip((1,2))
   optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y[indices]);assert torch.isfinite(loss);loss.backward();optimizer.step();loss_sum+=loss.item()*len(indices)
  logits=predict(model,tensors['calibration']);auc=float(roc_auc_score(labels['calibration'],logits));history.append({'epoch':epoch,'training_loss':loss_sum/len(y),'calibration_auc':auc})
  if auc>best+1e-5:
   best=auc;stale=0;torch.save({'model':model.state_dict(),'epoch':epoch,'calibration_auc':auc,'architecture':'published DilatedResNet1D defaults','seed':seed},checkpoint)
  else:stale+=1
  if stale>=10:break
 saved=torch.load(checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(saved['model'],strict=True)
 logits={f:predict(model,x) for f,x in tensors.items()}
 thresholds=np.unique(np.r_[0.0,logits['calibration']]);threshold=max(thresholds,key=lambda t:balanced_accuracy_score(labels['calibration'],logits['calibration']>=t))
 test=logits['test'];result={'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':saved['epoch'],'calibration_auc':best,'test_auc':float(roc_auc_score(labels['test'],test)),'test_balanced_accuracy':float(balanced_accuracy_score(labels['test'],test>=threshold)),'threshold_logit':float(threshold),'test_predictions':[{k:r[k] for k in ('id','group_id','category','source','label')}|{'logit':float(score),'synthetic_score':float(1/(1+np.exp(-np.clip(score,-40,40))))} for r,score in zip(groups['test'],test)],'history':history}
 if name=='main':
  result['models']={}
  for size in ('5m','10m'):
   chosen=[i for i,r in enumerate(groups['test']) if r['source'] in ('natural',size)]
   result['models'][size]={'auc':float(roc_auc_score(labels['test'][chosen],test[chosen])),'mean_synthetic_score':float(np.mean([result['test_predictions'][i]['synthetic_score'] for i in chosen if groups['test'][i]['source']==size]))}
 (OUT/f'{name}-seed{seed}.json').write_text(json.dumps(result,indent=2)+'\n');print('CNN_TRAINED',name,seed,'test_auc',result['test_auc'],flush=True)
 return result

def bootstrap(predictions,size=None):
 ids=sorted({r['id'] for r in predictions});rng=np.random.default_rng(6203);aucs=[]
 for _ in range(1000):
  chosen=rng.choice(ids,len(ids),replace=True);rows=[r for i in chosen for r in predictions if r['id']==i and (size is None or r['source'] in ('natural',size))]
  aucs.append(roc_auc_score([r['label'] for r in rows],[r['logit'] for r in rows]))
 return np.quantile(aucs,[.025,.975]).tolist()

if __name__=='__main__':
 torch.set_num_threads(4);prompts=list(map(json.loads,(DATA/'prompts.jsonl').open()));generated={size:{r['id']:r for r in map(json.loads,(DATA/f'generated-{size}.jsonl').open())} for size in ('5m','10m')}
 for size in generated:assert len(generated[size])==100 and all(len(r['generated'])==512 for r in generated[size].values())
 results=[fit(dataset(prompts,generated,'main'),seed,'main') for seed in (6204,6205,6206)]
 null=fit(dataset(prompts,generated,'natural_null'),6204,'natural_null');positive=fit(dataset(prompts,generated,'shuffled_control'),6204,'shuffled_control')
 combined=[]
 for i in range(len(results[0]['test_predictions'])):
  row=dict(results[0]['test_predictions'][i]);row['logit']=float(np.mean([r['test_predictions'][i]['logit'] for r in results]));row['synthetic_score']=float(1/(1+np.exp(-np.clip(row['logit'],-40,40))));combined.append(row)
 null_ci=bootstrap(null['test_predictions']);summary={'status':'complete','pilot':True,'window_length':512,'training_pairs':60,'calibration_pairs':20,'test_pairs':20,'cnn_parameters':results[0]['parameters'],'architecture':'Published DilatedResNet1D, base channels 64, stem kernel 15, residual kernel 7, dilations [1,2,4,8], global average pooling and binary head','source_commit':'2e2ab3808f4f55db1fbcf22df39610a17f4da1dc','label_definition':'1=generated, 0=natural; higher AUROC means easier detection, not better generation','models':{},'natural_null':{'test_auc':null['test_auc'],'bootstrap_ci95':null_ci,'chance_within_ci':bool(null_ci[0]<=.5<=null_ci[1])},'shuffled_positive_control':{'test_auc':positive['test_auc']},'seeds':[r['seed'] for r in results],'limitations':['Only 20 final source groups; confidence intervals are wide.','Source-row separation is not species/phylogenetic separation.','512-base continuations validate the evaluator but do not yet test long-distance deterioration.','Generation uses identical ACGT-constrained sampling at temperature 1; this choice defines the evaluated generator.','CNN scores measure distinguishability, not biological function.']}
 for size in ('5m','10m'):
  rows=[r for r in combined if r['source'] in ('natural',size)];summary['models'][size]={'ensemble_test_auc':float(roc_auc_score([r['label'] for r in rows],[r['logit'] for r in rows])),'bootstrap_ci95':bootstrap(combined,size),'mean_synthetic_score':float(np.mean([r['synthetic_score'] for r in rows if r['source']==size])),'individual_seed_aucs':[r['models'][size]['auc'] for r in results]}
 summary['validation_passed']=bool(summary['natural_null']['chance_within_ci'] and positive['test_auc']>=.75)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'test-predictions.json').write_text(json.dumps(combined,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(6,4));names=['5M','10M'];means=[summary['models'][s]['ensemble_test_auc'] for s in ('5m','10m')];cis=np.array([summary['models'][s]['bootstrap_ci95'] for s in ('5m','10m')]);ax.bar(names,means,color=['#31688e','#35b779']);ax.errorbar(names,means,yerr=np.array([np.array(means)-cis[:,0],cis[:,1]-np.array(means)]),fmt='none',color='black',capsize=5);ax.axhline(.5,ls='--',color='gray',label='Chance');ax.set_ylim(0,1.04);ax.set_ylabel('CNN detection AUROC (higher = easier to detect)');ax.set_title('512-base generation pilot; 20 held-out source groups');ax.legend();fig.tight_layout();fig.savefig(OUT/'cnn-detection.png',dpi=180);plt.close(fig)
 print(json.dumps(summary,indent=2))
