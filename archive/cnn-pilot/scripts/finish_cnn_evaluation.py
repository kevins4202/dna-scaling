import hashlib,json,pathlib
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'data/cnn-pilot';OUT=ROOT/'reports/cnn-pilot'
summary=json.loads((OUT/'summary.json').read_text());predictions=json.loads((OUT/'test-predictions.json').read_text());ids=sorted({r['id'] for r in predictions});rng=np.random.default_rng(6207);diff=[]
for _ in range(1000):
 sampled=rng.choice(ids,len(ids),replace=True);scores=[]
 for size in ('5m','10m'):
  rows=[r for i in sampled for r in predictions if r['id']==i and r['source'] in ('natural',size)]
  scores.append(roc_auc_score([r['label'] for r in rows],[r['logit'] for r in rows]))
 diff.append(scores[0]-scores[1])
summary['paired_detection_auc_difference_5m_minus_10m']={'estimate':summary['models']['5m']['ensemble_test_auc']-summary['models']['10m']['ensemble_test_auc'],'bootstrap_ci95':np.quantile(diff,[.025,.975]).tolist()}
summary['interpretation']='Controls passed, but both generator-specific confidence intervals include chance. This small pilot does not establish reliable detection or a difference between generator sizes. Increase independent source groups before using CNN scores as the main research conclusion or reward.'
prompts=list(map(json.loads,(DATA/'prompts.jsonl').read_text().splitlines()));groups={r['id']:r for r in prompts};assert len(groups)==100
fingerprints={}
for size in ('5m','10m'):
 path=DATA/f'generated-{size}.jsonl';generated=list(map(json.loads,path.read_text().splitlines()));assert len(generated)==100 and len({r['id'] for r in generated})==100
 for r in generated:
  assert r['group_id']==groups[r['id']]['group_id'] and r['fold']==groups[r['id']]['fold'] and len(r['generated'])==512 and set(r['generated'])<=set('ACGT')
 fingerprints[size]={'samples_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'unique_generated_sequences':len({r['generated'] for r in generated})}
summary['sample_audit']=fingerprints;summary['prompts_sha256']=hashlib.sha256((DATA/'prompts.jsonl').read_bytes()).hexdigest();summary['cnn_source_sha256']=hashlib.sha256((ROOT/'vendor/synthetic_genomes_benchmark/scripts/classifier/deep_detector_dilated_resnet1d.py').read_bytes()).hexdigest()
summary['checkpoint_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('main-seed*.pt')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['paired_detection_auc_difference_5m_minus_10m'],indent=2))
