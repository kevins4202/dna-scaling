import collections
import json
import pathlib
import random
import shutil
import time

import numpy as np
import torch

import train_project_cnn as base
import evaluate_project_cnn as evaluation

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT/'data/cnn-final-mixed'
OUT = ROOT/'reports/cnn-final'
RELEASE = ROOT/'artifacts/cnn-v1'


def records(rows, natural, generated):
    return [{'id':r['pair_id'],'group_id':r['group_id'],'category':r['tag'],
             'natural_logit':float(a),'generated_logit':float(b),'generator':r['generator']}
            for r,a,b in zip(rows,natural,generated)]


def project_results(rows, logits, bootstrap=False):
    result = {}
    for size in ('5m','10m'):
        selected = [i for i,r in enumerate(rows) if r['domain']=='project' and r['generator']==size]
        subset = [rows[i] for i in selected]
        predictions = records(subset, logits[2*np.array(selected)], logits[2*np.array(selected)+1])
        result[size] = evaluation.metrics(predictions)
        if bootstrap:
            result[size]['source_group_bootstrap_ci95'] = evaluation.bootstrap(predictions)
        result[size]['pairs'] = len(predictions)
        result[size]['source_groups'] = len({r['group_id'] for r in predictions})
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if RELEASE.exists() or (OUT/'selected.pt').exists():
        raise RuntimeError('Final CNN exists; refusing an implicit retrain')
    if __import__('os').environ.get('CUDA_VISIBLE_DEVICES')!='7' or torch.cuda.device_count()!=1:
        raise RuntimeError('Use only physical GPU 7')
    torch.set_num_threads(4);torch.manual_seed(6211);np.random.seed(6211);random.seed(6211)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,sha in manifest['file_sha256'].items():assert base.digest(DATA/name)==sha
    folds={f:[json.loads(x) for x in (DATA/f'{f}.jsonl').read_text().splitlines()] for f in ('train','calibration','test')}
    groups={f:{r['group_id'] for r in rows if r['domain']=='project'} for f,rows in folds.items()}
    assert not (groups['train']&groups['test'] or groups['train']&groups['calibration'] or groups['test']&groups['calibration'])
    tensors={f:base.encode([r[k] for r in rows for k in ('natural','generated')],'cuda') for f,rows in folds.items() if f!='test'}
    counts=collections.Counter((r['domain'],r['tag']) for r in folds['train'])
    external_tags={r['tag'] for r in folds['train'] if r['domain']=='external'}
    own_tags={r['tag'] for r in folds['train'] if r['domain']=='project'}
    own_total=sum(evaluation.CATEGORY_WEIGHTS[k] for k in own_tags)
    weights=torch.tensor([(1/len(external_tags) if r['domain']=='external' else evaluation.CATEGORY_WEIGHTS[r['tag']]/own_total)/counts[(r['domain'],r['tag'])] for r in folds['train']],device='cuda')
    protocol={'version':'cnn-v1','seed':6211,'parameters':1227905,'training_domains':'50% external paired benchmark; 50% project 5M/10M outputs, equally sampled generators',
        'project_category_weights':evaluation.CATEGORY_WEIGHTS,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,
        'pairs_per_epoch':4096,'maximum_epochs':40,'patience':8,
        'checkpoint_selection':'.75 * mean 5M/10M calibration mixture-weighted category AUROC + .25 * external calibration macro AUROC',
        'final_test':'Fresh independent reference source groups; old pilot examples are training/development only.',
        'external_reference_policy':'External-only development model and its previously examined test are diagnostic, not evidence of transfer.',
        'adoption_checks':'Both project test AUROC lower confidence bounds >.5; adjacent-natural null interval includes .5; strict finite reload.',
        'no_test_tuning':'One fixed training configuration and initialization, no search or retraining after this test.',
        'source_sha256':base.digest(base.SOURCE),'data_manifest_sha256':base.digest(DATA/'manifest.json')}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    protocol['runtime']={'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'training_precision':'float32 with TF32 enabled',
        'released_inference_precision':'float32 with TF32 disabled'}
    protocol['generation']={size:json.loads((ROOT/f'data/cnn-project-final/generation-{size}.json').read_text()) for size in ('5m','10m')}
    protocol['generation_batch_size']=48
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=base.architecture().cuda()
    assert sum(p.numel() for p in model.parameters())==protocol['parameters']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];best=-1.;stale=0;start=time.time()
    for epoch in range(1,41):
        model.train();chosen=torch.multinomial(weights,4096,replacement=True);total=0.
        for pairs in chosen.split(32):
            indices=torch.stack((2*pairs,2*pairs+1),dim=1).reshape(-1)
            x=base.onehot(tensors['train'][indices]);rc=torch.rand(len(x),device='cuda')<.5;x[rc]=x[rc].flip((1,2))
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),(indices%2).float())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=loss.item()*len(x)
        logits=base.score(model,tensors['calibration']);own=project_results(folds['calibration'],logits)
        indices=[i for i,r in enumerate(folds['calibration']) if r['domain']=='external']
        external,_=base.paired_metrics([folds['calibration'][i] for i in indices],logits[2*np.array(indices)],logits[2*np.array(indices)+1])
        criterion=.75*np.mean([r['mixture_weighted_stratified_auc'] for r in own.values()])+.25*external['macro_auc']
        history.append({'epoch':epoch,'training_loss':total/8192,'selection_score':float(criterion),'project_calibration':own,'external_calibration':external,'elapsed_seconds':time.time()-start})
        (OUT/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('EPOCH',epoch,'loss',history[-1]['training_loss'],'calibration_selection',criterion,'seconds',time.time()-start,flush=True)
        if criterion>best+1e-5:
            best=float(criterion);stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'protocol':protocol,'calibration_selection':best},OUT/'selected.pt')
        else:stale+=1
        if stale>=8:break
    saved=torch.load(OUT/'selected.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model'],strict=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    test=folds['test'];logits=base.score(model,base.encode([r[k] for r in test for k in ('natural','generated')],'cuda'))
    results=project_results(test,logits,bootstrap=True)
    natural=base.score(model,base.encode([r['null_natural'] for r in test],'cuda'))
    unique=[i for i,r in enumerate(test) if r['generator']=='5m']
    null_records=records([test[i] for i in unique],logits[2*np.array(unique)],natural[unique])
    null=evaluation.metrics(null_records);null['source_group_bootstrap_ci95']=evaluation.bootstrap(null_records)
    accepted=all(r['source_group_bootstrap_ci95'][0]>.5 for r in results.values()) and null['source_group_bootstrap_ci95'][0]<=.5<=null['source_group_bootstrap_ci95'][1]
    predictions=records(test,logits[::2],logits[1::2]);(OUT/'test-predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    report={'status':'adoption_checks_passed' if accepted else 'limited_detection_no_automatic_retrain','best_epoch':saved['epoch'],
        'project_test':results,'natural_null':null,'protocol':protocol,'data_counts':manifest['counts'],
        'elapsed_seconds':time.time()-start,'limitations':['Source-group separation and local homology screening do not establish species-level independence.',
        'Eligible ACGT windows omit promoters and short ncRNAs; mixture weights are renormalized over represented categories.',
        'The fixed detector is not guaranteed to detect future generator failures; scores do not establish biological function.',
        '512-base local scoring at increasing prompt distance does not measure correlations across a full long sequence.']}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not accepted:raise RuntimeError('Adoption checks failed; checkpoint preserved without a default release or automatic retraining')
    RELEASE.mkdir(parents=True)
    for source,target in [(OUT/'selected.pt','model.pt'),(base.SOURCE,'architecture.py'),(DATA/'manifest.json','data-manifest.json'),
        (OUT/'validation.json','validation.json'),(OUT/'protocol.json','protocol.json'),
        (ROOT/'scripts/score_project_cnn.py','score.py'),(ROOT/'scripts/evaluate_project_cnn.py','evaluate.py'),
        (ROOT/'CNN_FROZEN.md','README.md'),
        (ROOT/'vendor/synthetic_genomes_benchmark/LICENSE','LICENSE.architecture')]:shutil.copy2(source,RELEASE/target)
    retained={int(r['pair_id'].split('/')[-1]) for r in test}
    prompts=[json.loads(line) for line in (ROOT/'data/cnn-project-final/prompts.jsonl').read_text().splitlines()]
    reference=[dict(r,text=r['natural']) for r in prompts if r['id'] in retained]
    (RELEASE/'evaluation-prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reference))
    frozen=base.architecture(RELEASE/'architecture.py').cuda().eval()
    frozen.load_state_dict(torch.load(RELEASE/'model.pt',map_location='cuda',weights_only=False)['model'],strict=True)
    probe=base.encode([r['natural'] for r in test[:8]],'cuda')
    np.testing.assert_allclose(base.score(frozen,probe),base.score(model,probe),atol=1e-6,rtol=1e-6)
    files={p.name:base.digest(p) for p in RELEASE.iterdir() if p.is_file()}
    (RELEASE/'manifest.json').write_text(json.dumps({'version':'cnn-v1','frozen':True,'files_sha256':files,
        'input':'512 ACGT bases per chunk, conditioning prompt excluded',
        'scoring':'mean forward/reverse-complement logits per chunk; sequence logit is mean chunk logits; sigmoid applied after averaging',
        'primary_metric':'Category-specific detection AUROCs averaged with fixed OpenGenome2 mixture weights, renormalized over represented categories; group bootstrap within category.',
        'policy':'No automatic retraining, fine-tuning, recalibration, or replacement. Any future replacement requires explicit user approval and a new version.'},indent=2)+'\n')
    print('FROZEN_PROJECT_CNN',base.digest(RELEASE/'model.pt'),flush=True)


if __name__=='__main__':main()
