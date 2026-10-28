import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

import score_project_cnn as scoring

ROOT=pathlib.Path(__file__).resolve().parents[1]
RELEASE=ROOT/'artifacts/cnn-v1'
OUT=ROOT/'reports/cnn-final'


def main():
    torch.set_num_threads(4)
    seen_groups,seen_sequences=set(),set()
    for fold in ('train','calibration','test'):
        dataset=[json.loads(x) for x in (ROOT/f'data/cnn-final-mixed/{fold}.jsonl').read_text().splitlines()]
        groups={r['group_id'] for r in dataset}
        assert not groups&seen_groups
        keys=set()
        for row in dataset:
            for field in ('natural','generated','null_natural'):
                text=row[field]
                assert len(text)==512 and set(text)<=set('ACGT')
                rc=text.translate(str.maketrans('ACGT','TGCA'))[::-1]
                keys.add(min(text,rc))
        assert not keys&seen_sequences
        seen_groups.update(groups);seen_sequences.update(keys)
    model,manifest=scoring.load_release(RELEASE,'cuda')
    reference=[json.loads(x) for x in (RELEASE/'evaluation-prompts.jsonl').read_text().splitlines()]
    assert len(reference)>=50
    training=json.loads((RELEASE/'validation.json').read_text())
    results={}
    for size in ('5m','10m'):
        full={r['id']:r for r in map(json.loads,(ROOT/f'data/cnn-project-final/generated-{size}.jsonl').read_text().splitlines())}
        path=OUT/f'test-generated-{size}.jsonl'
        rows=[full[r['id']] for r in reference]
        assert all(g['group_id']==r['group_id'] for g,r in zip(rows,reference))
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        target=OUT/f'{size}-evaluation.json'
        subprocess.run([sys.executable,str(RELEASE/'evaluate.py'),'--natural',str(RELEASE/'evaluation-prompts.jsonl'),
            '--generated',str(path),'--output',str(target),'--device','cuda'],check=True)
        result=json.loads(target.read_text())
        expected=training['project_test'][size]['mixture_weighted_stratified_auc']
        assert abs(result['mixture_weighted_stratified_auc']-expected)<.001
        results[size]={'auc':result['mixture_weighted_stratified_auc'],
            'ci95':result['source_group_bootstrap_ci95'],'difference_from_training_verification':result['mixture_weighted_stratified_auc']-expected}
    text=reference[0]['text']
    gpu,chunks=scoring.score_text(model,text,'cuda')
    rc=text.translate(str.maketrans('ACGT','TGCA'))[::-1]
    np.testing.assert_allclose(gpu,scoring.score_text(model,rc,'cuda')[0],atol=1e-6,rtol=1e-6)
    long_score,long_chunks=scoring.score_text(model,text*4,'cuda')
    np.testing.assert_allclose(long_score,np.mean(long_chunks),atol=1e-6,rtol=1e-6)
    cpu_model,_=scoring.load_release(RELEASE,'cpu')
    cpu,_=scoring.score_text(cpu_model,text,'cpu')
    np.testing.assert_allclose(cpu,gpu,atol=.002,rtol=.0002)
    report={'status':'verified','model_sha256':manifest['files_sha256']['model.pt'],
        'verification_scope':'Artifact integrity, data separation, and reproducible scoring; not a claim of sufficient detection sensitivity.',
        'prespecified_sensitivity_gate_passed':manifest.get('prespecified_sensitivity_gate_passed',False),
        'manifest_sha256':hashlib.sha256((RELEASE/'manifest.json').read_bytes()).hexdigest(),
        'parameters':sum(p.numel() for p in model.parameters()),'evaluation_pairs':len(reference),
        'evaluation_source_groups':len({r['group_id'] for r in reference}),'strict_reload':True,
        'cross_fold_source_groups_disjoint':True,'cross_fold_exact_reverse_complement_duplicates':0,
        'cpu_gpu_absolute_logit_difference':abs(cpu-gpu),'reverse_complement_invariance':True,
        'four_chunk_aggregation':True,'frozen_cli_evaluation':results}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
