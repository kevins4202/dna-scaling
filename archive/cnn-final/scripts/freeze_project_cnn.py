import json
import pathlib
import shutil

import torch

import train_project_cnn as base

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/cnn-final'
RELEASE=ROOT/'artifacts/cnn-v1'


def main():
    if RELEASE.exists():raise RuntimeError('Frozen CNN exists; refusing to replace it')
    report=json.loads((OUT/'validation.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert base.digest(base.SOURCE)==protocol['source_sha256']
    assert base.digest(ROOT/'data/cnn-final-mixed/manifest.json')==protocol['data_manifest_sha256']
    selected=torch.load(OUT/'selected.pt',map_location='cpu',weights_only=False)
    assert selected['epoch']==report['best_epoch']
    model=base.architecture().eval();model.load_state_dict(selected['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==1227905
    assert all(torch.isfinite(p).all() for p in model.state_dict().values())
    RELEASE.mkdir(parents=True)
    for source,target in [(OUT/'selected.pt','model.pt'),(base.SOURCE,'architecture.py'),
        (ROOT/'data/cnn-final-mixed/manifest.json','data-manifest.json'),
        (OUT/'validation.json','validation.json'),(OUT/'protocol.json','protocol.json'),
        (ROOT/'scripts/score_project_cnn.py','score.py'),(ROOT/'scripts/evaluate_project_cnn.py','evaluate.py'),
        (ROOT/'CNN_FROZEN.md','README.md'),(ROOT/'vendor/synthetic_genomes_benchmark/LICENSE','LICENSE.architecture')]:
        shutil.copy2(source,RELEASE/target)
    test=[json.loads(x) for x in (ROOT/'data/cnn-final-mixed/test.jsonl').read_text().splitlines()]
    ids={int(r['pair_id'].split('/')[-1]) for r in test}
    prompts=[json.loads(x) for x in (ROOT/'data/cnn-project-final/prompts.jsonl').read_text().splitlines()]
    reference=[dict(r,text=r['natural']) for r in prompts if r['id'] in ids]
    (RELEASE/'evaluation-prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reference))
    manifest={'version':'cnn-v1','frozen':True,'validation_status':'fixed_diagnostic_limited_sensitivity',
        'prespecified_sensitivity_gate_passed':report['status']=='adoption_checks_passed',
        'validation_scope':'Fixed distinguishability diagnostic; not validated as a sole project quality measure or RL reward.',
        'files_sha256':{p.name:base.digest(p) for p in RELEASE.iterdir() if p.is_file()},
        'input':'512 ACGT bases per chunk, prompt excluded',
        'scoring':'mean forward/reverse-complement logits per chunk; sequence logit is mean chunk logits; sigmoid applied after averaging',
        'primary_metric':'Category-specific detection AUROCs averaged with fixed OpenGenome2 mixture weights, renormalized over represented categories; source-group bootstrap within category.',
        'policy':'Weights and scoring code stay fixed for all future models. No automatic retraining, fine-tuning, recalibration, or replacement. Any replacement requires explicit user approval and a new version.',
        'release_note':'Frozen as a qualified diagnostic at user request. Original failed sensitivity gate and held-out results are retained unchanged; no test-driven retraining was performed.'}
    (RELEASE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FROZEN_DIAGNOSTIC',base.digest(RELEASE/'model.pt'),flush=True)


if __name__=='__main__':main()
