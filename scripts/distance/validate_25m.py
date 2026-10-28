"""Validate the new generation and analysis against immutable baseline inputs."""
import hashlib,json,math,pathlib
from register_25m import verify
ROOT=pathlib.Path(__file__).resolve().parents[2];DATA=ROOT/'data/distance-2048-v1';OUT=ROOT/'reports/distance-2048-25m-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    extension=verify();prompts={r['id']:r for r in read(DATA/'prompts.jsonl')};rows=read(DATA/'generated-25m.jsonl')
    assert len(rows)==100 and {r['id'] for r in rows}==prompts.keys()
    for r in rows:
        p=prompts[r['id']];assert r['group_id']==p['group_id'] and r['seed']==p['seed'] and r['model']=='25m'
        assert len(r['generated'])==2048 and set(r['generated'])<=set('ACGT')
    generation=json.loads((DATA/'generation-25m.json').read_text());assert generation['status']=='complete' and generation['parameters']==24886444 and generation['count']==100 and generation['batch_size']==20
    assert generation['checkpoint_sha256']==extension['inputs_sha256']['checkpoints/25m/global_step3051/mp_rank_00_model_states.pt']
    assert generation['output_sha256']==sha(DATA/'generated-25m.jsonl')
    assert [c['length'] for c in generation['dynamic_inference_checks']]==[256,768,1280,1792,2303]
    assert all(math.isfinite(c['max_absolute_logit_difference']) and c['max_absolute_logit_difference']<.15 for c in generation['dynamic_inference_checks'])
    report=json.loads((OUT/'25m-scores.json').read_text());assert report['pairs']==100 and report['continuation_length']==2048 and report['source_groups']==90
    assert report['generated_input_sha256']==sha(DATA/'generated-25m.jsonl') and report['natural_input_sha256']==sha(DATA/'natural.jsonl')
    assert {int(r['id']) for r in report['predictions']}==prompts.keys()
    for r in report['predictions']:
        assert len(r['natural_chunk_logits'])==len(r['generated_chunk_logits'])==4
        assert all(math.isfinite(x) for key in ('natural_chunk_logits','generated_chunk_logits') for x in r[key])
    complete=json.loads((OUT/'complete.json').read_text());assert complete['status']=='complete'
    for name,expected in complete['evidence_sha256'].items():assert sha(ROOT/name)==expected
    for side in ('before','after'):assert json.loads((OUT/f'cnn-integrity-{side}.json').read_text())['status']=='verified'
    assert sha(ROOT/'artifacts/cnn-v1/model.pt')=='33621e4a90aafbb6c925ff52f7b6c45d74301a66f902cc561a05b83bc3da14b5'
    assert (OUT/'distance-curve.png').stat().st_size>1000
    result={'status':'verified_complete','model':'25m','parameters':24886444,'pairs':100,'source_groups':90,'extension_sha256':sha(DATA/'extension-25m.json'),'checkpoint_sha256':generation['checkpoint_sha256'],'generation_sha256':generation['output_sha256'],'complete_report_sha256':sha(OUT/'complete.json'),'frozen_cnn_sha256':sha(ROOT/'artifacts/cnn-v1/model.pt')}
    (OUT/'verification-final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
