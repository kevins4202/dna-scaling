"""Independently validate each completed larger-model distance evaluation."""
import argparse,json,math
from family import DATA,ROOT,report_dir,sha,verify

def read(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('size',choices=('50m','100m'));size=parser.parse_args().size
    extension=verify(size);out=report_dir(size);prompts={r['id']:r for r in read(DATA/'prompts.jsonl')};rows=read(DATA/f'generated-{size}.jsonl')
    assert len(rows)==100 and {r['id'] for r in rows}==prompts.keys()
    for row in rows:
        prompt=prompts[row['id']];assert row['group_id']==prompt['group_id'] and row['seed']==prompt['seed'] and row['model']==size
        assert len(row['generated'])==2048 and set(row['generated'])<=set('ACGT')
    metadata=json.loads((DATA/f'generation-{size}.json').read_text())
    assert metadata['status']=='complete' and metadata['parameters']==extension['parameters'] and metadata['count']==100 and metadata['batch_size']==20
    assert metadata['extension_sha256']==sha(DATA/f'extension-{size}.json')
    assert metadata['checkpoint_sha256']==extension['inputs_sha256'][f'checkpoints/{size}/global_step3051/mp_rank_00_model_states.pt']
    assert metadata['output_sha256']==sha(DATA/f'generated-{size}.jsonl')
    assert [c['length'] for c in metadata['dynamic_inference_checks']]==[256,768,1280,1792,2303]
    assert all(math.isfinite(c['max_absolute_logit_difference']) and c['max_absolute_logit_difference']<.15 for c in metadata['dynamic_inference_checks'])
    scores=json.loads((out/f'{size}-scores.json').read_text());assert scores['pairs']==100 and scores['continuation_length']==2048 and scores['source_groups']==90
    assert scores['generated_input_sha256']==metadata['output_sha256'] and scores['natural_input_sha256']==sha(DATA/'natural.jsonl')
    assert {int(r['id']) for r in scores['predictions']}==prompts.keys()
    for row in scores['predictions']:
        assert len(row['natural_chunk_logits'])==len(row['generated_chunk_logits'])==4
        assert all(math.isfinite(x) for key in ('natural_chunk_logits','generated_chunk_logits') for x in row[key])
    complete=json.loads((out/'complete.json').read_text());assert complete['status']=='complete' and complete['models'][size]['parameters']==extension['parameters']
    for name,expected in complete['evidence_sha256'].items():assert sha(ROOT/name)==expected,name
    for side in ('before','after'):assert json.loads((out/f'cnn-integrity-{side}.json').read_text())['status']=='verified'
    release=ROOT/'artifacts/cnn-v1';manifest=json.loads((release/'manifest.json').read_text())
    assert sha(release/'manifest.json')=='73d1c7823755cb722430c937918e67160284d62de64683cc14646dda4cca3f59'
    for name,expected in manifest['files_sha256'].items():assert sha(release/name)==expected,name
    assert (out/'distance-curve.png').stat().st_size>1000
    result={'status':'verified_complete','model':size,'parameters':extension['parameters'],'pairs':100,'source_groups':90,'extension_sha256':sha(DATA/f'extension-{size}.json'),'checkpoint_sha256':metadata['checkpoint_sha256'],'generation_sha256':metadata['output_sha256'],'complete_report_sha256':sha(out/'complete.json'),'frozen_cnn_sha256':sha(release/'model.pt')}
    (out/'verification-final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
