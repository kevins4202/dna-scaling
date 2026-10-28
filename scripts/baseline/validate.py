"""Final immutable-input, generation, scoring and evidence audit."""
import math
from common import ROOT,OUT,DATA,ORIGINAL,guard,read,write,rows,sha

def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def main():
    protocol=guard();smoke=read(OUT/'smoke-verification.json');assert smoke['status']=='verified' and smoke['strict_reload']['strict']
    assert smoke['protocol_sha256']==sha(OUT/'protocol.json')
    assert smoke['deterministic_replay'] and len(smoke['probes'])==5
    assert protocol['inference']['cached'] is False
    for p in smoke['probes']:
        assert p['finite'] and p['cached_used'] is False and p['mode']=='full_prefix_replay'
        assert p['max_replay_logit_difference']==0.0
    prompts=rows(ORIGINAL/'prompts.jsonl');generated=rows(DATA/'generated-evo2_7b_base.jsonl')
    assert len(generated)==100 and len({str(r['id']) for r in generated})==100
    for p,r in zip(prompts,generated):
        for k in ('id','group_id','category','seed','prefix'):assert p[k]==r[k],k
        assert len(r['generated'])==2048 and set(r['generated'])<=set('ACGT')
        assert r['protocol_sha256']==sha(OUT/'protocol.json')
        assert r['checkpoint_sha256']==protocol['checkpoint']['checkpoint_sha256']
    meta=read(DATA/'generation-evo2_7b_base.json');assert meta['status']=='complete' and meta['count']==100
    assert meta['output_sha256']==sha(DATA/'generated-evo2_7b_base.jsonl')
    scores=read(OUT/'released-7b-scores.json');assert len(scores['predictions'])==100
    assert scores['model_sha256']==protocol['cnn_sha256']
    assert scores['natural_input_sha256']==sha(ORIGINAL/'natural.jsonl')
    assert scores['generated_input_sha256']==sha(DATA/'generated-evo2_7b_base.jsonl')
    for p,s in zip(prompts,scores['predictions']):
        assert str(p['id'])==str(s['id']) and p['group_id']==s['group_id'] and p['category']==s['category']
        assert len(s['generated_chunk_logits'])==len(s['natural_chunk_logits'])==4
    result=read(OUT/'complete.json');assert result['status']=='complete' and result['protocol_sha256']==sha(OUT/'protocol.json')
    for name,digest in result['evidence_sha256'].items():assert sha(ROOT/name)==digest
    assert (OUT/'distance-curve.png').stat().st_size>1000
    for name in ('cnn-integrity-before.json','cnn-integrity-after.json'):
        integrity=read(OUT/name);assert integrity['status']=='verified' and integrity['checkpoint_sha256']==protocol['cnn_sha256']
    finite(smoke);finite(meta);finite(scores);finite(result)
    files=[OUT/'protocol.json',OUT/'download-manifest.json',OUT/'smoke-verification.json',OUT/'released-7b-scores.json',OUT/'complete.json',OUT/'distance-curve.png',OUT/'RESULTS.md',OUT/'cnn-integrity-before.json',OUT/'cnn-integrity-after.json',DATA/'generated-evo2_7b_base.jsonl',DATA/'generation-evo2_7b_base.json']
    write(OUT/'verification-final.json',{'status':'verified','pairs':100,'generated_bases_per_prompt':2048,'strict_checkpoint_reload':True,'finite_scores':True,'frozen_cnn_unchanged':True,'inputs_unchanged':True,'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}});print('RELEASED_7B_BASELINE_VERIFIED_COMPLETE',flush=True)
if __name__=='__main__':main()
