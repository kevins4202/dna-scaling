"""Independently validate registered DNA, exclusions, and generation evidence."""
import collections
import hashlib
import json
import pathlib
import sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
DATA=ROOT/'data/distance-2048-v1'
OUT=ROOT/'reports/distance-2048-v1'
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rc(s):return s.translate(str.maketrans('ACGT','TGCA'))[::-1]
def canonical(s):return min(s,rc(s))
def main():
    manifest=json.loads((DATA/'manifest.json').read_text());rows=read(DATA/'prompts.jsonl');natural=read(DATA/'natural.jsonl')
    assert len(rows)==100 and len({r['id'] for r in rows})==100
    assert natural==[dict(r,text=r['natural']) for r in rows]
    for name,expected in manifest['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    train_groups=set()
    for r in read(ROOT/'data/mixture-100m-bounded/train.jsonl'):
        train_groups.add(hashlib.sha256((r['source_shard']+':'+str(r['source_row'])).encode()).hexdigest()[:16])
    development_groups=set();development_sequences=set()
    for fold in ('train','calibration'):
        for r in read(ROOT/f'data/cnn-final-mixed/{fold}.jsonl'):
            if r['domain']=='project':development_groups.add(r['group_id'])
            development_sequences.update(canonical(r[k]) for k in ('natural','generated','null_natural') if k in r)
    seen=set();chunks=set();by_group=collections.defaultdict(list)
    for r in rows:
        assert len(r['prefix'])==256 and len(r['natural'])==2048 and set(r['prefix']+r['natural'])<=set('ACGT')
        assert r['group_id'] not in train_groups|development_groups
        key=canonical(r['prefix']+r['natural']);assert key not in seen;seen.add(key)
        by_group[r['group_id']].append((r['source_offset'],r['source_offset']+2304))
        for i in range(4):
            key=canonical(r['natural'][512*i:512*(i+1)])
            assert key not in development_sequences,'Exact/RC CNN-development match'
            assert key not in chunks,'Duplicate natural chunks';chunks.add(key)
    for intervals in by_group.values():
        intervals.sort()
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),'Overlapping natural regions'
    generations={}
    for size in ('5m','10m'):
        path=DATA/f'generated-{size}.jsonl'
        if not path.exists():continue
        values=read(path);assert len({r['id'] for r in values})==len(values)
        lookup={r['id']:r for r in rows}
        for r in values:
            assert r['id'] in lookup and r['group_id']==lookup[r['id']]['group_id'] and r['seed']==lookup[r['id']]['seed']
            assert len(r['generated'])==2048 and set(r['generated'])<=set('ACGT')
        generations[size]=len(values)
    result={'status':'verified','pairs':len(rows),'source_groups':len(by_group),'nonoverlapping_natural_windows':True,'exact_and_reverse_complement_development_matches':0,'natural_chunk_duplicates':0,'generation_counts':generations,'manifest_sha256':sha(DATA/'manifest.json'),'cnn_sha256':sha(ROOT/'artifacts/cnn-v1/model.pt')}
    if '--complete' in sys.argv:
        assert generations=={'5m':100,'10m':100}
        complete=json.loads((OUT/'complete.json').read_text());assert complete['status']=='complete'
        for name,expected in complete['evidence_sha256'].items():assert sha(OUT/name)==expected
        assert sha(ROOT/'artifacts/cnn-v1/model.pt')==complete['cnn_model_sha256']
        assert (OUT/'distance-curve.png').stat().st_size>1000
        for side in ('before','after'):assert json.loads((OUT/f'cnn-integrity-{side}.json').read_text())['status']=='verified'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/('verification-final.json' if '--complete' in sys.argv else 'verification-data.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
