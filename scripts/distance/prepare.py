"""Register a held-out 2048-base experiment without consulting CNN predictions."""
import collections
import hashlib
import json
import pathlib
import random
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT/'data/distance-2048-v1'
WEIGHTS = {'gtdb_v220_imgpr':18, 'metagenomes':24, 'imgvr_untagged':3,
           'mrna_splice_promoter':9, 'mrna':9, 'eukaryotic_genic_windows':35,
           'organelle':.5, 'ncrna':2, 'promoters':.02}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open() as handle:
        yield from (json.loads(line) for line in handle if line.strip())

def gid(row):
    return hashlib.sha256((row['source_shard']+':'+str(row['source_row'])).encode()).hexdigest()[:16]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        manifest=json.loads((OUT/'manifest.json').read_text())
        for name, sha in manifest['inputs_sha256'].items():
            if digest(ROOT/name)!=sha: raise ValueError('Registered experiment input changed: '+name)
        print('REGISTERED DATA VERIFIED', manifest['pairs']); return
    excluded={gid(r) for r in read(ROOT/'data/mixture-100m-bounded/train.jsonl')}
    development=[]
    for fold in ('train','calibration'):
        for row in read(ROOT/f'data/cnn-final-mixed/{fold}.jsonl'):
            if row['domain']=='project': excluded.add(row['group_id'])
            development.extend(row[key] for key in ('natural','generated','null_natural') if key in row)
    previous={r['group_id'] for r in read(ROOT/'data/cnn-project-final/prompts.jsonl') if r['fold']=='test'}
    candidates=[]; seen=set()
    for path in sorted((ROOT/'data/reference-independent').glob('*.jsonl')):
        for row in read(path):
            group=gid(row)
            if group in excluded: continue
            text=row['text'].upper()
            for start in range(0,len(text)-2304+1,2304):
                sequence=text[start:start+2304]
                if not set(sequence)<=set('ACGT'): continue
                key=(group,row.get('source_offset',0)+start)
                if key in seen: continue
                seen.add(key)
                candidates.append(dict(category=row['category'],group_id=group,
                    source_shard=row['source_shard'],source_row=row['source_row'],
                    source_offset=key[1],prefix=sequence[:256],natural=sequence[256:]))
    (OUT/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    (OUT/'development.fasta').write_text(''.join(f'>d{i}\n{s}\n' for i,s in enumerate(sorted(set(development)))))
    (OUT/'candidates.fasta').write_text(''.join(f'>q{i}\n{r["prefix"]+r["natural"]}\n' for i,r in enumerate(candidates)))
    with (OUT/'similarity.paf').open('w') as output:
        subprocess.run(['minimap2','-x','asm10','-t','4','-N','50',str(OUT/'development.fasta'),str(OUT/'candidates.fasta')],stdout=output,check=True)
    rejected=set()
    for line in (OUT/'similarity.paf').read_text().splitlines():
        p=line.split('\t'); span=int(p[3])-int(p[2]); aligned=int(p[10])
        if span>=256 and int(p[9])/max(1,aligned)>=.90: rejected.add(int(p[0][1:]))
    pools=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,row in enumerate(candidates):
        if i not in rejected: pools[row['category']][row['group_id']].append(row)
    ordered={}
    for category, groups in pools.items():
        keys=sorted(groups); random.Random('6219/'+category).shuffle(keys)
        for key in keys: random.Random('6219/'+key).shuffle(groups[key])
        ordered[category]=[groups[key][i] for i in range(max(map(len,groups.values()))) for key in keys if i<len(groups[key])]
    quotas=collections.Counter()
    for _ in range(100):
        available=[k for k,v in ordered.items() if quotas[k]<len(v)]
        if not available: raise ValueError('Insufficient screened reference windows')
        quotas[max(available,key=lambda k:WEIGHTS[k]/(quotas[k]+1))]+=1
    selected=[]
    for category in sorted(quotas):
        for row in ordered[category][:quotas[category]]:
            selected.append(dict(row,id=len(selected),seed=621900+len(selected)))
    natural=[dict(r,text=r['natural']) for r in selected]
    (OUT/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (OUT/'natural.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in natural))
    inputs=['data/distance-2048-v1/prompts.jsonl','data/distance-2048-v1/natural.jsonl',
            'artifacts/cnn-v1/manifest.json','artifacts/cnn-v1/model.pt']
    for size in ('5m','10m'): inputs.append(f'checkpoints/{size}/global_step3051/mp_rank_00_model_states.pt')
    manifest={'version':'distance-2048-v1','pairs':100,'prefix_bases':256,'continuation_bases':2048,
        'windows':[[i*512,(i+1)*512] for i in range(4)],'seed':6219,
        'sampling':'ACGT-constrained categorical, temperature 1, no top-k/top-p or repetition penalty; same prompt and seed per model',
        'source_groups':len({r['group_id'] for r in selected}),'category_counts':dict(quotas),
        'category_groups':{k:len({r['group_id'] for r in selected if r['category']==k}) for k in quotas},
        'eligible_before_similarity':len(candidates),'similarity_rejected':len(rejected),
        'eligible_after_similarity':{k:len(v) for k,v in ordered.items()},
        'fixed_category_weights':WEIGHTS,'missing_categories':sorted(set(WEIGHTS)-set(quotas)),
        'represented_reference_weight':sum(WEIGHTS[k] for k in quotas),
        'previous_cnn_test_source_groups_reused':len(previous & {r['group_id'] for r in selected}),
        'exclusions':'LM-training source rows and CNN project training/calibration source groups excluded. Minimap2 asm10 screen against all external/project CNN training/calibration natural, generated and null windows: reject query alignment span >=256 at >=90% identity.',
        'analysis':{'primary':'Fixed mixture-weighted within-category AUROC at each distance and mean-logit sequence aggregate',
          'bootstrap':'2000 category-stratified source-group resamples; identical resamples across models/distances',
          'contrasts':['10m minus 5m at each distance and aggregate','last minus first window for each model'],
          'seed':6220,'intervals':'95% percentile, descriptive unadjusted intervals; no confirmatory significance claims'},
        'limitations':['Eligibility requires 2304 contiguous ACGT bases; missing/rare categories and limited eukaryotic source groups constrain inference.',
          'Existing independent reference pool and some CNN test source groups were inspected previously; this is a new distance analysis, not a new untouched biological benchmark.',
          'Source rows are not guaranteed species independent. Repeated windows are clustered by source.',
          'Frozen CNN sensitivity gate failed for 5M; chance detection does not establish biological realism.'],
        'inputs_sha256':{p:digest(ROOT/p) for p in inputs}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
