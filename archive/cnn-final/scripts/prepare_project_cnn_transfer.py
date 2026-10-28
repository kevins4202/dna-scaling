import collections
import hashlib
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT/'data/cnn-project-final'
WEIGHTS = {'gtdb_v220_imgpr':18, 'metagenomes':24, 'imgvr_untagged':3, 'mrna_splice_promoter':9,
           'mrna':9, 'eukaryotic_genic_windows':35, 'organelle':.5, 'ncrna':2, 'promoters':.02}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'prompts.jsonl').exists():
        raise RuntimeError('Transfer dataset already exists; refusing to change it')
    old = [json.loads(x) for x in (ROOT/'data/cnn-pilot/prompts.jsonl').read_text().splitlines()]
    excluded = {(r['source_shard'], r['source_row']) for r in old}
    with (ROOT/'data/mixture-100m-bounded/train.jsonl').open() as handle:
        excluded.update((r['source_shard'], r['source_row']) for r in map(json.loads, handle))
    groups = collections.defaultdict(list)
    for path in sorted((ROOT/'data/reference-independent').glob('*.jsonl')):
        with path.open() as handle:
            for row in map(json.loads, handle):
                key = (row['source_shard'], row['source_row'])
                if key in excluded:
                    continue
                text = row['text'].upper()
                for start in range(0, len(text)-1280+1, 1280):
                    context = text[start:start+1280]
                    if set(context) <= set('ACGT'):
                        groups[key].append({'category':row['category'], 'source_shard':key[0], 'source_row':key[1],
                            'source_offset':row.get('source_offset',0)+start, 'prefix':context[:256],
                            'natural':context[256:768], 'null_natural':context[768:1280]})
    by_category = collections.defaultdict(list)
    for key, values in groups.items():
        by_category[values[0]['category']].append(key)
    pools = {fold:collections.defaultdict(list) for fold in ('train','calibration','test')}
    for category, keys in by_category.items():
        rng = random.Random(f'6214/{category}')
        rng.shuffle(keys)
        n = max(1, round(.2*len(keys)))
        assigned = {'test':keys[:n], 'calibration':keys[n:2*n], 'train':keys[2*n:]}
        for fold, chosen in assigned.items():
            pending = []
            for key in chosen:
                values = groups[key].copy()
                rng.shuffle(values)
                pending.append(values[:8])
            for i in range(8):
                for values in pending:
                    if i < len(values):
                        pools[fold][category].append(values[i])
    selected = []
    for fold, count in [('train',300),('calibration',80),('test',160)]:
        quotas = collections.Counter()
        for _ in range(count):
            available = [k for k,v in pools[fold].items() if quotas[k] < len(v)]
            if not available:
                raise ValueError('Insufficient independent reference windows')
            category = max(available, key=lambda k: WEIGHTS[k]/(quotas[k]+1))
            quotas[category] += 1
        for category, n in quotas.items():
            for row in pools[fold][category][:n]:
                row = dict(row, fold=fold)
                row['group_id'] = hashlib.sha256((row['source_shard']+':'+str(row['source_row'])).encode()).hexdigest()[:16]
                row['id'] = len(selected)
                selected.append(row)
    (OUT/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    manifest = {'seed':6214, 'fresh_windows':len(selected), 'reference':'data/reference-independent',
        'old_pilot_policy':'All 100 old pilot groups are development/training only; none is a final calibration or test group.',
        'exclusions':'All source rows used by LM training or the CNN pilot excluded from fresh reference pool.',
        'counts':dict(collections.Counter(r['fold'] for r in selected)),
        'groups':{f:len({r['group_id'] for r in selected if r['fold']==f}) for f in pools},
        'categories':{f:dict(collections.Counter(r['category'] for r in selected if r['fold']==f)) for f in pools},
        'limitations':['Only source rows with 1280 contiguous ACGT bases are eligible; promoters and short ncRNAs are not represented.',
        'Source-row separation is not species independence; cross-fold similarity screening is required before CNN training.',
        'Multiple windows can share a source group; evaluation must resample complete groups.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
