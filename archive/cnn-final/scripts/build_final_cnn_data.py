import collections
import hashlib
import json
import pathlib

import prepare_project_cnn as benchmark

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT/'data/cnn-final-mixed'


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Final CNN split already prepared; refusing to change it')
    folds = {f: [] for f in ('train', 'calibration', 'test')}
    for fold in ('train', 'calibration'):
        folds[fold].extend(dict(r, domain='external', generator='megadna' if r['tag']=='megaDNA_bacteriophages' else 'evo2') for r in read(ROOT/f'data/cnn-v1/{fold}.jsonl'))
    for directory, prefix in [('cnn-pilot', 'pilot'), ('cnn-project-final', 'fresh')]:
        base = ROOT/'data'/directory
        prompts = read(base/'prompts.jsonl')
        generated = {}
        for size in ('5m','10m'):
            values=read(base/f'generated-{size}.jsonl')
            mapping={r['id']:r for r in values}
            assert len(values)==len(mapping)==len(prompts)
            assert set(mapping)=={r['id'] for r in prompts}
            metadata=json.loads((base/f'generation-{size}.json').read_text())
            assert metadata['status']=='complete' and metadata['count']==len(prompts)
            generated[size]=mapping
        for size, mapping in generated.items():
            assert len(mapping) == len(prompts)
        for r in prompts:
            fold = 'train' if prefix == 'pilot' else r['fold']
            for size in generated:
                g = generated[size][r['id']]
                assert g['group_id'] == r['group_id']
                assert g['fold']==r['fold'] and len(g['generated'])==512 and set(g['generated'])<=set('ACGT')
                folds[fold].append({'id': f'{prefix}/{r["id"]}/{size}', 'pair_id': f'{prefix}/{r["id"]}',
                    'group_id': r['group_id'], 'tag': r['category'], 'category': r['category'],
                    'fold': fold, 'domain': 'project', 'generator': size,
                    'offset': r['source_offset']+256, 'natural': r['natural'],
                    'generated': g['generated'], 'null_natural': r['null_natural']})
    own_groups = {f: {r['group_id'] for r in rows if r['domain']=='project'} for f, rows in folds.items()}
    assert not (own_groups['train'] & own_groups['calibration'] or own_groups['train'] & own_groups['test'] or own_groups['calibration'] & own_groups['test'])
    seen = set()
    exact_removed = {}
    for fold, rows in folds.items():
        kept = [r for r in rows if not ({benchmark.canonical(r[k]) for k in ('natural','generated','null_natural')} & seen)]
        exact_removed[fold] = len(rows)-len(kept)
        folds[fold] = kept
        seen.update(benchmark.canonical(r[k]) for r in kept for k in ('natural','generated','null_natural'))
    benchmark.OUT = OUT
    removed = {}
    folds['calibration'], removed['calibration'] = benchmark.remove_overlap(folds['train'], folds['calibration'], 'train-calibration')
    folds['test'], removed['test'] = benchmark.remove_overlap(folds['train']+folds['calibration'], folds['test'], 'development-test')
    # Keep both generator examples on exactly the same natural prompts after screening.
    for fold in folds:
        sizes = collections.defaultdict(set)
        for r in folds[fold]:
            if r['domain']=='project':
                sizes[r['pair_id']].add(r['generator'])
        folds[fold] = [r for r in folds[fold] if r['domain']=='external' or sizes[r['pair_id']]=={'5m','10m'}]
    manifest = {'source': 'External benchmark plus both project generators; earlier pilot is training data only.',
                'similarity_removed': removed, 'exact_overlap_removed':exact_removed, 'counts': {}, 'file_sha256': {}}
    for fold, rows in folds.items():
        path = OUT/f'{fold}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        project = [r for r in rows if r['domain']=='project']
        manifest['counts'][fold] = {'all_pairs': len(rows), 'project_pairs_per_generator':len(project)//2,
            'project_source_groups':len({r['group_id'] for r in project}),
            'external_pairs':len(rows)-len(project),
            'project_category_windows':dict(collections.Counter(r['tag'] for r in project if r['generator']=='5m'))}
        manifest['file_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['counts']['test']['project_source_groups']>=50
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
