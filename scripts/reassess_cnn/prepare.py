import collections
import gzip
import importlib.util
import subprocess
from pathlib import Path

from common import ROOT, OUT, DATA, MODELS, GENERATIONS, OLD_SCORES, sha, read, rows, write, guard


def fasta(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        name, parts = None, []
        for line in f:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(parts).upper()
                name, parts = line[1:].strip(), []
            else:
                parts.append(line.strip())
        if name is not None:
            yield name, ''.join(parts).upper()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        guard()
        print('Registered screened protocol verified', flush=True)
        return
    original = read(ROOT / 'reports/cnn-original-v1/protocol.json')
    assert read(ROOT / 'reports/cnn-original-v1/verification-final.json')['folds'] == 210
    prompts = rows(ROOT / 'data/distance-2048-v1/prompts.jsonl')
    generations = {m: {str(r['id']): r for r in rows(ROOT / GENERATIONS[m])} for m in MODELS}
    assert len(prompts) == 100 and all(len(g) == 100 for g in generations.values())
    checkpoints = []
    for i in range(30):
        folder = ROOT / 'reports/cnn-original-v1/folds' / f'{i:03d}'
        meta = read(folder / 'complete.json')
        task = meta['task']
        assert task['mode'] == 'chunks' and task['n_eval_chunks'] == 32
        assert sha(folder / 'selected.pt') == meta['files_sha256']['selected.pt']
        checkpoints.append({'index': i, 'path': str((folder / 'selected.pt').relative_to(ROOT)),
                            'domain': task['domain'], 'holdout': task['holdout'],
                            'weight': 1 / (3 * len(original['domains'][task['domain']]))})
    assert abs(sum(c['weight'] for c in checkpoints) - 1) < 1e-12
    protected = ['data/distance-2048-v1/prompts.jsonl', 'data/distance-2048-v1/natural.jsonl',
                 'data/distance-2048-v1/manifest.json', 'data/mixture-100m-bounded/train.fasta',
                 'data/mixture-100m-bounded/train.jsonl', 'reports/cnn-original-v1/protocol.json',
                 'reports/cnn-original-v1/verification-final.json', 'reports/cnn-original-v1/dataset-inventory.json',
                 *GENERATIONS.values(), *OLD_SCORES.values(), *[c['path'] for c in checkpoints]]
    protected += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'scripts/reassess_cnn').glob('*')) if p.is_file()]
    protected += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'artifacts/cnn-v1').rglob('*')) if p.is_file()]
    upstream = ROOT / 'vendor/synthetic_genomes_benchmark/scripts/classifier/deep_detector_dilated_resnet1d.py'
    protected.append(str(upstream.relative_to(ROOT)))
    inputs = {name: sha(ROOT / name) for name in protected}
    # All benchmark sequence bytes used for any fit/calibration/test fold are screened.
    for name, expected in original['inputs_sha256'].items():
        if name.startswith('data/cnn-benchmark/'):
            assert sha(ROOT / name) == expected
            inputs[name] = expected
    design = {'experiment': 'cnn-reassessment-v1', 'models': MODELS, 'checkpoints': checkpoints,
              'selection': 'First registered 32-chunk configuration, all 30 folds; no checkpoint or ensemble selection on reassessment outcomes.',
              'scoring': 'Two nonoverlapping 1024-base windows of generated continuation only. Average raw logits across the 30 CNNs with equal total weight per domain, then average the two windows for sequence score. Per-domain ensembles reported as fixed secondary diagnostics. No reverse-complement augmentation or calibration change.',
              'eligibility': 'Exclude an entire original source group for every generator if any natural prefix+continuation or any of the six generated continuations has a reported alignment >=256 query bases at >=90% identity to the complete original CNN corpus or the known small-LM training FASTA. Exclude original source rows present in small-LM training. Apply before any CNN inference. Report all exclusions and original IDs.',
              'screening': 'Full original CNN corpus including its heldout examples, stronger than excluding just fit/calibration. Two minimap2 passes: asm10 and sensitive sr with k11/w5; both strands, CIGAR output, all reported secondary alignments. Exact full sequence/reverse-complement equality checked as well.',
              'analysis': {'bootstrap': 2000, 'seed': 6220, 'unit': 'whole source group stratified by category, same resamples for all six models and both windows',
                           'primary': 'Original category-mixture-weighted within-category aggregate AUROC, weights renormalized over retained categories',
                           'contrasts': 'All pairwise aggregate differences; released minus each small model; per-category and per-domain results; old CNN rescored statistically on exactly the same retained IDs',
                           'scope': 'Exploratory, unadjusted percentile intervals; no claim of independent confirmation or biological function'},
              'limitations': ['No detected sequence overlap at the registered screening thresholds is not proof of species/phylogenetic independence; source taxonomy is incomplete.',
                             'Released 7B pretraining-source overlap cannot be established from available records.',
                             'An exclusion triggered by one generator is applied to all six, so the result concerns the common screened panel and omits detected memorization cases; those exclusions remain explicit.',
                             'Same previously examined prompt panel, one generation and one small-model training seed; uncertainty excludes training-seed variation.',
                             'Original detectors were trained on long original benchmark sequences; transfer to 2048-base continuations is not guaranteed. These scores are not the original 32-chunk benchmark.',
                             'Original item-level calibration splitting limitations remain; weights are not retrained or chosen using these results.'],
              'inputs_sha256': inputs}
    if (OUT / 'design.json').exists():
        assert read(OUT / 'design.json') == design
    else:
        write(OUT / 'design.json', design)
    query_map = {}
    with (DATA / 'queries.fasta').open('w') as f:
        for p in prompts:
            ident = str(p['id'])
            natural = p['prefix'] + p['natural']
            assert len(natural) == 2304 and set(natural) <= set('ACGT')
            sequence_map = {'natural': natural}
            for m in MODELS:
                r = generations[m][ident]
                assert r['group_id'] == p['group_id'] and r['seed'] == p['seed']
                assert len(r['generated']) == 2048 and set(r['generated']) <= set('ACGT')
                sequence_map[m] = r['generated']
            for label, text in sequence_map.items():
                key = f'q{len(query_map)}'
                query_map[key] = {'id': ident, 'group_id': p['group_id'], 'kind': label}
                f.write(f'>{key}\n{text}\n')
    write(DATA / 'query-map.json', query_map)
    inventory = read(ROOT / 'reports/cnn-original-v1/dataset-inventory.json')
    with (DATA / 'cnn-corpus.fasta').open('w') as f:
        seen = set()
        for r in inventory:
            for key in ('orig', 'syn'):
                path = ROOT / 'data/cnn-original-v1' / r[key]
                if path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                text = ''.join(s for _, s in fasta(path))
                f.write(f'>cnn{len(seen)}_{r["tag"]}_{r["id"]}_{key}\n{text}\n')
    # Full-query exact checks catch low-complexity full duplicates even if alignment seeding masks them.
    query_text = dict(fasta(DATA / 'queries.fasta'))
    canonical = lambda s: min(s, s.translate(str.maketrans('ACGT', 'TGCA'))[::-1])
    query_canon = collections.defaultdict(list)
    for key, text in query_text.items():
        query_canon[canonical(text)].append(key)
    hits = []
    for corpus in (DATA / 'cnn-corpus.fasta', ROOT / 'data/mixture-100m-bounded/train.fasta'):
        for name, text in fasta(corpus):
            for key in query_canon.get(canonical(text), []):
                hits.append({'query': key, 'target': name, 'corpus': str(corpus.relative_to(ROOT)), 'reason': 'exact_full_sequence'})
    training_rows = {(r['source_shard'], r['source_row']) for r in rows(ROOT / 'data/mixture-100m-bounded/train.jsonl')}
    bad_groups = {p['group_id'] for p in prompts if (p['source_shard'], p['source_row']) in training_rows}
    commands = []
    for label, target in [('cnn', DATA / 'cnn-corpus.fasta'), ('lm', ROOT / 'data/mixture-100m-bounded/train.fasta')]:
        for preset in ('asm10', 'sr'):
            output = DATA / f'{label}-{preset}.paf'
            cmd = ['minimap2', '-x', preset, '-c', '--secondary=yes', '-N', '100', '-t', '4']
            if preset == 'sr':
                cmd += ['-k', '11', '-w', '5']
            cmd += [str(target), str(DATA / 'queries.fasta')]
            commands.append(cmd)
            with output.open('w') as f, (OUT / f'{label}-{preset}.log').open('w') as log:
                subprocess.run(cmd, stdout=f, stderr=log, check=True)
            for line in output.open():
                fields = line.rstrip().split('\t')
                span = int(fields[3]) - int(fields[2])
                identity = int(fields[9]) / max(1, int(fields[10]))
                if span >= 256 and identity >= .9:
                    hits.append({'query': fields[0], 'target': fields[5], 'corpus': label,
                                 'pass': preset, 'query_span': span, 'identity': identity})
    for hit in hits:
        bad_groups.add(query_map[hit['query']]['group_id'])
    retained = [p for p in prompts if p['group_id'] not in bad_groups]
    assert len(retained) >= 2, 'No usable common screened panel; do not score or substitute prompts'
    write(DATA / 'eligible.json', retained)
    write(OUT / 'overlap-audit.json', {'original_pairs': 100, 'retained_pairs': len(retained),
          'retained_groups': len({p['group_id'] for p in retained}),
          'retained_category_counts': dict(collections.Counter(p['category'] for p in retained)),
          'excluded_source_groups': sorted(bad_groups), 'excluded_ids': [str(p['id']) for p in prompts if p['group_id'] in bad_groups],
          'hits': hits, 'commands': commands, 'selection_before_cnn_inference': True,
          'no_detected_overlap_for_retained_records': True})
    protected_screen = [DATA / 'eligible.json', DATA / 'queries.fasta', DATA / 'query-map.json', DATA / 'cnn-corpus.fasta',
                        OUT / 'overlap-audit.json', *sorted(DATA.glob('*.paf'))]
    write(OUT / 'protocol.json', {'design_sha256': sha(OUT / 'design.json'),
          'pairs': len(retained), 'groups': len({p['group_id'] for p in retained}),
          'fixed_category_weights': read(ROOT / 'data/distance-2048-v1/manifest.json')['fixed_category_weights'],
          'screening_sha256': {str(p.relative_to(ROOT)): sha(p) for p in protected_screen}})
    guard()
    print('SCREENED_PROTOCOL_REGISTERED', len(retained), 'pairs', len(bad_groups), 'excluded groups', flush=True)


if __name__ == '__main__':
    main()
