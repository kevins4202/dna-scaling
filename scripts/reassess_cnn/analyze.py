import collections
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, OUT, DATA, MODELS, OLD_SCORES, sha, read, write, guard


def estimates(records, samples, weights, dimensions):
    out = np.zeros((len(samples), dimensions))
    categories = sorted({r['category'] for r in records})
    total = sum(weights[c] for c in categories)
    for category in categories:
        part = [r for r in records if r['category'] == category]
        counts = np.array([[s[r['group_id']] for r in part] for s in samples], dtype=float)
        assert np.all(counts.sum(axis=1) > 0)
        for j in range(dimensions):
            a = np.array([r['natural'][j] for r in part])
            b = np.array([r['generated'][j] for r in part])
            delta = b[:, None] - a[None, :]
            comparisons = (delta > 0).astype(float) + .5 * (delta == 0)
            out[:, j] += weights[category] / total * np.einsum('bi,ij,bj->b', counts, comparisons, counts, optimize=True) / counts.sum(axis=1) ** 2
    assert np.isfinite(out).all()
    return out


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def main():
    design = guard()
    protocol = read(OUT / 'protocol.json')
    selected = read(DATA / 'eligible.json')
    score = read(OUT / 'scores.json')
    assert score['protocol_sha256'] == sha(OUT / 'protocol.json')
    index = {(r['id'], r['model']): r for r in score['records']}
    assert len(index) == len(selected) * 7
    groups = collections.defaultdict(set)
    for p in selected:
        groups[p['category']].add(p['group_id'])
    rng = np.random.default_rng(6220)
    samples = [collections.Counter({p['group_id']: 1 for p in selected})]
    for _ in range(2000):
        sample = collections.Counter()
        for category in sorted(groups):
            keys = sorted(groups[category])
            sample.update(rng.choice(keys, len(keys), replace=True))
        samples.append(sample)
    values, categories, arrays, old_values = {}, {}, {}, {}
    for model in MODELS:
        records = []
        for p in selected:
            n, g = index[(str(p['id']), 'natural')], index[(str(p['id']), model)]
            assert n['group_id'] == g['group_id'] == p['group_id']
            records.append({'category': p['category'], 'group_id': p['group_id'],
                            'natural': n['window_logits'] + [n['aggregate_logit']],
                            'generated': g['window_logits'] + [g['aggregate_logit']]})
        a = estimates(records, samples, protocol['fixed_category_weights'], 3)
        arrays[model] = a
        values[model] = {'windows': [{'auc': float(a[0, j]), 'ci95': interval(a[1:, j])} for j in range(2)],
                         'aggregate': {'auc': float(a[0, 2]), 'ci95': interval(a[1:, 2])},
                         'second_minus_first': {'difference': float(a[0, 1] - a[0, 0]), 'ci95': interval(a[1:, 1] - a[1:, 0])}}
        for category in sorted(groups):
            c = estimates([r for r in records if r['category'] == category], samples, {category: 1}, 3)
            categories.setdefault(category, {'groups': len(groups[category]), 'pairs': sum(p['category'] == category for p in selected), 'models': {}})['models'][model] = {'auc': float(c[0, 2]), 'ci95': interval(c[1:, 2])}
        old = {str(r['id']): r for r in read(ROOT / OLD_SCORES[model])['predictions']}
        old_records = [{'category': p['category'], 'group_id': p['group_id'],
                        'natural': [old[str(p['id'])]['natural_logit']],
                        'generated': [old[str(p['id'])]['generated_logit']]} for p in selected]
        old_a = estimates(old_records, samples, protocol['fixed_category_weights'], 1)
        old_values[model] = {'auc': float(old_a[0, 0]), 'ci95': interval(old_a[1:, 0]),
                             'new_minus_old': {'difference': float(a[0, 2] - old_a[0, 0]), 'ci95': interval(a[1:, 2] - old_a[1:, 0])}}
    domain_results = {}
    for domain in ('euk', 'prok', 'vir'):
        domain_results[domain] = {}
        for model in MODELS:
            records = [{'category': p['category'], 'group_id': p['group_id'],
                        'natural': [float(np.mean(index[(str(p['id']), 'natural')]['domain_window_logits'][domain]))],
                        'generated': [float(np.mean(index[(str(p['id']), model)]['domain_window_logits'][domain]))]} for p in selected]
            a = estimates(records, samples, protocol['fixed_category_weights'], 1)
            domain_results[domain][model] = {'auc': float(a[0, 0]), 'ci95': interval(a[1:, 0])}
    contrasts = {}
    for first, second in itertools.combinations(MODELS, 2):
        delta = arrays[second][:, 2] - arrays[first][:, 2]
        contrasts[f'{second}_minus_{first}'] = {'difference': float(delta[0]), 'ci95': interval(delta[1:])}
    write(OUT / 'analysis.json', {'protocol_sha256': sha(OUT / 'protocol.json'), 'pairs': len(selected),
          'groups': len({p['group_id'] for p in selected}), 'models': values, 'categories': categories,
          'paired_aggregate_differences': contrasts, 'per_domain_ensemble': domain_results,
          'old_cnn_on_identical_retained_ids': old_values, 'limitations': design['limitations']})
    lines = ['# Six-model reassessment with reproduced CNN ensemble', '',
             f'{len(selected)} matched pairs from {len({p["group_id"] for p in selected})} source groups retained after the registered overlap screen. '
             'All six generators use identical retained IDs and the same fixed domain-balanced 30-detector ensemble. '
             'Higher AUROC means easier detection. Scores cover two 1024-base windows of new sequence, not the original long-sequence 32-chunk benchmark.', '',
             '| Generator | New CNN AUROC | 95% source-group interval | Old CNN, same IDs |',
             '|---|---:|---|---:|']
    for m in MODELS:
        a = values[m]['aggregate']
        lines.append(f'| {m} | {a["auc"]:.4f} | {a["ci95"][0]:.4f} to {a["ci95"][1]:.4f} | {old_values[m]["auc"]:.4f} |')
    lines += ['', 'Intervals and contrasts are descriptive and unadjusted. Reusing the panel and incomplete taxonomy limit independent confirmation. '
              'Overlap means no detected match at the declared sequence thresholds against available known training files; released 7B pretraining overlap remains unknown.', '',
              '## Paired aggregate differences', '', '| Contrast | Difference | 95% interval |', '|---|---:|---|']
    for label, a in contrasts.items():
        lines.append(f'| {label} | {a["difference"]:.4f} | {a["ci95"][0]:.4f} to {a["ci95"][1]:.4f} |')
    lines += ['', '## Limitations', ''] + ['- ' + x for x in design['limitations']]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, model in enumerate(MODELS):
        a = values[model]['aggregate']
        axes[0].errorbar(i, a['auc'], yerr=[[max(0, a['auc'] - a['ci95'][0])], [max(0, a['ci95'][1] - a['auc'])]], fmt='o', capsize=3)
        axes[1].plot([512, 1536], [r['auc'] for r in values[model]['windows']], 'o-', label=model)
    axes[0].set_xticks(range(6), MODELS, rotation=25)
    axes[0].set_title('Matched screened panel: aggregate')
    axes[1].set_xticks([512, 1536], ['0–1024', '1024–2048'])
    axes[1].set_title('New-sequence windows')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(.5, color='gray', linestyle='--')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Detection AUROC')
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=180)
    plt.close(fig)
    print('ANALYSIS_COMPLETE', {m: values[m]['aggregate'] for m in MODELS}, flush=True)


if __name__ == '__main__':
    main()
