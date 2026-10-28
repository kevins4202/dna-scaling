import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from prepare import OUT, digest, write


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    grouped = defaultdict(list)
    completed = []
    for index, task in enumerate(protocol['tasks']):
        folder = OUT / 'folds' / f'{index:03d}'
        if not (folder / 'complete.json').exists():
            continue
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete['task'] == task
        assert complete['protocol_sha256'] == digest(OUT / 'protocol.json')
        for name, sha in complete['files_sha256'].items():
            if name != 'selected.pt':
                assert digest(folder / name) == sha
        result = json.loads((folder / 'result.json').read_text())
        predictions = json.loads((folder / 'predictions.json').read_text())
        if task['mode'] == 'chunks':
            assert abs(roc_auc_score(predictions[0]['labels'], predictions[0]['probabilities']) - result['auc']) < 1e-12
            grouped[('chunks', task['domain'], task['n_eval_chunks'])].append(result['auc'])
        else:
            assert len(predictions) == len(result['per_distance'])
            for pred, row in zip(predictions, result['per_distance']):
                if row['auc'] is not None:
                    assert abs(roc_auc_score(pred['labels'], pred['probabilities']) - row['auc']) < 1e-12
                    grouped[('distance', task['domain'], row['distance_bp'])].append(row['auc'])
        completed.append({'index': index, **task, 'result': result, 'seconds': complete['seconds']})
    rows = [{'mode': mode, 'domain': domain, 'x': x, 'mean_auc': float(np.mean(values)),
             'std_across_tags': float(np.std(values, ddof=1)) if len(values) > 1 else None,
             'tags_completed': len(values), 'tags_expected': len(protocol['domains'][domain])}
            for (mode, domain, x), values in sorted(grouped.items())]
    write(OUT / 'summary.json', {'completed_folds': len(completed), 'expected_folds': len(protocol['tasks']),
                               'aggregates': rows, 'folds': completed})
    lines = ['# Original benchmark CNN procedure reproduction', '',
             f'Completed {len(completed)} of {len(protocol["tasks"])} registered folds.', '',
             'Unmodified upstream training/scoring functions, original full sequences, domain-specific leave-one-tag-out evaluation. '
             'Launcher defaults are reproduced; exact published commands/RNG states are unavailable. '
             'Distance coordinates are from sequence start, not automatically from prompt end. '
             'Standard deviations describe variation across tags, not confidence intervals.', '',
             '| Experiment | Domain | Chunks / start bp | Mean AUROC | Tags completed / expected |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["mode"]} | {row["domain"]} | {row["x"]} | {row["mean_auc"]:.4f} | '
                     f'{row["tags_completed"]}/{row["tags_expected"]} |')
    lines += ['', '## Differences from our frozen project CNN', '',
              '- Original runner: 1024-base chunks, base width 96, dilations 1/2/2/4; project: 512 bases, width 64, dilations 1/2/4/8.',
              '- Original: Adam, learning rate 0.0003, 10 epochs, batch 8 sequences with four chunks each; project: AdamW, learning rate 0.001, different sampling and stopping.',
              '- Original: separate detector per domain and heldout tag; project: one mixture-trained detector with different source splits.',
              '- Original: average up to 32 chunks distributed over long FASTA sequences; project distance experiment: four local chunks from 2048 new bases.',
              '- Original: concatenates FASTA records, samples from the whole sequence including conditioning prefixes, and independently splits natural/generated items between fit and calibration. These original limitations are retained and audited.',
              '- Controlled per-fold Python/NumPy/Torch seeds and PYTHONHASHSEED=0 added because upstream leaves global initialization unspecified. Different software/hardware can still change results.',
              '- Frozen project CNN and previous reports remain unchanged. Differences here alone do not isolate a causal explanation; matched follow-up comparisons are required.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ('chunks', 'distance')):
        for domain in protocol['domains']:
            selected = sorted((r for r in rows if r['mode'] == mode and r['domain'] == domain), key=lambda r: r['x'])
            if selected:
                ax.plot([r['x'] for r in selected], [r['mean_auc'] for r in selected], 'o-', label=domain)
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Detection AUROC')
        ax.set_xlabel('Chunks averaged' if mode == 'chunks' else 'Start position in sequence (bp)')
        ax.set_title('Original benchmark: ' + mode)
        if ax.lines and len(ax.lines) > 1:
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'reproduction.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
