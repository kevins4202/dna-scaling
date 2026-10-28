import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from prepare import ROOT, OUT, DATA, SOURCE, digest, write


def main():
    torch.set_num_threads(4)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    complete = json.loads((OUT / 'complete.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    assert complete['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert complete['summary_sha256'] == digest(OUT / 'summary.json')
    assert summary['completed_folds'] == summary['expected_folds'] == complete['folds'] == 210
    for name, sha in protocol['inputs_sha256'].items():
        assert digest(ROOT / name) == sha, name
    for name, sha in protocol['frozen_cnn_sha256'].items():
        assert digest(ROOT / name) == sha, name
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('verification_original', SOURCE / 'deep_detector_dilated_resnet1d.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    groups = defaultdict(list)
    fold_checks = []
    for i, task in enumerate(protocol['tasks']):
        folder = OUT / 'folds' / f'{i:03d}'
        meta = json.loads((folder / 'complete.json').read_text())
        assert meta['task'] == task
        assert meta['protocol_sha256'] == digest(OUT / 'protocol.json')
        for name, sha in meta['files_sha256'].items():
            assert digest(folder / name) == sha, (i, name)
        model = mod.DilatedResNet1D(base_ch=96, dilations=(1, 2, 2, 4))
        state = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        assert sum(p.numel() for p in model.parameters()) == 2757313
        assert all(torch.isfinite(v).all() for v in state.values())
        splits = json.loads((folder / 'splits.json').read_text())
        assert {r['tag'] for r in splits['test']} == {task['holdout']}
        assert all(r['tag'] != task['holdout'] for key in ('fit', 'calibration') for r in splits[key])
        result = json.loads((folder / 'result.json').read_text())
        predictions = json.loads((folder / 'predictions.json').read_text())
        if task['mode'] == 'chunks':
            assert len(predictions) == 1
            a = float(roc_auc_score(predictions[0]['labels'], predictions[0]['probabilities']))
            assert a == result['auc']
            assert len(predictions[0]['labels']) == result['n_test'] == len(splits['test'])
            groups[('chunks', task['domain'], task['n_eval_chunks'])].append(a)
        else:
            assert len(predictions) == len(result['per_distance'])
            for pred, row in zip(predictions, result['per_distance']):
                assert len(pred['labels']) == row['n_used']
                if row['auc'] is not None:
                    a = float(roc_auc_score(pred['labels'], pred['probabilities']))
                    assert a == row['auc']
                    groups[('distance', task['domain'], row['distance_bp'])].append(a)
        for pred in predictions:
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in pred['probabilities'])
        fold_checks.append({'fold': i, 'strict_reload': True, 'prediction_metrics_match': True,
                            'fit_calibration_shared_pairs': splits['fit_calibration_shared_pair_count']})
    for row in summary['aggregates']:
        values = groups[(row['mode'], row['domain'], row['x'])]
        assert row['mean_auc'] == float(np.mean(values))
        assert row['tags_completed'] == len(values)
    prefixes = defaultdict(list)
    inventory = json.loads((OUT / 'dataset-inventory.json').read_text())
    tag_domain = {tag: domain for domain, tags in protocol['domains'].items() for tag in tags}
    for row in inventory:
        a = mod.read_fasta_sequence(DATA / row['orig'])
        b = mod.read_fasta_sequence(DATA / row['syn'])
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        prefixes[tag_domain[row['tag']]].append(n)
    prefix_summary = {domain: {'pairs': len(values), 'common_prefix_quantiles': np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
                              'pairs_identical_first_3072_bases': sum(x >= 3072 for x in values)}
                      for domain, values in prefixes.items()}
    write(OUT / 'prefix-audit.json', prefix_summary)
    evidence = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                if p.is_file() and p.name not in ('verification-final.json', 'evidence-sha256.json')}
    write(OUT / 'evidence-sha256.json', evidence)
    write(OUT / 'verification-final.json', {'status': 'independently_verified_complete', 'folds': 210,
          'strict_checkpoint_reloads': 210, 'all_saved_prediction_aurocs_reproduced': True,
          'all_registered_input_hashes_match': True, 'frozen_project_cnn_unchanged': True,
          'all_heldout_tags_excluded_from_fit_and_calibration': True,
          'protocol_sha256': digest(OUT / 'protocol.json'), 'verifier_sha256': digest(__import__('pathlib').Path(__file__)),
          'evidence_manifest_sha256': digest(OUT / 'evidence-sha256.json'),
          'prefix_audit': prefix_summary, 'fold_checks': fold_checks})
    print(json.dumps({'verified_folds': 210, 'evidence_files': len(evidence), 'prefix_audit': prefix_summary}))


if __name__ == '__main__':
    main()
