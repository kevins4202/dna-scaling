import collections

import numpy as np
from sklearn.metrics import roc_auc_score

from common import ROOT, OUT, DATA, MODELS, sha, read, write, guard


def main():
    design = guard()
    protocol = read(OUT / 'protocol.json')
    selected = read(DATA / 'eligible.json')
    selected_groups = {p['group_id'] for p in selected}
    audit = read(OUT / 'overlap-audit.json')
    assert not selected_groups & set(audit['excluded_source_groups'])
    query_map = read(DATA / 'query-map.json')
    for p in DATA.glob('*.paf'):
        for line in p.open():
            f = line.rstrip().split('\t')
            if int(f[3]) - int(f[2]) >= 256 and int(f[9]) / max(1, int(f[10])) >= .9:
                assert query_map[f[0]]['group_id'] not in selected_groups
    score = read(OUT / 'scores.json')
    raw = []
    for c in design['checkpoints']:
        r = read(OUT / 'detector-scores' / f'{c["index"]:03d}.json')
        assert r['checkpoint_sha256'] == sha(ROOT / c['path'])
        assert r['protocol_sha256'] == sha(OUT / 'protocol.json')
        assert r['strict_reload'] and r['eval_mode'] and r['runtime_weights_unchanged']
        assert all({k: x[k] for k in ('id', 'group_id', 'category', 'model')} == y for x, y in zip(score['records'], r['records']))
        a = np.array(r['window_logits'])
        assert a.shape == (len(selected) * 7, 2) and np.isfinite(a).all()
        raw.append(a * c['weight'])
    independently_combined = np.sum(raw, axis=0)
    assert np.allclose(independently_combined, [r['window_logits'] for r in score['records']], atol=1e-12, rtol=0)
    index = {(r['id'], r['model']): r for r in score['records']}
    assert len(index) == len(selected) * 7
    result = read(OUT / 'analysis.json')
    assert result['pairs'] == protocol['pairs'] == len(selected)
    assert result['groups'] == len(selected_groups)
    weights = protocol['fixed_category_weights']
    categories = sorted({p['category'] for p in selected})
    for model in MODELS:
        point = 0.
        for category in categories:
            ids = [str(p['id']) for p in selected if p['category'] == category]
            natural = [float(np.mean(index[(i, 'natural')]['window_logits'])) for i in ids]
            generated = [float(np.mean(index[(i, model)]['window_logits'])) for i in ids]
            expected = roc_auc_score([0] * len(ids) + [1] * len(ids), natural + generated)
            assert abs(expected - result['categories'][category]['models'][model]['auc']) < 1e-12
            point += weights[category] / sum(weights[c] for c in categories) * expected
        assert abs(point - result['models'][model]['aggregate']['auc']) < 1e-12
    evidence = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
                if p.is_file() and p.name not in ('complete.json', 'verification-final.json', 'evidence-sha256.json')}
    write(OUT / 'evidence-sha256.json', evidence)
    write(OUT / 'verification-final.json', {'status': 'verified', 'pairs': len(selected), 'source_groups': len(selected_groups),
          'models': MODELS, 'detectors': 30, 'same_ids_and_weights_for_all_models': True,
          'no_detected_registered_threshold_overlap': True, 'all_source_input_hashes_match': True,
          'independent_ensemble_and_sklearn_auroc_checks': True,
          'protocol_sha256': sha(OUT / 'protocol.json'), 'evidence_manifest_sha256': sha(OUT / 'evidence-sha256.json')})
    write(OUT / 'complete.json', {'status': 'verified_complete', 'pairs': len(selected), 'source_groups': len(selected_groups),
                                'models': MODELS, 'protocol_sha256': sha(OUT / 'protocol.json'),
                                'verification_sha256': sha(OUT / 'verification-final.json')})
    print('REEVALUATION_VERIFIED_COMPLETE', len(selected), 'pairs', flush=True)


if __name__ == '__main__':
    main()
