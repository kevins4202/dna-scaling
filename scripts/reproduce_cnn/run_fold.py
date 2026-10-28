import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prepare import ROOT, OUT, DATA, SOURCE, digest, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int)
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    protocol = json.loads((OUT / 'protocol.json').read_text())
    task = protocol['tasks'][args.task]
    dest = OUT / 'folds' / f'{args.task:03d}'
    if (dest / 'complete.json').exists():
        result = json.loads((dest / 'complete.json').read_text())
        assert result['task'] == task
        for name, sha in result['files_sha256'].items():
            assert digest(dest / name) == sha
        print('Verified completed fold', args.task, flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    source = SOURCE / ('deep_detector_dilated_resnet1d.py' if task['mode'] == 'chunks'
                       else 'deep_detector_distance_eval.py')
    assert digest(source) == protocol['inputs_sha256'][str(source.relative_to(ROOT))]
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('original_detector', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tags = sorted(protocol['domains'][task['domain']])
    holdout = task['holdout']
    items = {tag: m.load_items_for_tag(DATA, tag) for tag in tags}
    train = [it for tag in tags if tag != holdout for it in items[tag]]
    test = items[holdout]
    assert not {x['tag'] for x in train} & {x['tag'] for x in test}
    seed = 7 + (hash(holdout) % 100000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    fit, calib = m.stratified_split(train, frac_calib=0.1, seed=seed + 12345)
    identity = lambda x: {'tag': x['tag'], 'id': x['id'], 'which': x['which'], 'label': x['label']}
    fit_ids = {(x['tag'], x['id']) for x in fit}
    cal_ids = {(x['tag'], x['id']) for x in calib}
    write(dest / 'splits.json', {'fit': list(map(identity, fit)), 'calibration': list(map(identity, calib)),
                               'test': list(map(identity, test)),
                               'fit_calibration_shared_pair_count': len(fit_ids & cal_ids),
                               'note': 'Original item-level split retained, including pairs split across fit and calibration.'})
    models, captures = [], []
    original_class = m.DilatedResNet1D

    class ObservedModel(original_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            models.append(self)

    m.DilatedResNet1D = ObservedModel
    score_name = 'collect_scores' if task['mode'] == 'chunks' else 'collect_probs'
    original_collect = getattr(m, score_name)

    def collect(model, loader, device):
        y, p = original_collect(model, loader, device)
        assert all(math.isfinite(float(v)) for v in p)
        captures.append({'labels': y, 'probabilities': p})
        return y, p

    setattr(m, score_name, collect)
    settings = dict(protocol['settings'])
    settings.pop('distance_chunks')
    settings['dilations'] = tuple(settings['dilations'])
    common = dict(train_items=train, test_items=test, **settings,
                  device=torch.device('cuda'), seed=seed)
    start = time.time()
    if task['mode'] == 'chunks':
        result = m.train_one_length(**common, n_eval_chunks=task['n_eval_chunks'])
        final_predictions = captures[-1:]
    else:
        result = m.train_and_distance_eval_for_holdout(**common, n_calib_chunks=32,
                    distances=protocol['distance_grid'][task['domain']], distance_chunks=1)
        final_predictions = captures[11:]
    assert len(models) == 1
    state = {k: v.detach().cpu() for k, v in models[0].state_dict().items()}
    assert all(torch.isfinite(v).all().item() for v in state.values())
    tmp = dest / 'selected.pt.tmp'
    torch.save(state, tmp)
    tmp.replace(dest / 'selected.pt')
    reload_model = original_class(dropout=0.2, base_ch=96, stem_kernel=15,
                                 block_kernel=7, dilations=(1, 2, 2, 4))
    reload_model.load_state_dict(torch.load(dest / 'selected.pt', weights_only=True), strict=True)
    clean = lambda obj: ({k: clean(v) for k, v in obj.items()} if isinstance(obj, dict)
                         else [clean(v) for v in obj] if isinstance(obj, (tuple, list))
                         else None if isinstance(obj, float) and not math.isfinite(obj) else obj)
    write(dest / 'predictions.json', clean(final_predictions))
    write(dest / 'result.json', clean(result))
    write(dest / 'complete.json', {'task': task, 'seed': seed, 'seconds': time.time() - start,
          'parameters': sum(p.numel() for p in reload_model.parameters()),
          'strict_reload': True, 'finite_weights_and_scores': True,
          'protocol_sha256': digest(OUT / 'protocol.json'),
          'files_sha256': {n: digest(dest / n) for n in ('selected.pt', 'predictions.json', 'result.json', 'splits.json')}})
    print('Completed', args.task, task, result, flush=True)


if __name__ == '__main__':
    main()
