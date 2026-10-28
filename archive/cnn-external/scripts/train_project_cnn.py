import collections
import hashlib
import importlib.util
import json
import pathlib
import random
import shutil
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cnn-v1'
OUT = ROOT / 'reports/cnn-v1-training'
SOURCE = ROOT / 'vendor/synthetic_genomes_benchmark/scripts/classifier/deep_detector_dilated_resnet1d.py'
SEED = 6211


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture(path=SOURCE):
    spec = importlib.util.spec_from_file_location('project_cnn_architecture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DilatedResNet1D()


def encode(texts, device):
    if any(len(t) != 512 or not set(t) <= set('ACGT') for t in texts):
        raise ValueError('CNN inputs must be exactly 512 ACGT bases')
    lookup = np.full(256, -1, dtype=np.int64)
    lookup[list(b'ACGT')] = np.arange(4)
    indices = lookup[np.frombuffer(''.join(texts).encode(), dtype=np.uint8)].reshape(-1, 512)
    return torch.tensor(indices, device=device, dtype=torch.int64)


def onehot(indices):
    return torch.nn.functional.one_hot(indices, 4).permute(0, 2, 1).float()


def score(model, indices):
    model.eval()
    result = []
    with torch.inference_mode():
        for batch in indices.split(64):
            x = onehot(batch)
            values = (model(x) + model(x.flip((1, 2)))) / 2
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite classifier logits')
            result.extend(values.cpu().tolist())
    return np.array(result)


def paired_metrics(rows, real, fake):
    grouped = collections.defaultdict(lambda: [[], []])
    for row, a, b in zip(rows, real, fake):
        grouped[(row['tag'], row['pair_id'])][0].append(float(a))
        grouped[(row['tag'], row['pair_id'])][1].append(float(b))
    tags = collections.defaultdict(list)
    for (tag, pair), (a, b) in grouped.items():
        tags[tag].append((pair, float(np.mean(a)), float(np.mean(b))))
    def auc(values):
        a, b = zip(*[(r[1], r[2]) for r in values])
        return float(roc_auc_score([0]*len(a)+[1]*len(b), list(a)+list(b)))
    per_tag = {tag: auc(values) for tag, values in tags.items()}
    return {'macro_auc': float(np.mean(list(per_tag.values()))), 'per_tag_auc': per_tag,
            'pooled_pair_auc': auc([r for values in tags.values() for r in values]),
            'pairs': len(grouped)}, tags


def interval(tags):
    rng = np.random.default_rng(6212)
    names = sorted(tags)
    values = []
    for _ in range(1000):
        aucs = []
        for name in rng.choice(names, len(names), replace=True):
            records = tags[name]
            selected = [records[i] for i in rng.integers(0, len(records), len(records))]
            real, fake = [r[1] for r in selected], [r[2] for r in selected]
            aucs.append(roc_auc_score([0]*len(real)+[1]*len(fake), real+fake))
        values.append(np.mean(aucs))
    return np.quantile(values, [.025, .975]).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    release = ROOT / 'artifacts/cnn-v1'
    if release.exists() or (OUT / 'selected.pt').exists():
        raise RuntimeError('Training or release already exists; refusing an implicit retrain')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    if __import__('os').environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise RuntimeError('This training run is authorized only on physical GPU 7')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((DATA/'manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        assert digest(DATA/name) == expected
    folds = {f: [json.loads(line) for line in (DATA/f'{f}.jsonl').open()] for f in ('train', 'calibration', 'test')}
    tags = {f: {r['tag'] for r in rows} for f, rows in folds.items()}
    assert not (tags['train'] & tags['test'] or tags['train'] & tags['calibration'] or tags['test'] & tags['calibration'])
    for fold in folds:
        assert all(r['offset'] >= r['excluded_prefix'] >= 4096 for r in folds[fold])
    tensors = {f: encode([r[k] for r in rows for k in ('natural', 'generated')], 'cuda') for f, rows in folds.items() if f != 'test'}
    model = architecture().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    counts = collections.Counter(r['tag'] for r in folds['train'])
    weights = torch.tensor([1/counts[r['tag']] for r in folds['train']], device='cuda')
    history, best, stale = [], -1., 0
    protocol = {'version': 'cnn-v1', 'seed': SEED, 'parameters': sum(p.numel() for p in model.parameters()),
                'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .0001,
                'maximum_epochs': 40, 'patience': 8, 'pairs_per_epoch': min(len(folds['train']), 4096),
                'criterion': 'binary cross entropy, balanced matched pairs; equal expected taxonomic-tag weight',
                'selection': 'calibration macro AUROC after averaging logits within each source pair',
                'acceptance': 'held-out macro AUROC CI lower bound >0.5; natural-null CI contains 0.5; strict finite reload',
                'test_usage': 'one post-selection evaluation; no automatic retraining on failure',
                'source_sha256': digest(SOURCE), 'data_manifest_sha256': digest(DATA/'manifest.json')}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    start = time.time()
    for epoch in range(1, 41):
        model.train()
        chosen = torch.multinomial(weights, protocol['pairs_per_epoch'], replacement=True)
        total = 0.
        for pairs in chosen.split(32):
            indices = torch.stack((2*pairs, 2*pairs+1), dim=1).reshape(-1)
            x = onehot(tensors['train'][indices])
            reverse = torch.rand(len(x), device='cuda') < .5
            x[reverse] = x[reverse].flip((1, 2))
            y = (indices % 2).float()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x)
        cal = score(model, tensors['calibration'])
        metrics, _ = paired_metrics(folds['calibration'], cal[::2], cal[1::2])
        current = metrics['macro_auc']
        history.append({'epoch': epoch, 'loss': total/(2*len(chosen)), 'calibration': metrics,
                        'elapsed_seconds': time.time()-start})
        (OUT/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print('EPOCH', epoch, 'loss', history[-1]['loss'], 'calibration_macro_auc', current, 'seconds', time.time()-start, flush=True)
        if current > best + 1e-5:
            best, stale = current, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'protocol': protocol,
                        'calibration_macro_auc': best}, OUT/'selected.pt')
        else:
            stale += 1
        if stale >= 8:
            break
    selected = torch.load(OUT/'selected.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(selected['model'], strict=True)
    test_rows = folds['test']
    test = score(model, encode([r[k] for r in test_rows for k in ('natural', 'generated')], 'cuda'))
    test_metrics, test_groups = paired_metrics(test_rows, test[::2], test[1::2])
    test_metrics['hierarchical_bootstrap_ci95'] = interval(test_groups)
    control = score(model, encode([r['null_natural'] for r in test_rows], 'cuda'))
    null_metrics, null_groups = paired_metrics(test_rows, test[::2], control)
    null_metrics['hierarchical_bootstrap_ci95'] = interval(null_groups)
    predictions = [{k: r[k] for k in ('id', 'tag', 'pair_id', 'group_id', 'offset')} |
                   {'natural_logit': float(a), 'generated_logit': float(b), 'null_logit': float(c)}
                   for r, a, b, c in zip(test_rows, test[::2], test[1::2], control)]
    (OUT/'test-predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
    transfer = {}
    prompts_path = ROOT/'data/cnn-pilot/prompts.jsonl'
    if prompts_path.exists():
        prompts = [json.loads(line) for line in prompts_path.open()]
        natural = score(model, encode([r['natural'] for r in prompts], 'cuda'))
        for size in ('5m', '10m'):
            generated = {r['id']: r for r in map(json.loads, (prompts_path.parent/f'generated-{size}.jsonl').open())}
            fake = score(model, encode([generated[r['id']]['generated'] for r in prompts], 'cuda'))
            records = [dict(r, tag=r['category'], pair_id=str(r['id'])) for r in prompts]
            metrics, groups = paired_metrics(records, natural, fake)
            metrics['hierarchical_bootstrap_ci95'] = interval(groups)
            metrics['purpose'] = 'Transfer diagnostic on previously examined pilot prompts, unused by this CNN training or selection; not a new blind experiment'
            transfer[size] = metrics
    null_ci = null_metrics['hierarchical_bootstrap_ci95']
    accepted = test_metrics['hierarchical_bootstrap_ci95'][0] > .5 and null_ci[0] <= .5 <= null_ci[1]
    report = {'status': 'verified' if accepted else 'validation_failed_no_release', 'protocol': protocol,
              'best_epoch': selected['epoch'], 'external_test': test_metrics,
              'natural_null': null_metrics, 'small_model_transfer': transfer,
              'elapsed_seconds': time.time()-start, 'limitations': manifest['limitations'] +
              ['Sigmoid scores are not calibrated probabilities of biological function.',
               'A fixed external detector can miss defects in future generators; no claim of universal detection.',
               'Pilot transfer examples were previously examined, so transfer scores are diagnostic.']}
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if not accepted:
        raise RuntimeError('Prespecified validation failed; saved results without releasing or retraining')
    release.mkdir(parents=True)
    shutil.copy2(OUT/'selected.pt', release/'model.pt')
    shutil.copy2(SOURCE, release/'architecture.py')
    shutil.copy2(DATA/'manifest.json', release/'data-manifest.json')
    shutil.copy2(OUT/'validation.json', release/'validation.json')
    shutil.copy2(OUT/'protocol.json', release/'protocol.json')
    shutil.copy2(ROOT/'vendor/synthetic_genomes_benchmark/LICENSE', release/'LICENSE.architecture')
    shutil.copy2(ROOT/'scripts/score_project_cnn.py', release/'score.py')
    # Reload from the exported files, rather than the training module.
    frozen = architecture(release/'architecture.py').cuda().eval()
    frozen.load_state_dict(torch.load(release/'model.pt', map_location='cuda', weights_only=False)['model'], strict=True)
    probe = encode([r['natural'] for r in test_rows[:4]], 'cuda')
    np.testing.assert_allclose(score(frozen, probe), score(model, probe), atol=1e-6, rtol=1e-6)
    files = {p.name: digest(p) for p in release.iterdir() if p.is_file()}
    (release/'manifest.json').write_text(json.dumps({'version': 'cnn-v1', 'frozen': True,
        'files_sha256': files, 'input': '512 ACGT bases per chunk, prompt excluded',
        'scoring': 'mean forward/reverse-complement logits per chunk; sequence logit is mean chunk logits; sigmoid applied after averaging',
        'policy': 'No automatic retraining or weight updates. Any replacement requires explicit approval and a new version.'}, indent=2)+'\n')
    print('FROZEN_RELEASE', release, 'sha256', digest(release/'model.pt'), flush=True)


if __name__ == '__main__':
    main()
