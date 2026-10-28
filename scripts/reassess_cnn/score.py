import importlib.util
import os

import numpy as np
import torch

from common import ROOT, OUT, DATA, MODELS, GENERATIONS, sha, read, rows, write, guard


def combine(raw, checkpoints):
    domains = sorted({c['domain'] for c in checkpoints})
    by_domain = {d: np.mean([raw[str(c['index'])] for c in checkpoints if c['domain'] == d], axis=0)
                 for d in domains}
    balanced = np.mean(list(by_domain.values()), axis=0)
    return balanced, by_domain


def main():
    design = guard()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    selected = read(DATA / 'eligible.json')
    gen = {m: {str(r['id']): r for r in rows(ROOT / GENERATIONS[m])} for m in MODELS}
    records, texts = [], []
    for p in selected:
        for label in ['natural', *MODELS]:
            text = p['natural'] if label == 'natural' else gen[label][str(p['id'])]['generated']
            assert len(text) == 2048 and set(text) <= set('ACGT')
            records.append({'id': str(p['id']), 'group_id': p['group_id'], 'category': p['category'], 'model': label})
            texts.extend([text[:1024], text[1024:]])
    source = ROOT / 'vendor/synthetic_genomes_benchmark/scripts/classifier/deep_detector_dilated_resnet1d.py'
    spec = importlib.util.spec_from_file_location('reproduced_cnn', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lookup = np.full(256, -1, dtype=np.int64)
    lookup[list(b'ACGT')] = np.arange(4)
    encoded = lookup[np.frombuffer(''.join(texts).encode(), dtype=np.uint8)].reshape(-1, 1024)
    assert encoded.min() >= 0
    raw = {}
    for c in design['checkpoints']:
        dest = OUT / 'detector-scores' / f'{c["index"]:03d}.json'
        if dest.exists():
            old = read(dest)
            assert old['protocol_sha256'] == sha(OUT / 'protocol.json')
            assert old['checkpoint_sha256'] == sha(ROOT / c['path'])
            assert old['records'] == records
            values = np.array(old['window_logits'])
            assert values.shape == (len(records), 2) and np.isfinite(values).all()
            raw[str(c['index'])] = values
            continue
        state = torch.load(ROOT / c['path'], map_location='cpu', weights_only=True)
        model = mod.DilatedResNet1D(base_ch=96, dilations=(1, 2, 2, 4)).cuda()
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        result = []
        with torch.inference_mode():
            for start in range(0, len(encoded), 64):
                indices = torch.from_numpy(encoded[start:start + 64]).cuda()
                x = torch.nn.functional.one_hot(indices, 4).permute(0, 2, 1).float()
                logits = model(x)
                assert torch.isfinite(logits).all()
                result.extend(logits.cpu().tolist())
        assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
        values = np.array(result).reshape(-1, 2)
        write(dest, {'protocol_sha256': sha(OUT / 'protocol.json'), 'checkpoint_sha256': sha(ROOT / c['path']),
                     'records': records, 'window_logits': values.tolist(), 'strict_reload': True,
                     'eval_mode': True, 'runtime_weights_unchanged': True})
        raw[str(c['index'])] = values
        print('SCORED_DETECTOR', c['index'], c['domain'], c['holdout'], flush=True)
        del model
    balanced, domain_scores = combine(raw, design['checkpoints'])
    assert np.isfinite(balanced).all()
    scored = []
    for i, record in enumerate(records):
        scored.append({**record, 'window_logits': balanced[i].tolist(), 'aggregate_logit': float(balanced[i].mean()),
                       'domain_window_logits': {d: a[i].tolist() for d, a in domain_scores.items()}})
    write(OUT / 'scores.json', {'protocol_sha256': sha(OUT / 'protocol.json'), 'records': scored,
                               'detectors': design['checkpoints'], 'device': torch.cuda.get_device_name(0),
                               'torch': torch.__version__, 'precision': 'FP32', 'windows': [[0, 1024], [1024, 2048]]})
    guard()


if __name__ == '__main__':
    main()
