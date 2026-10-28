import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import torch


def default_release():
    path = pathlib.Path(__file__).resolve()
    return path.parent if (path.parent/'manifest.json').is_file() else path.parents[1]/'artifacts/cnn-v1'


def load_release(release, device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((release/'manifest.json').read_text())
    if not manifest.get('frozen') or manifest['version'] != 'cnn-v1':
        raise ValueError('Expected a frozen cnn-v1 release')
    for name, expected in manifest['files_sha256'].items():
        if pathlib.Path(name).name != name or hashlib.sha256((release/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen CNN artifact changed: {name}')
    if hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest() != manifest['files_sha256']['score.py']:
        raise ValueError('Scoring implementation differs from the frozen release')
    spec = importlib.util.spec_from_file_location('frozen_cnn_v1', release/'architecture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DilatedResNet1D().to(device).eval()
    model.load_state_dict(torch.load(release/'model.pt', map_location=device, weights_only=False)['model'], strict=True)
    return model, manifest


def score_text(model, text, device):
    if not isinstance(text, str) or len(text) < 512 or len(text) % 512 or not set(text) <= set('ACGT'):
        raise ValueError('Input must contain only uppercase ACGT, with length a positive multiple of 512; prompts excluded')
    lookup = np.full(256, -1, dtype=np.int64)
    lookup[list(b'ACGT')] = np.arange(4)
    values = lookup[np.frombuffer(text.encode(), dtype=np.uint8)].reshape(-1, 512)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(values), 64):
            indices = torch.tensor(values[start:start+64], device=device)
            x = torch.nn.functional.one_hot(indices, 4).permute(0, 2, 1).float()
            output = (model(x)+model(x.flip((1, 2))))/2
            if not torch.isfinite(output).all():
                raise ValueError('Nonfinite CNN scores')
            logits.extend(output.cpu().tolist())
    logit = float(np.mean(logits))
    return logit, logits


def main():
    parser = argparse.ArgumentParser(description='Score DNA using the immutable project CNN v1. Inputs contain continuations only.')
    parser.add_argument('--input', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--release', type=pathlib.Path, default=default_release())
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, manifest = load_release(args.release, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.tmp')
    count = 0
    try:
        with args.input.open() as source, temporary.open('w') as output:
            writer = csv.DictWriter(output, fieldnames=['id', 'cnn_version', 'sequence_logit', 'synthetic_score', 'natural_score', 'chunk_logits'])
            writer.writeheader()
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get('generated', row.get('text'))
                logit, chunks = score_text(model, text, args.device)
                probability = float(1/(1+np.exp(-np.clip(logit, -40, 40))))
                writer.writerow({'id': row.get('id', number-1), 'cnn_version': manifest['version'],
                    'sequence_logit': logit, 'synthetic_score': probability, 'natural_score': 1-probability,
                    'chunk_logits': json.dumps(chunks)})
                count += 1
        if not count:
            raise ValueError('Input contains no sequences')
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    args.output.with_suffix(args.output.suffix+'.provenance.json').write_text(json.dumps({
        'version': manifest['version'], 'model_sha256': manifest['files_sha256']['model.pt'],
        'validation_status': manifest.get('validation_status','unspecified'),
        'release_manifest_sha256': hashlib.sha256((args.release/'manifest.json').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'sequences': count,
        'device': args.device, 'window_length': 512,
        'score_definition': manifest['scoring']}, indent=2)+'\n')
    print('SCORED', count, 'with frozen', manifest['version'])


if __name__ == '__main__':
    main()
