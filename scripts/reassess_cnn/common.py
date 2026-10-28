import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/cnn-reassessment-v1'
DATA = ROOT / 'data/cnn-reassessment-v1'
MODELS = ['5m', '10m', '25m', '50m', '100m', 'released_7b']
GENERATIONS = {m: f'data/distance-2048-v1/generated-{m}.jsonl' for m in MODELS[:-1]}
GENERATIONS['released_7b'] = 'data/evo7b-baseline-v1/generated-evo2_7b_base.jsonl'
OLD_SCORES = {'5m': 'reports/distance-2048-v1/5m-scores.json',
              '10m': 'reports/distance-2048-v1/10m-scores.json',
              '25m': 'reports/distance-2048-25m-v1/25m-scores.json',
              '50m': 'reports/distance-2048-50m-v1/50m-scores.json',
              '100m': 'reports/distance-2048-100m-v1/100m-scores.json',
              'released_7b': 'reports/evo7b-baseline/released-7b-scores.json'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def guard():
    design = read(OUT / 'design.json')
    for name, expected in design['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    if (OUT / 'protocol.json').exists():
        p = read(OUT / 'protocol.json')
        assert p['design_sha256'] == sha(OUT / 'design.json')
        for name, expected in p['screening_sha256'].items():
            assert sha(ROOT / name) == expected, name
    return design
