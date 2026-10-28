import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/cnn-original-v1'
DATA = ROOT / 'data/cnn-original-v1'
SOURCE = ROOT / 'vendor/synthetic_genomes_benchmark/scripts/classifier'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def module(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        old = json.loads((OUT / 'protocol.json').read_text())
        for name, sha in old['inputs_sha256'].items():
            assert digest(ROOT / name) == sha, name
        print('Existing registered protocol and inputs verified', flush=True)
        return
    archive = ROOT / 'data/cnn-benchmark/paired-v2.tar.gz'
    assert hashlib.md5(archive.read_bytes()).hexdigest() == '353488a6529ce72f8d0f435d5d761e1d'
    raw = ROOT / 'data/cnn-benchmark/raw'
    runner = module('run_evalchunks')
    domains = {'euk': runner.TAGS_EUK, 'prok': runner.TAGS_PROK, 'vir': runner.TAGS_VIR}
    inputs = {str(archive.relative_to(ROOT)): digest(archive)}
    entries, counts, repairs = [], {}, []
    for tags in domains.values():
        for tag in tags:
            original = raw / 'data' / tag / f'pairs.{tag}.csv'
            inputs[str(original.relative_to(ROOT))] = digest(original)
            rows = list(csv.DictReader(original.open()))
            assert len({r['id'] for r in rows}) == len(rows), tag
            counts[tag] = len(rows)
            target = DATA / 'data' / tag / original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            for row in rows:
                for field in ('orig', 'syn'):
                    value = row[field]
                    p = raw / value.lstrip('/')
                    if not p.is_file():
                        found = [f for f in original.parent.iterdir()
                                 if f.name.lower() == Path(value).name.lower()]
                        assert len(found) == 1, (tag, value, found)
                        p = found[0]
                    inputs[str(p.relative_to(ROOT))] = digest(p)
                    relative = f'data/{tag}/{p.name}'
                    dest = DATA / relative
                    if not dest.exists():
                        dest.symlink_to(p.resolve())
                    assert dest.resolve() == p.resolve()
                    if relative != value:
                        repairs.append({'tag': tag, 'id': row['id'], 'field': field,
                                        'original': value, 'staged': relative})
                    row[field] = relative
                entries.append({'tag': tag, 'id': row['id'], 'orig': row['orig'], 'syn': row['syn']})
            with target.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            inputs[str(target.relative_to(ROOT))] = digest(target)
    for folder in (SOURCE, ROOT / 'scripts/reproduce_cnn'):
        for p in sorted(folder.glob('*.py')):
            inputs[str(p.relative_to(ROOT))] = digest(p)
    inputs['scripts/reproduce_cnn/queue.sh'] = digest(ROOT / 'scripts/reproduce_cnn/queue.sh')
    frozen = ROOT / 'artifacts/cnn-v1'
    integrity = {str(p.relative_to(ROOT)): digest(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    tasks = []
    for nev in (32, 1, 2, 4, 8, 16):
        for domain, tags in domains.items():
            for tag in sorted(tags):
                tasks.append({'mode': 'chunks', 'domain': domain, 'holdout': tag, 'n_eval_chunks': nev})
    for domain, tags in domains.items():
        for tag in sorted(tags):
            tasks.append({'mode': 'distance', 'domain': domain, 'holdout': tag, 'n_eval_chunks': 32})
    protocol = {
        'name': 'Original benchmark CNN procedure reproduction v1',
        'upstream_commit': '2e2ab3808f4f55db1fbcf22df39610a17f4da1dc',
        'dataset_url': 'https://zenodo.org/records/18826945',
        'domains': domains, 'pair_counts': counts, 'total_pairs': sum(counts.values()),
        'settings': {'L': 1024, 'epochs': 10, 'batch': 8, 'lr': 0.0003,
                     'weight_decay': 0.0001, 'n_train_chunks': 4, 'calib_frac': 0.1,
                     'dropout': 0.2, 'base_ch': 96, 'stem_kernel': 15, 'block_kernel': 7,
                     'dilations': [1, 2, 2, 4], 'distance_chunks': 1},
        'distance_grid': {'euk': [0, 256, 2048, 4096, 10000, 20000, 50000, 100000, 200000, 290000],
                          'prok': [0, 256, 2048, 4096, 10000, 20000, 50000, 100000, 200000, 290000],
                          'vir': [0, 256, 2048, 4096, 10000, 20000, 40000]},
        'settings_basis': 'Unmodified upstream run_all_experiments/run_evalchunks/run_distance launcher defaults. Distance grid is explicitly chosen because upstream requires it and supplies no original grid.',
        'randomness': 'PYTHONHASHSEED=0; fold seed=7+(hash(tag)%100000), as upstream; explicitly seed Python, NumPy and Torch per fold. Upstream does not initialize all global RNGs, so exact published weights are not reproducible.',
        'fidelity': ['Original full FASTA sequences including conditioning prefixes, concatenation, ambiguous-base handling and original item-level calibration split retained.',
                     'No project-generated examples or project weighting. Leave one tag out within each original domain.',
                     'Upstream original training, checkpoint selection and scoring functions execute unchanged; observation hooks save checkpoints, splits and predictions.',
                     'Original known sampling/splitting limitations are recorded, not silently corrected. This is a procedural reproduction, not bitwise replication or an independent validation.',
                     'Published exact trained weights, complete original commands and RNG states are unavailable; no claim of exact published-number replication.'],
        'tasks': tasks, 'inputs_sha256': inputs, 'frozen_cnn_sha256': integrity,
    }
    write(OUT / 'path-repairs.json', repairs)
    write(OUT / 'dataset-inventory.json', entries)
    write(OUT / 'cnn-integrity-before.json', integrity)
    write(OUT / 'protocol.json', protocol)
    print(json.dumps({'pairs': sum(counts.values()), 'tasks': len(tasks), 'tags': len(counts)}), flush=True)


if __name__ == '__main__':
    main()
