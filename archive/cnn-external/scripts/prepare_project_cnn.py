import collections
import csv
import gzip
import hashlib
import json
import pathlib
import random
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/cnn-benchmark/raw'
OUT = ROOT / 'data/cnn-v1'
SEED = 6210
LENGTH = 512


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(text):
    return min(text, text.translate(str.maketrans('ACGT', 'TGCA'))[::-1])


def fasta(path):
    opener = gzip.open if path.suffix == '.gz' else open
    records = []
    with opener(path, 'rt') as handle:
        name, parts = None, []
        for line in handle:
            if line.startswith('>'):
                if name is not None:
                    records.append((name, ''.join(parts).upper()))
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
        if name is not None:
            records.append((name, ''.join(parts).upper()))
    return records


def resolve(value, tag):
    value = value.lstrip('/')
    candidate = RAW / value
    if candidate.is_file():
        return candidate
    candidate = RAW / 'data' / tag / pathlib.Path(value).name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(value)


def tag_splits(tags):
    families = collections.defaultdict(list)
    for tag in sorted(tags):
        family = ('virus' if tag.endswith('viricota') or tag == 'megaDNA_bacteriophages'
                  else 'prokaryote' if tag in ('Archaea', 'Chlamydiota', 'Mycoplasmatota', 'Pseudomonadota')
                  else 'eukaryote')
        families[family].append(tag)
    result = {}
    for family, values in families.items():
        random.Random(f'{SEED}/{family}').shuffle(values)
        n = max(1, round(len(values) * .2))
        for i, tag in enumerate(values):
            result[tag] = 'test' if i < n else 'calibration' if i < 2*n else 'train'
    return result


def write_fasta(rows, path):
    with path.open('w') as handle:
        for i, row in enumerate(rows):
            for field in ('natural', 'generated', 'null_natural'):
                handle.write(f'>{i}/{field}\n{row[field]}\n')


def remove_overlap(reference, query, label):
    ref_path, query_path = OUT / f'{label}-reference.fa', OUT / f'{label}-query.fa'
    write_fasta(reference, ref_path)
    write_fasta(query, query_path)
    executable = shutil.which('minimap2')
    if not executable:
        raise RuntimeError('minimap2 is required for cross-fold similarity screening')
    paf = OUT / f'{label}.paf'
    with paf.open('w') as output, (OUT / f'{label}.log').open('w') as log:
        subprocess.run([executable, '-x', 'asm10', '-t', '4', '-N', '50', str(ref_path), str(query_path)],
                       stdout=output, stderr=log, check=True)
    bad = set()
    for line in paf.open():
        fields = line.split('\t')
        if int(fields[10]) >= 256 and int(fields[9]) / int(fields[10]) >= .9:
            bad.add(int(fields[0].split('/')[0]))
    return [row for i, row in enumerate(query) if i not in bad], len(bad)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Prepared dataset already exists; refusing to change its split')
    archive = ROOT / 'data/cnn-benchmark/paired-v2.tar.gz'
    assert hashlib.md5(archive.read_bytes()).hexdigest() == '353488a6529ce72f8d0f435d5d761e1d'
    manifests = sorted((RAW / 'data').glob('*/pairs.*.csv'))
    splits = tag_splits([path.parent.name for path in manifests])
    seed_lengths = {}
    for path in manifests:
        maximum = 4096
        for meta in path.parent.rglob('*.json'):
            try:
                record = json.loads(meta.read_text())
                if isinstance(record, dict):
                    for key in ('seed_len', 'seed_length', 'prompt_len', 'prompt_length'):
                        value = record.get(key)
                        if isinstance(value, int):
                            maximum = max(maximum, value)
            except (ValueError, UnicodeError):
                continue
        seed_lengths[path.parent.name] = maximum
    rows, issues = [], []
    for manifest in manifests:
        tag = manifest.parent.name
        for item in csv.DictReader(manifest.open()):
            try:
                orig_path, syn_path = resolve(item['orig'], tag), resolve(item['syn'], tag)
                natural, generated = fasta(orig_path), fasta(syn_path)
                if len(natural) != 1 or len(generated) != 1:
                    raise ValueError('multiple contigs require explicit contig pairing; excluded')
                real, fake = natural[0][1], generated[0][1]
                prefix = 0
                for a, b in zip(real, fake):
                    if a != b:
                        break
                    prefix += 1
                start = ((max(seed_lengths[tag], prefix) + LENGTH - 1) // LENGTH) * LENGTH
                stop = min(len(real), len(fake)) - 2*LENGTH
                positions = list(range(start, stop + 1, 2*LENGTH))
                random.Random(f'{SEED}/{tag}/{item["id"]}').shuffle(positions)
                count = 0
                for pos in positions:
                    a, b, control = real[pos:pos+LENGTH], fake[pos:pos+LENGTH], real[pos+LENGTH:pos+2*LENGTH]
                    if not set(a+b+control) <= set('ACGT'):
                        continue
                    rows.append({'id': f'{tag}/{item["id"]}/{pos}', 'tag': tag, 'fold': splits[tag],
                                 'group_id': f'{tag}/{orig_path.name}', 'pair_id': f'{tag}/{item["id"]}',
                                 'offset': pos, 'excluded_prefix': start, 'natural': a, 'generated': b,
                                 'null_natural': control, 'natural_file': str(orig_path.relative_to(RAW)),
                                 'generated_file': str(syn_path.relative_to(RAW))})
                    count += 1
                    if count == 12:
                        break
                if not count:
                    issues.append({'tag': tag, 'pair': item['id'], 'reason': 'no eligible post-prompt ACGT windows'})
            except (OSError, ValueError) as exc:
                issues.append({'tag': tag, 'pair': item['id'], 'reason': str(exc)})
    folds = {fold: [r for r in rows if r['fold'] == fold] for fold in ('train', 'calibration', 'test')}
    seen, exact_removed = set(), {}
    for fold, records in folds.items():
        kept = []
        for row in records:
            keys = {canonical(row[k]) for k in ('natural', 'generated', 'null_natural')}
            if keys & seen:
                continue
            kept.append(row)
        exact_removed[fold] = len(records)-len(kept)
        folds[fold] = kept
        seen.update(canonical(row[k]) for row in kept for k in ('natural', 'generated', 'null_natural'))
    folds['calibration'], cal_removed = remove_overlap(folds['train'], folds['calibration'], 'train-calibration')
    folds['test'], test_removed = remove_overlap(folds['train']+folds['calibration'], folds['test'], 'development-test')
    for fold, records in folds.items():
        assert len(records) >= 100, (fold, len(records))
        (OUT / f'{fold}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report = {'version': 'cnn-v1', 'seed': SEED, 'length': LENGTH, 'archive_sha256': digest(archive),
              'dataset_url': 'https://zenodo.org/records/18826945', 'tag_folds': splits,
              'counts': {f: len(r) for f, r in folds.items()},
              'groups': {f: len({x['group_id'] for x in r}) for f, r in folds.items()},
              'tags': {f: dict(collections.Counter(x['tag'] for x in r)) for f, r in folds.items()},
              'excluded_seed_bases_by_tag': seed_lengths, 'exact_overlap_removed': exact_removed,
              'similarity_removed': {'calibration': cal_removed, 'test': test_removed}, 'excluded_pairs': issues,
              'file_sha256': {f'{f}.jsonl': digest(OUT/f'{f}.jsonl') for f in folds},
              'limitations': ['External Evo2/MegaDNA benchmark distribution differs from OpenGenome2 and our small generators.',
                              'Whole taxonomic tags are disjoint; this is not an exhaustive phylogenetic exclusion.',
                              'Minimap2 screen removes reported >=90% identity alignments spanning >=256 bases; it is not proof of no homology.',
                              'Only 512-base ACGT windows after conditioning prefixes are represented.']}
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('counts', 'groups', 'tags', 'similarity_removed')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
