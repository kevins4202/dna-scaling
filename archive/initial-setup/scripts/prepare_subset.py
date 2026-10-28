import gzip
import hashlib
import json
import pathlib
import re
import tarfile
import argparse

p = argparse.ArgumentParser()
p.add_argument('archive', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for split in ('train', 'valid', 'test'):
    (a.output / (split + '.jsonl')).write_text('')
records = []
with tarfile.open(a.archive) as archive:
    for member in archive:
        if not member.isfile() or not member.name.endswith('.fasta.gz'):
            continue
        if member.offset_data + member.size > a.archive.stat().st_size:
            break
        raw = archive.extractfile(member).read()
        fasta = gzip.decompress(raw).decode('ascii')
        sequences = []
        for record in fasta.split('>')[1:]:
            lines = record.splitlines()
            sequences.extend(re.split('[^ACGT]+', ''.join(lines[1:]).upper()))
        split = 'train' if len(records) < 8 else ('valid' if len(records) == 8 else 'test')
        count = bases = 0
        with (a.output / (split + '.jsonl')).open('a') as out:
            for sequence in sequences:
                for start in range(0, len(sequence) - 4096 + 1, 4096):
                    text = sequence[start:start + 4096]
                    out.write(json.dumps({'text': text}) + '\n')
                    count += 1
                    bases += len(text)
        records.append({'source_member': member.name, 'sha256_gzip': hashlib.sha256(raw).hexdigest(), 'split': split, 'windows': count, 'bases': bases})
        if len(records) == 10:
            break
manifest = {'dataset': 'arcinstitute/opengenome2', 'source': 'https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/fasta/gtdb_v220/v220_update.tar', 'download_bytes': a.archive.stat().st_size, 'archive_prefix_sha256': hashlib.sha256(a.archive.read_bytes()).hexdigest(), 'selection': 'First 10 complete genome files in a 16 MiB archive prefix; debugging subset, not representative. First 8 train, ninth valid, tenth test. Whole-genome split does not eliminate homology.', 'window_length': 4096, 'processing': 'Uppercase; split at non-ACGT characters; nonoverlapping full windows; discard tails.', 'records': records}
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({split: sum(r['bases'] for r in records if r['split'] == split) for split in ('train', 'valid', 'test')}))
