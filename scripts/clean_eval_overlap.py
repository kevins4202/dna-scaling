import collections, concurrent.futures, gzip, hashlib, json, pathlib, random, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sample_opengenome as sampler

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mixture-100m-bounded"
REPORT = ROOT / "reports/overlap"
BIN = "/home/ksong37/miniconda3/envs/cis6200/bin/minimap2"


def candidate(category, split):
    quota = sampler.quotas(1_000_000)[category]
    files = sampler.list_files(category, split)
    rng = random.Random("clean-eval/" + split + category)
    rng.shuffle(files)
    result = []
    budget = 12 * quota
    for shard in files[:2]:
        with sampler.request(
            f"https://huggingface.co/datasets/{sampler.REPO}/resolve/{sampler.REVISION}/"
            + shard["path"]
        ) as response, gzip.GzipFile(fileobj=response) as stream:
            chars = 0
            for index, line in enumerate(stream, 1):
                row = json.loads(line)
                text = row["text"]
                for offset in range(0, len(text), 4096):
                    part = text[offset : offset + 4096]
                    result.append(
                        {
                            "text": part,
                            "category": category,
                            "source_shard": shard["path"],
                            "source_row": index,
                            "source_offset": offset,
                            "source_record": row.get("record"),
                        }
                    )
                    chars += len(part)
                    if chars >= budget:
                        break
                if chars >= budget:
                    break
    return result


def fasta(rows, path):
    with path.open("w") as f:
        for i, row in enumerate(rows):
            f.write(f'>q{i}\n{row["text"].upper()}\n')


def canon(text):
    text = text.upper()
    reverse = text.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return hashlib.sha256(min(text, reverse).encode()).hexdigest()


def run():
    manifest = json.loads((DATA / "manifest.json").read_text())
    original = {
        s: [json.loads(l) for l in (DATA / f"{s}.jsonl").open()]
        for s in ("train", "valid", "test")
    }
    train_exact = {canon(r["text"]) for r in original["train"] if len(r["text"]) >= 128}
    summary = {}
    for split in ("valid", "test"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            lists = list(pool.map(lambda c: candidate(c, split), sampler.WEIGHTS))
        rows = [r for group in lists for r in group]
        path = REPORT / f"{split}-candidates.fasta"
        fasta(rows, path)
        paf = REPORT / f"{split}-candidates-vs-train.paf"
        with paf.open("w") as out:
            subprocess.run(
                [
                    BIN,
                    "-x",
                    "asm10",
                    "-c",
                    "--secondary=no",
                    "-t",
                    "4",
                    str(DATA / "train.mmi"),
                    str(path),
                ],
                stdout=out,
                check=True,
            )
        bad = set()

        def add_bad(paf):
            for line in paf.open():
                x = line.split()
                length = int(x[10])
                identity = int(x[9]) / max(1, length)
                if length >= 128 and identity >= 0.9:
                    bad.add(int(x[0][1:]))

        add_bad(paf)
        if split == "test":
            paf = REPORT / "test-candidates-vs-clean-valid.paf"
            with paf.open("w") as out:
                subprocess.run(
                    [
                        BIN,
                        "-x",
                        "asm10",
                        "-c",
                        "--secondary=no",
                        "-t",
                        "4",
                        str(DATA / "valid.fasta"),
                        str(path),
                    ],
                    stdout=out,
                    check=True,
                )
            add_bad(paf)
        pool = collections.defaultdict(list)
        seen = set(train_exact)
        if split == "test":
            seen.update(
                canon(r["text"]) for r in original["valid"] if len(r["text"]) >= 128
            )
        duplicates = 0
        for i, row in enumerate(rows):
            digest = canon(row["text"])
            if i in bad or (len(row["text"]) >= 128 and digest in seen):
                duplicates += 1
                continue
            seen.add(digest)
            pool[row["category"]].append(row)
        output = []
        for category, quota in sampler.quotas(1_000_000).items():
            candidates = pool[category]
            rng = random.Random("choose-clean/" + split + category)
            rng.shuffle(candidates)
            for row in candidates:
                if quota == 0:
                    break
                row = dict(row)
                if len(row["text"]) > quota:
                    offset = rng.randrange(len(row["text"]) - quota + 1)
                    row["text"] = row["text"][offset : offset + quota]
                    row["source_offset"] += offset
                quota -= len(row["text"])
                output.append(row)
            if quota:
                raise RuntimeError(
                    f"{split}/{category}: insufficient clean candidate characters; missing {quota}"
                )
        random.Random("shuffle-clean/" + split).shuffle(output)
        (DATA / f"{split}.before-overlap-cleanup.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in original[split])
        )
        (DATA / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in output)
        )
        fasta(output, DATA / f"{split}.fasta")
        item = manifest["splits"][split]
        item["records"] = len(output)
        item["sha256"] = hashlib.sha256(
            (DATA / f"{split}.jsonl").read_bytes()
        ).hexdigest()
        for category in sampler.WEIGHTS:
            counts = collections.Counter()
            for row in output:
                if row["category"] == category:
                    counts.update(sampler.stats(row["text"]))
            item["categories"][category]["selected_stats"] = dict(counts)
        original[split] = output
        summary[split] = {
            "candidates": len(rows),
            "excluded_similarity_or_duplicates": duplicates,
            "screen": "minimap2 asm10, alignment block >=128 and matching-base fraction >=0.90; reverse-complement exact-record dedup >=128; not exhaustive homology exclusion",
        }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (REPORT / "cleanup.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("CLEANUP_COMPLETE", summary, flush=True)


if __name__ == "__main__":
    run()
