import collections, concurrent.futures, gzip, hashlib, json, math, pathlib, random, re, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sample_opengenome as sampler

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mixture-100m-bounded"
REF = ROOT / "data/reference-independent"
REF.mkdir(exist_ok=True, parents=True)
LOOKUP = np.full(256, 4, dtype=np.uint8)
for i, c in enumerate(b"ACGT"):
    LOOKUP[c] = i


def measures(records):
    counts = np.zeros(64, dtype=np.int64)
    mono = np.zeros(5, dtype=np.int64)
    gcwindows = []
    lengths = []
    lower = repeat = total = 0
    for row in records:
        text = row["text"]
        up = text.upper()
        a = LOOKUP[np.frombuffer(up.encode("ascii"), dtype=np.uint8)]
        mono += np.bincount(a, minlength=5)
        total += len(text)
        lengths.append(len(text))
        lower += len(text) - len(
            text.translate(str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz"))
        )
        for c in "ACGT":
            repeat += sum(len(m.group()) for m in re.finditer(c + "{8,}", up))
        if len(a) >= 3:
            valid = (a[:-2] < 4) & (a[1:-1] < 4) & (a[2:] < 4)
            codes = 16 * a[:-2] + 4 * a[1:-1] + a[2:]
            counts += np.bincount(codes[valid], minlength=64)
        for start in range(0, len(a) - 1023, 1024):
            w = a[start : start + 1024]
            den = int((w < 4).sum())
            if den:
                gcwindows.append(float(((w == 1) | (w == 2)).sum() / den))
    acgt = int(mono[:4].sum())
    return {
        "chars": total,
        "gc": float((mono[1] + mono[2]) / max(1, acgt)),
        "non_acgt_fraction": float(mono[4] / max(1, total)),
        "lowercase_fraction": lower / max(1, total),
        "homopolymer_ge8_fraction": repeat / max(1, total),
        "trimer_counts": counts.tolist(),
        "length_quantiles": np.quantile(
            lengths, [0, 0.25, 0.5, 0.75, 0.95, 1]
        ).tolist(),
        "gc_window_quantiles": (
            np.quantile(gcwindows, [0, 0.1, 0.5, 0.9, 1]).tolist() if gcwindows else []
        ),
    }


def reference(category):
    path = REF / f"{category}.jsonl"
    provenance = REF / f"{category}.source.json"
    if path.exists() and provenance.exists():
        return
    manifest = json.loads((DATA / "manifest.json").read_text())
    used = manifest["splits"]["train"]["categories"][category]["shards"]
    blocked = {x["path"]: x["records_scanned"] for x in used}
    files = sampler.list_files(category, "train")
    rng = random.Random("independent-20261025/" + category)
    rng.shuffle(files)
    unused = [x for x in files if x["path"] not in blocked]
    chosen = (unused or files)[:2]
    records = []
    sources = []
    for shard in chosen:
        skip = blocked.get(shard["path"], 0)
        observed = 0
        url = (
            f"https://huggingface.co/datasets/{sampler.REPO}/resolve/{sampler.REVISION}/"
            + shard["path"]
        )
        with sampler.request(url) as response, gzip.GzipFile(
            fileobj=response
        ) as stream:
            for index, line in enumerate(stream, 1):
                if index <= skip:
                    continue
                original = json.loads(line)
                text = original["text"]
                if not text:
                    continue
                records.append(
                    {
                        "text": text,
                        "category": category,
                        "source_shard": shard["path"],
                        "source_row": index,
                        "source_record": original.get("record"),
                    }
                )
                observed += len(text)
                if observed >= 4_000_000:
                    break
        sources.append(
            {
                "path": shard["path"],
                "skipped_rows": skip,
                "scanned_characters": observed,
            }
        )
    rng.shuffle(records)
    remaining = 2_000_000
    selected = []
    for row in records:
        if not remaining:
            break
        text = row["text"]
        offset = (
            rng.randrange(len(text) - remaining + 1) if len(text) > remaining else 0
        )
        row["source_offset"] = offset
        row["text"] = text[offset : offset + remaining]
        selected.append(row)
        remaining -= len(row["text"])
    if remaining:
        raise ValueError(category + " insufficient reference")
    path.write_text("".join(json.dumps(x) + "\n" for x in selected))
    provenance.write_text(json.dumps(sources, indent=2) + "\n")
    print("REFERENCE_READY", category, flush=True)


def load(path):
    return [json.loads(x) for x in path.open()]


def run():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(reference, sampler.WEIGHTS))
    train = load(DATA / "train.jsonl")
    groups = collections.defaultdict(list)
    for row in train:
        groups[row["category"]].append(row)
    comparison = {}
    for category in sampler.WEIGHTS:
        sample = measures(groups[category])
        ref = measures(load(REF / f"{category}.jsonl"))
        p = np.array(sample["trimer_counts"], dtype=float)
        q = np.array(ref["trimer_counts"], dtype=float)
        p /= max(1, p.sum())
        q /= max(1, q.sum())
        differences = {
            "gc_absolute": abs(sample["gc"] - ref["gc"]),
            "trimer_total_variation": float(abs(p - q).sum() / 2),
            "non_acgt_absolute": abs(
                sample["non_acgt_fraction"] - ref["non_acgt_fraction"]
            ),
        }
        comparison[category] = {
            "subset": sample,
            "independent_reference": ref,
            "differences": differences,
            "large_difference": differences["gc_absolute"] > 0.05
            or differences["trimer_total_variation"] > 0.10
            or differences["non_acgt_absolute"] > 0.02,
        }
        print(
            category,
            differences,
            "LARGE",
            comparison[category]["large_difference"],
            flush=True,
        )
    out = ROOT / "reports/distribution-audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "reference_seed": "independent-20261025",
                "reference_characters_per_category": {
                    k: v["independent_reference"]["chars"]
                    for k, v in comparison.items()
                },
                "thresholds": {
                    "gc_absolute": 0.05,
                    "trimer_total_variation": 0.10,
                    "non_acgt_absolute": 0.02,
                },
                "note": "Reference uses disjoint shards where possible, otherwise disjoint source coordinates; GTDB was broadened to 16 disjoint shards and eukaryotic reference to all 40 shards. It is independent of selected records but is still bounded prefix sampling, not a full-population reference. Thresholds are screening rules, not statistical significance tests.",
                "categories": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    # Export FASTA for alignment-based cross-split similarity checks.
    for split in ("train", "valid", "test"):
        with (DATA / f"{split}.fasta").open("w") as f:
            for index, row in enumerate(load(DATA / f"{split}.jsonl")):
                text = row["text"].upper()
                f.write(f">{split}_{index}\n{text}\n")
    print("AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    run()
