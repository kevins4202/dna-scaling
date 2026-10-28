"""Stratified, seeded OpenGenome2 sampling. Python standard library only."""

import argparse
import collections
import gzip
import hashlib
import heapq
import json
import math
import pathlib
import random
import re
import time
import urllib.parse
import urllib.request

REPO = "arcinstitute/opengenome2"
REVISION = "84d2a7e690c8bb395d1f4868822dfa031ea561c6"
WEIGHTS = {
    "gtdb_v220_imgpr": 18,
    "metagenomes": 24,
    "imgvr_untagged": 3,
    "mrna_splice_promoter": 9,
    "mrna": 9,
    "eukaryotic_genic_windows": 35,
    "organelle": 0.5,
    "ncrna": 2,
    "promoters": 0.02,
}
SOURCE = (
    "https://huggingface.co/datasets/arcinstitute/opengenome2#training-data-composition"
)


def request(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "cis6200-opengenome-sampler/1"}
                ),
                timeout=90,
            )
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def list_files(category, split):
    path = "json/pretraining_or_both_phases/" + category
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{path}?limit=100"
    files = []
    while url:
        with request(url) as response:
            files.extend(
                x
                for x in json.load(response)
                if x["type"] == "file"
                and f"_{split}_" in x["path"]
                and x["path"].endswith(".jsonl.gz")
            )
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else None
    if not files:
        raise ValueError(f"No shards for {category}/{split}")
    return sorted(files, key=lambda x: x["path"])


def quotas(total):
    exact = {k: total * w / sum(WEIGHTS.values()) for k, w in WEIGHTS.items()}
    result = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - result[k], reverse=True)[
        : total - sum(result.values())
    ]:
        result[k] += 1
    return result


def stats(text):
    upper = text.upper()
    acgt = sum(upper.count(c) for c in "ACGT")
    return collections.Counter(
        chars=len(text),
        acgt=acgt,
        gc=upper.count("G") + upper.count("C"),
        ambiguous=upper.count("N"),
        lowercase=sum(c.islower() for c in text),
    )


def sample_category(category, split, budget, args):
    rng = random.Random(f"{args.seed}/{category}/{split}")
    files = list_files(category, split)
    rng.shuffle(files)
    if args.shards_per_category:
        files = files[: args.shards_per_category]
    if args.mode == "smoke":
        files = files[:1]
    elif args.mode == "bounded":
        files = files[:4]
    heap, kept_chars, observed, provenance = [], 0, collections.Counter(), []
    serial = 0
    for shard in files:
        url = (
            f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
            + urllib.parse.quote(shard["path"])
        )
        read_chars = records = 0
        digest = hashlib.sha256()
        with request(url) as response:
            with gzip.GzipFile(fileobj=response) as compressed:
                for line in compressed:
                    digest.update(line)
                    row = json.loads(line)
                    text = row["text"]
                    if not isinstance(text, str) or not text.isascii():
                        raise ValueError(
                            f'Non-ASCII/non-string text in {shard["path"]}'
                        )
                    if not text:
                        continue
                    read_chars += len(text)
                    records += 1
                    observed.update(stats(text))
                    serial += 1
                    priority = rng.random()
                    item = (-priority, serial, text, shard["path"], records)
                    heapq.heappush(heap, item)
                    kept_chars += len(text)
                    while heap and kept_chars - len(heap[0][2]) >= budget:
                        kept_chars -= len(heapq.heappop(heap)[2])
                    if args.mode in ("smoke", "bounded") and read_chars >= max(
                        args.smoke_scan_chars, math.ceil(4 * budget / len(files))
                    ):
                        break
        provenance.append(
            {
                "path": shard["path"],
                "compressed_size": shard["size"],
                "records_scanned": records,
                "chars_scanned": read_chars,
                "scanned_jsonl_sha256": digest.hexdigest(),
                "complete_scan": args.mode == "full",
            }
        )
    if kept_chars < budget:
        raise ValueError(
            f"{category}/{split}: only {kept_chars} characters available for {budget}"
        )
    selected = [(text, path, row) for _, _, text, path, row in heap]
    rng.shuffle(selected)
    output, remaining, selected_stats = [], budget, collections.Counter()
    for text, path, row in selected:
        if remaining == 0:
            break
        offset = (
            rng.randrange(len(text) - remaining + 1) if len(text) > remaining else 0
        )
        text = text[offset : offset + remaining]
        output.append(
            {
                "text": text,
                "category": category,
                "source_shard": path,
                "source_row": row,
                "source_offset": offset,
            }
        )
        selected_stats.update(stats(text))
        remaining -= len(text)
    return output, {
        "target_chars": budget,
        "selected_stats": dict(selected_stats),
        "reference_stats": dict(observed),
        "shards": provenance,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--mode", choices=["smoke", "bounded", "full"], default="smoke")
    p.add_argument("--train-tokens", type=int, default=100_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=6200)
    p.add_argument(
        "--shards-per-category",
        type=int,
        default=0,
        help="0 scans all shards; positive values approximate using randomly selected shards",
    )
    p.add_argument("--smoke-scan-chars", type=int, default=32768)
    args = p.parse_args()
    if min(args.train_tokens, args.eval_tokens) < 10000 or args.shards_per_category < 0:
        p.error("Use at least 10,000 tokens per split and nonnegative shard count")
    if args.mode == "smoke" and max(args.train_tokens, args.eval_tokens) > 1_000_000:
        p.error(
            "Smoke mode is limited to 1M characters per split; explicitly select --mode full for 100M"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "incomplete",
        "dataset": REPO,
        "revision": REVISION,
        "mode": args.mode,
        "seed": args.seed,
        "shards_per_category": args.shards_per_category,
        "smoke_scan_chars": args.smoke_scan_chars,
        "weight_source": SOURCE,
        "raw_weights": WEIGHTS,
        "raw_weight_sum": sum(WEIGHTS.values()),
        "token_definition": "ASCII text characters; excludes end-of-document tokens added by Savanna",
        "sampling": "Seeded uniform random record priorities; retain lowest-priority records until character quota is covered, shuffle, randomly crop final record. Approximate within-category sampling, not an exact uniform token sample.",
        "limitations": [
            "Smoke and bounded modes scan prefixes; within-category representativeness is not guaranteed.",
            "Full scans all shards unless shards-per-category is set.",
            "Upstream splits retained; homology disjointness is not established.",
            "Exact category quotas do not guarantee species-level distribution.",
            "Record-size stopping and final cropping introduce finite-sample effects.",
        ],
        "splits": {},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for split in ("train", "valid", "test"):
            budget = args.train_tokens if split == "train" else args.eval_tokens
            targets = quotas(budget)
            output, detail = [], {}
            for category in WEIGHTS:
                print(f"{split}: {category}, quota={targets[category]}", flush=True)
                records, report = sample_category(
                    category, split, targets[category], args
                )
                output.extend(records)
                detail[category] = report
            random.Random(f"{args.seed}/{split}/shuffle").shuffle(output)
            path = args.output / f"{split}.jsonl"
            with path.open("w") as f:
                for record in output:
                    f.write(json.dumps(record) + "\n")
            manifest["splits"][split] = {
                "characters": sum(len(x["text"]) for x in output),
                "records": len(output),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "categories": detail,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception as error:
        manifest["error"] = str(error)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
