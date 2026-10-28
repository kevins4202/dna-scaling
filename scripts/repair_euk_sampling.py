import concurrent.futures, gzip, hashlib, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sample_opengenome as sampler

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mixture-100m-bounded"
REF = ROOT / "data/reference-independent"
CATEGORY = "eukaryotic_genic_windows"


def allocation(total, files):
    exact = [total * f["size"] / sum(x["size"] for x in files) for f in files]
    q = [int(x) for x in exact]
    for i in sorted(range(len(q)), key=lambda i: exact[i] - q[i], reverse=True)[
        : total - sum(q)
    ]:
        q[i] += 1
    return q


def choose(pool, budget, rng):
    rng.shuffle(pool)
    result = []
    for row in pool:
        if budget <= 0:
            break
        row = dict(row)
        if len(row["text"]) > budget:
            delta = rng.randrange(len(row["text"]) - budget + 1)
            row["text"] = row["text"][delta : delta + budget]
            row["source_offset"] += delta
        budget -= len(row["text"])
        result.append(row)
    if budget:
        raise ValueError("insufficient candidate windows")
    return result


def one(job):
    shard, train_budget, ref_budget = job
    rng = random.Random("broad-euk-6200/" + shard["path"])
    train_pool = []
    ref_pool = []
    train_chars = ref_chars = 0
    train_limit = 4 * train_budget
    ref_limit = 4 * ref_budget
    lastrow = 0
    with sampler.request(
        f"https://huggingface.co/datasets/{sampler.REPO}/resolve/{sampler.REVISION}/"
        + shard["path"]
    ) as response, gzip.GzipFile(fileobj=response) as stream:
        done = False
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            text = row["text"]
            lastrow = index
            for offset in range(0, len(text), 4096):
                fragment = text[offset : offset + 4096]
                record = {
                    "text": fragment,
                    "category": CATEGORY,
                    "source_shard": shard["path"],
                    "source_row": index,
                    "source_offset": offset,
                    "source_record": row.get("record"),
                }
                if train_chars < train_limit:
                    train_pool.append(record)
                    train_chars += len(fragment)
                elif ref_chars < ref_limit:
                    ref_pool.append(record)
                    ref_chars += len(fragment)
                else:
                    done = True
                    break
            if done:
                break
    train = choose(train_pool, train_budget, rng)
    ref = choose(
        ref_pool, ref_budget, random.Random("reference-broad/" + shard["path"])
    )
    print("SHARD_READY", shard["path"], train_budget, ref_budget, flush=True)
    return (
        train,
        ref,
        {
            "path": shard["path"],
            "records_scanned": lastrow,
            "train_pool_chars": train_chars,
            "reference_pool_chars": ref_chars,
            "complete_scan": False,
        },
    )


def run():
    manifest = json.loads((DATA / "manifest.json").read_text())
    files = sampler.list_files(CATEGORY, "train")
    budget = manifest["splits"]["train"]["categories"][CATEGORY]["target_chars"]
    train_alloc = allocation(budget, files)
    ref_alloc = allocation(2000000, files)
    train = []
    ref = []
    sources = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for a, b, c in pool.map(one, zip(files, train_alloc, ref_alloc)):
            train.extend(a)
            ref.extend(b)
            sources.append(c)
    old = [json.loads(x) for x in (DATA / "train.jsonl").open()]
    output = [r for r in old if r["category"] != CATEGORY] + train
    random.Random("repaired-mixture-6200").shuffle(output)
    (DATA / "train.before-euk-repair.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in old)
    )
    (DATA / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in output))
    (REF / f"{CATEGORY}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ref))
    (REF / f"{CATEGORY}.source.json").write_text(json.dumps(sources, indent=2) + "\n")
    item = manifest["splits"]["train"]
    item["sha256"] = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    item["records"] = len(output)
    selected = {}
    for row in train:
        for k, v in sampler.stats(row["text"]).items():
            selected[k] = selected.get(k, 0) + v
    item["categories"][CATEGORY].update(
        {
            "selected_stats": selected,
            "shards": sources,
            "repair": "All 40 source shards, allocation proportional to compressed shard size; uniform bounded 4096-character candidate windows; independent reference from subsequent nonoverlapping regions. Corrects domination by very long source records. Exact within-shard population distribution remains unverified.",
        }
    )
    manifest["repair_note"] = (
        "Eukaryotic category rebuilt across all 40 shards using bounded windows; independent reference uses disjoint subsequent regions."
    )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ROOT / "reports/distribution-audit.json"
    if report.exists():
        (ROOT / "reports/distribution-audit-before.json").write_text(report.read_text())
    print("REPAIR_COMPLETE", flush=True)


if __name__ == "__main__":
    run()
