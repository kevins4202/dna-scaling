import concurrent.futures, gzip, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sample_opengenome as s
from repair_euk_sampling import choose

R = pathlib.Path(__file__).resolve().parents[1]
D = R / "data/mixture-100m-bounded"
REF = R / "data/reference-independent"
cat = "gtdb_v220_imgpr"
manifest = json.loads((D / "manifest.json").read_text())
used = {x["path"] for x in manifest["splits"]["train"]["categories"][cat]["shards"]}
files = [x for x in s.list_files(cat, "train") if x["path"] not in used]
random.Random("broader-independent-gtdb").shuffle(files)
files = files[:16]


def one(f):
    rows = []
    chars = 0
    with s.request(
        f"https://huggingface.co/datasets/{s.REPO}/resolve/{s.REVISION}/" + f["path"]
    ) as response, gzip.GzipFile(fileobj=response) as stream:
        for index, line in enumerate(stream, 1):
            raw = json.loads(line)
            text = raw["text"]
            for offset in range(0, len(text), 4096):
                part = text[offset : offset + 4096]
                rows.append(
                    {
                        "text": part,
                        "category": cat,
                        "source_shard": f["path"],
                        "source_row": index,
                        "source_offset": offset,
                        "source_record": raw.get("record"),
                    }
                )
                chars += len(part)
                if chars >= 2000000:
                    break
            if chars >= 2000000:
                break
    return choose(rows, 500000, random.Random("reference/" + f["path"])), {
        "path": f["path"],
        "scanned_characters": chars,
        "independent_shard": True,
    }


rows = []
sources = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for a, b in pool.map(one, files):
        rows.extend(a)
        sources.append(b)
(REF / f"{cat}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
(REF / f"{cat}.source.json").write_text(json.dumps(sources, indent=2) + "\n")
print(
    "GTDB_REFERENCE_BROADENED", len(rows), sum(len(x["text"]) for x in rows), flush=True
)
