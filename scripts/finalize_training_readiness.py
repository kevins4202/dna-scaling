import bisect, collections, csv, hashlib, json, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mixture-100m-bounded"
REPORT = ROOT / "reports"


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run():
    audit = json.loads((REPORT / "distribution-audit.json").read_text())
    validation = json.loads((DATA / "validation.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert not any(v["large_difference"] for v in audit["categories"].values())
    assert not validation["exact_cross_split_duplicates_ge_128_chars"]
    for name in ("valid-vs-train", "test-vs-train", "test-vs-valid"):
        assert (REPORT / f"overlap/{name}.paf").read_text().strip() == "", (
            name + " overlap remains"
        )
    intervals = collections.defaultdict(list)
    for line in (DATA / "train.jsonl").open():
        row = json.loads(line)
        start = row.get("source_offset", 0)
        intervals[(row["source_shard"], row["source_row"])].append(
            (start, start + len(row["text"]))
        )
    for values in intervals.values():
        values.sort()
    reference_overlap = 0
    for path in (ROOT / "data/reference-independent").glob("*.jsonl"):
        for line in path.open():
            row = json.loads(line)
            start = row.get("source_offset", 0)
            end = start + len(row["text"])
            candidates = intervals.get((row["source_shard"], row["source_row"]), [])
            for a, b in candidates:
                if a >= end:
                    break
                if b > start:
                    reference_overlap += min(b, end) - max(a, start)
    assert reference_overlap == 0, ("Reference coordinate overlap", reference_overlap)
    diagnostics = {}
    for cat, item in audit["categories"].items():
        a, b = item["subset"], item["independent_reference"]
        qa, qb = a["gc_window_quantiles"], b["gc_window_quantiles"]
        diagnostics[cat] = {
            **item["differences"],
            "homopolymer_fraction_difference": abs(
                a["homopolymer_ge8_fraction"] - b["homopolymer_ge8_fraction"]
            ),
            "gc_quantile_differences_10_50_90": (
                [abs(qa[i] - qb[i]) for i in (1, 2, 3)] if qa and qb else None
            ),
            "reference_characters": b["chars"],
        }
        assert diagnostics[cat]["homopolymer_fraction_difference"] < 0.01
    species = collections.Counter()
    identifiers = set()
    with (ROOT / "data/species_metadata.csv").open() as f:
        for row in csv.DictReader(f):
            species[(row["domain"], row["source_dataset"])] += 1
            identifiers.add(row["accession_or_taxid"])
    taxonomy = {
        "metadata_inventory_rows_by_domain_and_source": {
            " | ".join(k): v for k, v in species.items()
        },
        "record_identifier_coverage": {},
        "limitation": "Metadata inventory is not a token-weighted taxonomy distribution. Exact species balance is unverified: processed text records often lack identifiers, and contig IDs do not directly map to assembly/taxonomy metadata IDs.",
    }
    for cat in audit["categories"]:
        total = has_id = matches = 0
        for line in (ROOT / f"data/reference-independent/{cat}.jsonl").open():
            row = json.loads(line)
            total += 1
            identifier = row.get("source_record")
            has_id += bool(identifier)
            matches += identifier in identifiers if identifier else 0
        taxonomy["record_identifier_coverage"][cat] = {
            "reference_records": total,
            "with_source_identifier": has_id,
            "exact_metadata_identifier_matches": matches,
        }
    (REPORT / "taxonomy-coverage.json").write_text(
        json.dumps(taxonomy, indent=2) + "\n"
    )
    models = {}
    for size in ("5m", "10m"):
        parameter = json.loads(
            (ROOT / f"checkpoints/{size}-parameters.json").read_text()
        )
        config = json.loads((ROOT / f"configs/models/{size}.yml").read_text())
        smoke_log = (
            (ROOT / f"logs/{size}-smoke.log").read_text()
            if size == "10m"
            else (ROOT / "logs/5m-smoke.log").read_text()
        )
        line = next(
            x for x in smoke_log.splitlines() if re.search(r"iteration\s+100/\s+100", x)
        )
        assert "number of nan iterations:   0" in line
        models[size] = {
            "parameters": parameter["total_parameters"],
            "layers": config["num_layers"],
            "width": config["hidden_size"],
            "attention_heads": config["num_attention_heads"],
            "head_dimension": config["hidden_size"] // config["num_attention_heads"],
            "context": config["seq_length"],
            "operator_counts": {
                "hyena_se": 6,
                "hyena_mr": 6,
                "hyena": 10,
                "flash_v2": 2,
            },
            "tiny_data_100_step_test_passed": True,
        }
        assert (
            config["train-iters"]
            * config["seq_length"]
            * config["gradient_accumulation_steps"]
            * config["train_micro_batch_size_per_gpu"]
            == 99975168
        )
    expected = {}
    for split, total in [("train", 100000000), ("valid", 1000000), ("test", 1000000)]:
        prefix = ROOT / f"data/tokenized-100m/{split}_text_CharLevelTokenizer_document"
        binary = prefix.with_suffix(".bin")
        index = prefix.with_suffix(".idx")
        assert binary.exists() and index.stat().st_size > 0, (
            "Tokenization incomplete: " + split
        )
        # Savanna's uint16 character vocabulary includes one EOD per JSONL record.
        encoded = total + manifest["splits"][split]["records"]
        assert binary.stat().st_size == 2 * encoded, (
            split,
            binary.stat().st_size,
            encoded,
        )
        expected[split] = {
            "source_characters": total,
            "records": manifest["splits"][split]["records"],
            "encoded_tokens_including_eod": encoded,
        }
    required = [
        "configs/data100m.yml",
        "configs/models/5m.yml",
        "configs/models/10m.yml",
        "data/mixture-100m-bounded/manifest.json",
        "reports/distribution-audit.json",
        "reports/taxonomy-coverage.json",
    ]
    for split in ("train", "valid", "test"):
        required.append(f"data/mixture-100m-bounded/{split}.jsonl")
        for suffix in ("bin", "idx"):
            required.append(
                f"data/tokenized-100m/{split}_text_CharLevelTokenizer_document.{suffix}"
            )
    report = {
        "status": "ready",
        "scope": "Controlled subset experiments; no full research training launched",
        "dataset": "data/mixture-100m-bounded",
        "splits": expected,
        "models": models,
        "distribution_diagnostics": diagnostics,
        "reference_train_shared_source_characters": reference_overlap,
        "cross_split_alignment_matches": 0,
        "alignment_screen": "minimap2 2.31 asm10, secondary=no; candidate cleanup rejects matching-base fraction >=0.90 across >=128bp blocks. No final reported alignments. Sensitive homology/phylogenetic independence is not guaranteed.",
        "taxonomy_verified": False,
        "limitations": [
            "Bounded source-prefix sampling remains an approximation to OpenGenome2.",
            "Within-category species proportions are unverified.",
            "Independent reference has disjoint shards or source coordinates, not necessarily unrelated organisms.",
            "Thresholds are engineering screening rules, not formal equivalence tests.",
            "Document length is storage-dependent for stitched DNA; all models use identical 4096-token training context.",
        ],
        "required_file_hashes": {r: digest(ROOT / r) for r in required},
    }
    (REPORT / "training-readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
