"""Validate budgets, provenance, and basic distribution summaries."""

import argparse
import collections
import hashlib
import json
import pathlib


def validate(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Dataset is incomplete")
    summary = {
        "mode": manifest["mode"],
        "splits": {},
        "exact_cross_split_duplicates_ge_128_chars": [],
    }
    seen = {}
    for split, report in manifest["splits"].items():
        path = root / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["sha256"]:
            raise ValueError(f"{split}: file hash mismatch")
        counts = collections.Counter()
        source_records = set()
        for line in path.open():
            row = json.loads(line)
            text = row["text"]
            assert text and text.isascii()
            assert f"_{split}_" in row["source_shard"]
            counts[row["category"]] += len(text)
            source_records.add((row["source_shard"], row["source_row"]))
            if len(text) >= 128:
                digest = hashlib.sha256(text.upper().encode()).hexdigest()
                if digest in seen and seen[digest] != split:
                    summary["exact_cross_split_duplicates_ge_128_chars"].append(
                        [seen[digest], split, digest]
                    )
                seen[digest] = split
        assert sum(counts.values()) == report["characters"]
        categories = {}
        for category, detail in report["categories"].items():
            assert counts[category] == detail["target_chars"]
            selected, reference = detail["selected_stats"], detail["reference_stats"]
            categories[category] = {
                "characters": counts[category],
                "fraction": counts[category] / report["characters"],
                "sample_gc": selected["gc"] / max(1, selected["acgt"]),
                "reference_gc": reference["gc"] / max(1, reference["acgt"]),
                "sample_acgt_fraction": selected["acgt"] / selected["chars"],
            }
        summary["splits"][split] = {
            "characters": sum(counts.values()),
            "source_records": len(source_records),
            "categories": categories,
        }
    summary["homology_checked"] = False
    summary["representative"] = (
        False
        if manifest["mode"] == "smoke"
        else "requires review of category diagnostics and species coverage; not guaranteed"
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=pathlib.Path)
    args = p.parse_args()
    report = validate(args.dataset)
    (args.dataset / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["characters"] for k, v in report["splits"].items()}))
    print(
        "Exact cross-split duplicates >=128 characters:",
        len(report["exact_cross_split_duplicates_ge_128_chars"]),
    )
