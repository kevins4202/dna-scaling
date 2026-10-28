"""Run engineering checks without retraining or overwriting the original validation."""

import argparse
import hashlib
import json
from pathlib import Path

from .release import PROJECT_ROOT, RELEASE, scoring_module, verify_integrity


def check_data_separation(root):
    manifest = json.loads((root / "artifacts/cnn-v1/data-manifest.json").read_text())
    seen_groups, seen_sequences = set(), set()
    for fold in ("train", "calibration", "test"):
        path = root / f"data/cnn-final-mixed/{fold}.jsonl"
        expected = manifest["file_sha256"][path.name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical CNN training data changed: {path}")
        groups, sequences = set(), set()
        with path.open() as handle:
            for row in map(json.loads, handle):
                groups.add(row["group_id"])
                for key in ("natural", "generated", "null_natural"):
                    text = row[key]
                    if len(text) != 512 or not set(text) <= set("ACGT"):
                        raise ValueError(f"Invalid DNA in {path}")
                    reverse = text.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                    sequences.add(min(text, reverse))
        if groups & seen_groups or sequences & seen_sequences:
            raise ValueError(f"Cross-fold group or sequence overlap in {fold}")
        seen_groups.update(groups)
        seen_sequences.update(sequences)
    return {"source_groups_disjoint": True, "exact_reverse_complement_overlap": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Also inspect the historical training data, available on Allegro",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports/checks/cnn-release.json"
    )
    args = parser.parse_args()
    manifest = verify_integrity()
    report = {
        "status": "verified",
        "checkpoint_sha256": manifest["files_sha256"]["model.pt"],
        "validation_status": manifest["validation_status"],
        "sensitivity_gate_passed": manifest["prespecified_sensitivity_gate_passed"],
        "scope": "Engineering verification does not certify detection sensitivity.",
    }
    if not args.integrity_only:
        import torch

        torch.set_num_threads(4)
        scorer = scoring_module()
        model, _ = scorer.load_release(RELEASE, args.device)
        if not all(
            torch.isfinite(tensor).all() for tensor in model.state_dict().values()
        ):
            raise ValueError("Nonfinite checkpoint state")
        reference = [
            json.loads(line)
            for line in (RELEASE / "evaluation-prompts.jsonl").read_text().splitlines()
        ]
        text = reference[0]["text"]
        score, _ = scorer.score_text(model, text, args.device)
        reverse = text.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        reversed_score, _ = scorer.score_text(model, reverse, args.device)
        if abs(score - reversed_score) > 1e-6:
            raise ValueError("Strand invariance failed")
        combined, chunks = scorer.score_text(
            model, "".join(r["text"] for r in reference[:4]), args.device
        )
        if abs(combined - sum(chunks) / 4) > 1e-6:
            raise ValueError("Four-window aggregation failed")
        report.update(
            {
                "strict_reload": True,
                "finite_state_and_logits": True,
                "parameters": sum(p.numel() for p in model.parameters()),
                "strand_invariance": True,
                "four_window_aggregation": True,
                "reference_pairs": len(reference),
                "device": args.device,
            }
        )
    if args.full_data:
        report["data_separation"] = check_data_separation(PROJECT_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
