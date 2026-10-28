"""One-time source-layout migration. Original code is preserved before any move."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

GROUPS = {
    "cnn-pilot": [
        "prepare_cnn_data.py",
        "generate_cnn_samples.py",
        "train_cnn_evaluator.py",
        "score_pilot_cnn.py",
        "test_cnn_evaluator.py",
        "finish_cnn_evaluation.py",
    ],
    "cnn-external": [
        "prepare_project_cnn.py",
        "train_project_cnn.py",
        "queue_project_cnn.sh",
    ],
    "cnn-final": [
        "prepare_project_cnn_transfer.py",
        "generate_project_cnn_samples.py",
        "build_final_cnn_data.py",
        "train_final_project_cnn.py",
        "freeze_project_cnn.py",
        "queue_project_cnn_generation.sh",
        "queue_project_cnn_finish.sh",
        "verify_project_cnn.py",
        "test_project_cnn.py",
        "score_project_cnn.py",
        "evaluate_project_cnn.py",
        "score_cnn.py",
    ],
    "initial-setup": [
        "prepare_subset.py",
        "prepare_full_dataset.py",
        "setup_gpu.sh",
        "setup_allegro.sh",
        "queue_gpu7_smoke.sh",
        "verify_checkpoint_gpu.py",
    ],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--remote-staging", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = root / "provenance/source-layout.json"
    if receipt.exists():
        raise SystemExit(
            "Migration already recorded; refusing to archive new code over old code"
        )
    plan = []
    for phase, names in GROUPS.items():
        for name in names:
            source = root / "scripts" / name
            if source.is_file():
                plan.append((source, root / f"archive/{phase}/scripts/{name}", "move"))
    for source in sorted((args.remote_staging / "scripts").glob("*")):
        if source.is_file() and not (root / "scripts" / source.name).exists():
            plan.append(
                (source, root / f"archive/recovery/scripts/{source.name}", "copy")
            )
    print(
        json.dumps(
            [
                {"source": str(a), "destination": str(b), "action": action}
                for a, b, action in plan
            ],
            indent=2,
        )
    )
    if not args.apply:
        return
    receipt.parent.mkdir(parents=True, exist_ok=True)
    snapshot = receipt.parent / "source-before-organization.tar.gz"
    with tarfile.open(snapshot, "w:gz") as archive:
        for directory in ("scripts", "configs"):
            for path in sorted((root / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(
                        path, arcname=str(path.relative_to(root)), recursive=False
                    )
        for path in sorted((args.remote_staging / "scripts").glob("*")):
            if path.is_file() and not (root / "scripts" / path.name).exists():
                archive.add(
                    path, arcname=f"remote-only/scripts/{path.name}", recursive=False
                )
        archive.add(
            args.remote_staging / "configs/models/manifest.json",
            arcname="remote-before/configs/models/manifest.json",
        )
    records = []
    for source, destination, action in plan:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Archive destination exists: {destination}")
        sha = digest(source)
        if action == "move":
            source.rename(destination)
        else:
            shutil.copy2(source, destination)
        assert digest(destination) == sha
        records.append(
            {
                "original_path": f"scripts/{source.name}",
                "archived_path": str(destination.relative_to(root)),
                "sha256": sha,
                "origin": "remote-only" if action == "copy" else "local-and-remote",
            }
        )
    receipt.write_text(
        json.dumps(
            {
                "version": 1,
                "archive_sha256": digest(snapshot),
                "files": records,
                "policy": "Historical source, including old relative paths, is preserved verbatim. Do not execute archive repair/training scripts against the live frozen project.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
