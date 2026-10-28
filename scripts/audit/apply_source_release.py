"""Apply a staged source-only deployment after checking for unexpected remote edits."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DIRECTORIES = {"scripts", "configs", "docs", "archive", "provenance", "requirements"}
TOP_FILES = {
    "README.md",
    "MODELS.md",
    "SAMPLING.md",
    "TRAINING_READY.md",
    "CNN_EVALUATOR.md",
    "CNN_FROZEN.md",
    "SAVANNA_COMMIT",
    "CNN_BENCHMARK_COMMIT",
    ".gitignore",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--top-level-baseline", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())["source_files"]
    baseline.update(json.loads(args.top_level_baseline.read_text()))
    receipt = json.loads((args.incoming / "provenance/source-layout.json").read_text())
    incoming = [p for p in args.incoming.rglob("*") if p.is_file()]
    for path in incoming:
        relative = path.relative_to(args.incoming)
        if relative.parts[0] not in DIRECTORIES and str(relative) not in TOP_FILES:
            raise ValueError(f"Protected or unexpected deployment path: {relative}")
        target = args.root / relative
        if target.exists() and sha(target) != sha(path):
            expected = baseline.get(str(relative))
            if expected is None or sha(target) != expected:
                raise ValueError(f"Remote source changed since audit: {relative}")
    for record in receipt["files"]:
        target = args.root / record["original_path"]
        if target.exists() and sha(target) != record["sha256"]:
            raise ValueError(f"Cannot archive unexpected old source: {target}")
    for path in incoming:
        relative = path.relative_to(args.incoming)
        target = args.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    removed = []
    current_paths = {str(p.relative_to(args.incoming)) for p in incoming}
    for record in receipt["files"]:
        if record["original_path"] not in current_paths:
            target = args.root / record["original_path"]
            if target.exists():
                target.unlink()
                removed.append(record["original_path"])
    print(
        json.dumps(
            {
                "deployed_files": len(incoming),
                "archived_old_entrypoints": removed,
                "protected": "artifacts, checkpoints, datasets, vendor source, logs, and original results are untouched",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
