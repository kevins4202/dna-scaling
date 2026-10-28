"""Inventory project source and verify immutable CNN files without importing ML code."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SUFFIXES = {
    ".py",
    ".sh",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".md",
    ".txt",
    ".c",
    ".cc",
    ".hpp",
    ".in",
    ".cfg",
    ".ini",
    ".mk",
    ".bash",
    ".cpp",
    ".h",
    ".cu",
    ".cuh",
}
SKIP = {".git", "__pycache__", "build", "dist", "node_modules"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, names):
    files = {}
    for name in names:
        base = root / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and (
                    path.suffix in SOURCE_SUFFIXES
                    or path.name in {"Makefile", "CMakeLists.txt", "Dockerfile"}
                )
                and not SKIP.intersection(path.relative_to(base).parts)
            ):
                files[str(path.relative_to(root))] = sha256(path)
    return files


def verify_release(root):
    release = root / "artifacts/cnn-v1"
    manifest = json.loads((release / "manifest.json").read_text())
    changed = []
    for name, expected in manifest["files_sha256"].items():
        path = release / name
        if Path(name).name != name or not path.is_file() or sha256(path) != expected:
            changed.append(name)
    return {
        "intact": not changed,
        "changed_files": changed,
        "manifest_sha256": sha256(release / "manifest.json"),
        "checkpoint_sha256": manifest["files_sha256"]["model.pt"],
        "validation_status": manifest["validation_status"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {
        "source_files": inventory(
            root, ("scripts", "configs", "patches", "requirements", "archive")
        ),
        "vendor_files": inventory(
            root, ("vendor/savanna", "vendor/synthetic_genomes_benchmark")
        ),
        "frozen_cnn": verify_release(root),
    }
    if args.compare:
        prior = json.loads(args.compare.read_text())
        result["comparison"] = {}
        for section in ("source_files", "vendor_files"):
            a, b = prior[section], result[section]
            result["comparison"][section] = {
                "only_baseline": sorted(set(a) - set(b)),
                "only_current": sorted(set(b) - set(a)),
                "changed": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_files": len(result["source_files"]),
                "vendor_files": len(result["vendor_files"]),
                "frozen_cnn": result["frozen_cnn"],
                "comparison": result.get("comparison"),
            },
            indent=2,
        )
    )
    if not result["frozen_cnn"]["intact"]:
        raise SystemExit("Frozen CNN integrity check failed")


if __name__ == "__main__":
    main()
