"""Record installed packages and detect edits to Python files in runtime dependencies."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

CHECK_DISTRIBUTIONS = (
    "torch",
    "deepspeed",
    "transformer-engine",
    "transformer-engine-cu12",
    "transformer-engine-torch",
    "flash-attn",
    "ring-flash-attn",
    "causal-conv1d",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements-output", type=Path)
    args = parser.parse_args()
    packages = sorted(
        {(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()}
    )
    checks = {}
    for name in CHECK_DISTRIBUTIONS:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        checked, modified, unverified = 0, [], []
        for entry in dist.files or ():
            if not str(entry).endswith(".py"):
                continue
            path = dist.locate_file(entry)
            if not entry.hash or entry.hash.mode != "sha256" or not path.is_file():
                unverified.append(str(entry))
                continue
            actual = (
                base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            checked += 1
            if actual != entry.hash.value:
                modified.append(str(entry))
        checks[name] = {
            "version": dist.version,
            "python_files_verified": checked,
            "modified_python_files": modified,
            "unverified_python_files": unverified,
        }
    conda = shutil.which("conda") or str(Path(sys.prefix).parents[1] / "bin/conda")
    conda_packages = []
    if Path(conda).is_file():
        result = subprocess.run(
            [conda, "list", "--prefix", sys.prefix, "--json"],
            text=True,
            capture_output=True,
            check=True,
        )
        conda_packages = [
            {key: row.get(key) for key in ("name", "version", "build_string")}
            for row in json.loads(result.stdout)
        ]
    report = {
        "python": sys.version,
        "prefix": sys.prefix,
        "platform": sys.platform,
        "packages": [{"name": name, "version": version} for name, version in packages],
        "conda_packages": conda_packages,
        "installed_code_checks": checks,
        "scope": "A snapshot of the working environment, not a claim that a cold install was tested. CUDA hardware was not initialized by this audit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.requirements_output:
        args.requirements_output.parent.mkdir(parents=True, exist_ok=True)
        args.requirements_output.write_text(
            "# Snapshot of Allegro Linux/Python 3.12; see docs/REPRODUCIBILITY.md.\n"
            + "".join(f"{name}=={version}\n" for name, version in packages)
        )
    print(
        json.dumps(
            {
                "packages": len(packages),
                "conda_packages": len(conda_packages),
                "installed_code_checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
