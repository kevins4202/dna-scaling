"""Locate, verify, and invoke the frozen CNN without changing its implementation."""

import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE = PROJECT_ROOT / "artifacts/cnn-v1"


def verify_integrity(release=RELEASE):
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest.get("version") != "cnn-v1" or manifest.get("frozen") is not True:
        raise ValueError("Expected the frozen cnn-v1 artifact")
    for name, expected in manifest["files_sha256"].items():
        path = release / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Invalid artifact file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return manifest


def scoring_module(release=RELEASE):
    verify_integrity(release)
    spec = importlib.util.spec_from_file_location(
        "frozen_project_cnn_score", release / "score.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(filename):
    if filename not in ("score.py", "evaluate.py"):
        raise ValueError("Unknown frozen CNN command")
    verify_integrity()
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(RELEASE))
        runpy.run_path(str(RELEASE / filename), run_name="__main__")
    finally:
        sys.path[:] = previous
