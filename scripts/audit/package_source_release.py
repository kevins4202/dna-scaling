"""Build a source-only tarball without macOS sidecar metadata or experiment artifacts."""

import argparse
import tarfile
from pathlib import Path

from apply_source_release import DIRECTORIES, TOP_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for name in sorted(DIRECTORIES):
        for path in sorted((args.root / name).rglob("*")):
            relative = path.relative_to(args.root)
            if (
                path.is_file()
                and "__pycache__" not in relative.parts
                and not any(part.startswith("._") for part in relative.parts)
            ):
                if path.is_symlink():
                    raise ValueError(f"Unexpected source symlink: {relative}")
                files.append(path)
    files.extend(
        args.root / name for name in sorted(TOP_FILES) if (args.root / name).is_file()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(args.root)), recursive=False)
    print(
        f"Packaged {len(files)} source files; artifacts, data, checkpoints and results excluded"
    )


if __name__ == "__main__":
    main()
