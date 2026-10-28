import hashlib, json, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[1]
report = json.loads((root / "reports/training-readiness.json").read_text())
if report.get("status") != "ready":
    raise SystemExit("Dataset/model readiness checks have not passed")
size = sys.argv[1]
if size not in report["models"]:
    raise SystemExit("Model not verified in readiness report")
for relative, digest in report["required_file_hashes"].items():
    path = root / relative
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != digest:
        raise SystemExit("Readiness artifact changed: " + relative)
print("Verified training readiness:", size)
