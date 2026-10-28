import argparse, csv, hashlib, json, pathlib, re

root = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Export verified research metrics without changing prior comparisons")
parser.add_argument("--models", nargs="+", choices=("5m", "10m", "25m", "50m", "100m"), default=["5m", "10m"])
parser.add_argument("--output", default="reports/research")
args = parser.parse_args()
out = root / args.output
out.mkdir(parents=True, exist_ok=True)
summary = {}
for size in args.models:
    complete = json.loads((root / "reports/research" / f"{size}-complete.json").read_text())
    checkpoint = pathlib.Path(complete["final_checkpoint"])
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == complete["checkpoint_sha256"]
    )
    rows = []
    evaluations = {}
    for line in (root / f"logs/research/{size}-train.log").open():
        match = re.search(
            r"iteration\s+(\d+)/\s*3051.*?learning rate:\s*([\d.Ee+-]+).*?token index into training data:\s*(\d+).*?lm_loss:\s*([\d.Ee+-]+)",
            line,
        )
        if match:
            rows.append(
                {
                    "step": int(match[1]),
                    "learning_rate": float(match[2]),
                    "processed_tokens": int(match[3]),
                    "training_loss": float(match[4]),
                }
            )
        match = re.search(
            r"(validation|test) results at the end of training.*?lm_loss value:\s*([\d.Ee+-]+).*?lm_loss_ppl value:\s*([\d.Ee+-]+)",
            line,
        )
        if match:
            evaluations[match[1]] = {
                "loss": float(match[2]),
                "perplexity": float(match[3]),
            }
    with (out / f"{size}-training-metrics.csv").open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["step", "learning_rate", "processed_tokens", "training_loss"]
        )
        w.writeheader()
        w.writerows(rows)
    summary[size] = {
        "parameters": complete["checkpoint_reload_and_generation"]["total_parameters"],
        "steps": complete["training_steps"],
        "processed_tokens": complete["processed_training_tokens"],
        "evaluations": evaluations,
        "final_checkpoint": complete["final_checkpoint"],
        "checkpoint_sha256": complete["checkpoint_sha256"],
        "strict_reload_and_generation_passed": True,
    }
(out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
