import copy
import json
import pathlib
import re


def encode_config(config):
    return re.sub(r"(?<= )(-?\d+)e([+-]\d+)", r"\1.0e\2", json.dumps(config, indent=2))


WIDTHS = {"5m": 160, "10m": 224, "25m": 352, "50m": 512, "100m": 704}


def build(root):
    base = json.loads((root / "configs/smoke.yml").read_text())
    destination = root / "configs/models"
    destination.mkdir(parents=True, exist_ok=True)
    models = []
    for name, width in WIDTHS.items():
        c = copy.deepcopy(base)
        c.update(
            {
                "hidden_size": width,
                "num_attention_heads": width // 32,
                "num_groups_hyena": width,
                "num_groups_hyena_medium": width // 16,
                "num_groups_hyena_short": width // 16,
                "num_groups_hyena_mlp": width // 16,
                "hyena_width_expansion": 1.0,
                "hyena_mlp_expansion_factor": 1.0,
                "hyena_mlp_len": 7,
                "seed": 6200,
                "train-iters": 3051,
                "lr-decay-iters": 3051,
                "gradient_accumulation_steps": 8,
                "eval-interval": 250,
                "eval-iters": 20,
                "checkpoint-factor": 500,
                "keep-last-n-checkpoints": 2,
                "save": f"../../checkpoints/{name}",
                "log-dir": f"../../logs/{name}",
                "master_port": 29502,
            }
        )
        (destination / f"{name}.yml").write_text(encode_config(c) + "\n")
        smoke = copy.deepcopy(c)
        smoke.update(
            {
                "train-iters": 100,
                "lr-decay-iters": 100,
                "gradient_accumulation_steps": 1,
                "eval-interval": 50,
                "eval-iters": 5,
                "checkpoint-factor": 100,
                "save": f"../../checkpoints/smoke-{name}",
                "log-dir": f"../../logs/smoke-{name}",
            }
        )
        (destination / f"{name}_smoke.yml").write_text(encode_config(smoke) + "\n")
        models.append(
            {
                "name": name,
                "target_parameters": int(name[:-1]) * 1000000,
                "layers": 24,
                "width": width,
                "attention_heads": width // 32,
                "projection_parameters_only": 24 * 8 * width * width,
                "actual_parameters": None,
                "status": "configuration created; GPU instantiation pending",
            }
        )
    for model in models:
        report_path = root / "reports" / (model["name"] + "-parameters.json")
        if report_path.exists():
            parameter_report = json.loads(report_path.read_text())
            if parameter_report["hidden_size"] == model["width"]:
                model["actual_parameters"] = parameter_report["total_parameters"]
                model["status"] = (
                    "Configuration created; actual parameter count verified separately"
                )
                completed_path = (
                    root / "reports/research" / (model["name"] + "-complete.json")
                )
                if completed_path.exists():
                    completed = json.loads(completed_path.read_text())
                    checked = completed.get("checkpoint_reload_and_generation", {})
                    if (
                        completed.get("status") == "verified_complete"
                        and completed.get("training_steps") == c["train-iters"]
                        and completed.get("processed_training_tokens") == 99975168
                        and checked.get("total_parameters")
                        == model["actual_parameters"]
                    ):
                        model["status"] = (
                            "Research training verified complete: 3051 steps, 99975168 tokens, strict checkpoint reload and generation"
                        )
    manifest = {
        "models": models,
        "sequence_length": 4096,
        "tokens_per_step_single_gpu": 32768,
        "train_steps": 3051,
        "training_tokens_single_gpu": 99975168,
        "size_note": "Projection count is a lower bound, not a total parameter estimate. Filters, convolutions, embeddings, biases and norms add parameters. Names are approximate targets pending GPU counts.",
        "comparison": "Same 24-layer operator schedule, head dimension 32, Hyena grouping ratio, tokenizer, context, seed, effective batch, optimizer, learning-rate schedule and token budget; only width/head/group counts differ.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    print(json.dumps(build(root), indent=2))
