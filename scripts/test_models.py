import collections
import json
import pathlib
import shutil
import subprocess
import tempfile

import build_models
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ModelFamilyTests(unittest.TestCase):
    def test_family_controls_and_shapes(self):
        baseline = None
        allowed = {
            "hidden_size",
            "num_attention_heads",
            "num_groups_hyena",
            "num_groups_hyena_medium",
            "num_groups_hyena_short",
            "num_groups_hyena_mlp",
            "save",
            "log-dir",
        }
        for size in ("5m", "10m", "25m", "50m", "100m"):
            c = json.loads((ROOT / f"configs/models/{size}.yml").read_text())
            self.assertEqual(c["hidden_size"] // c["num_attention_heads"], 32)
            for key in (
                "num_groups_hyena",
                "num_groups_hyena_medium",
                "num_groups_hyena_short",
                "num_groups_hyena_mlp",
            ):
                self.assertEqual(c["hidden_size"] % c[key], 0)
            counts = collections.Counter()
            for operators, repeat in c["operator-config"]:
                for op in operators:
                    counts[op] += repeat
            self.assertEqual(
                counts, {"hyena_se": 6, "hyena_mr": 6, "hyena": 10, "flash_v2": 2}
            )
            fixed = {k: v for k, v in c.items() if k not in allowed}
            if baseline is not None:
                self.assertEqual(fixed, baseline)
            baseline = fixed
            smoke = json.loads((ROOT / f"configs/models/{size}_smoke.yml").read_text())
            self.assertNotEqual(smoke["save"], c["save"])
            self.assertEqual(smoke["train-iters"], 100)
            self.assertEqual(
                c["train-iters"]
                * c["seq_length"]
                * c["train_micro_batch_size_per_gpu"]
                * c["gradient_accumulation_steps"],
                99975168,
            )

    def test_generator_retains_verified_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "configs").mkdir()
            (root / "reports/research").mkdir(parents=True)
            shutil.copy2(ROOT / "configs/smoke.yml", root / "configs/smoke.yml")
            (root / "reports/5m-parameters.json").write_text(
                json.dumps({"hidden_size": 160, "total_parameters": 5413780})
            )
            (root / "reports/research/5m-complete.json").write_text(
                json.dumps(
                    {
                        "status": "verified_complete",
                        "training_steps": 3051,
                        "processed_training_tokens": 99975168,
                        "checkpoint_reload_and_generation": {
                            "total_parameters": 5413780
                        },
                    }
                )
            )
            result = build_models.build(root)
            first = result["models"][0]
            self.assertEqual(first["actual_parameters"], 5413780)
            self.assertIn("Research training verified complete", first["status"])
            self.assertIsNone(result["models"][2]["actual_parameters"])

    def test_training_commands_can_be_reviewed_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "scripts").mkdir()
            (root / "reports/research").mkdir(parents=True)
            for name in ("run_model.sh", "queue_research_gpu7.sh"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            for size in ("5m", "10m"):
                (root / f"reports/research/{size}-complete.json").write_text(
                    json.dumps({"status": "verified_complete"})
                )
            command = subprocess.run(
                [
                    "bash",
                    str(root / "scripts/run_model.sh"),
                    "25m",
                    "train",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("GPU 7", command.stdout)
            self.assertIn("readiness", command.stdout)
            queue = subprocess.run(
                [
                    "bash",
                    str(root / "scripts/queue_research_gpu7.sh"),
                    "--dry-run",
                    "5m",
                    "10m",
                    "25m",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("SKIP 5m", queue.stdout)
            self.assertIn("SKIP 10m", queue.stdout)
            self.assertIn("PLAN 25m", queue.stdout)


if __name__ == "__main__":
    unittest.main()
