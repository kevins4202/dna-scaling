# DNA generation scaling project

We train a controlled StripedHyena 2 model family with Savanna and evaluate DNA continuations with a frozen project CNN and a separately reproduced benchmark CNN procedure. Detector rankings are evaluator-sensitive and do not establish biological quality. The original proposal is [proposal/main.tex](proposal/main.tex); the current execution plan is [docs/PLAN.md](docs/PLAN.md).

## Current state

| Component | State |
|---|---|
| OpenGenome2 subset | Prepared and screened on Allegro: 100M training characters plus 1M validation and 1M test characters. Bounded sampling; species balance remains unverified. |
| 5M model | 5,413,780 parameters; 3,051 training steps and 99,975,168 processed tokens; checkpoint reload and sampling verified. |
| 10M model | 10,331,804 parameters; same steps and token budget; checkpoint reload and sampling verified. |
| 25M / 50M / 100M | Trained and verified at 24,886,444 / 51,927,104 / 97,352,024 parameters, with the same steps and token budget as 5M and 10M. |
| CNN v1 | One frozen 1,227,905-parameter checkpoint. Engineering checks passed; detection sensitivity is limited. |
| Distance experiment | All five models completed the same 100-prompt, 2,048-base continuation experiment. No robust monotonic size trend was established. |
| Released Evo 2 7B reference | Completed the same prompt panel with frozen CNN scoring; different pretraining history prevents a controlled size comparison. |
| Original benchmark CNN reproduction | All 210 folds completed and independently verified; strong detection recovered on the original benchmark. |
| Six-model CNN reassessment | All six saved generation sets reevaluated with the same fixed ensemble of 30 reproduced detectors. All 100 prompts passed the registered known-training overlap screen; rankings differ from the original project CNN. |

CNN v1's fresh-test AUROCs are 0.5452 for 5M and 0.5847 for 10M. The 5M interval includes chance, so this is a fixed diagnostic, not a validated sole quality measure or RL reward. Keep its weights and scoring rules unchanged for all subsequent models. See [CNN contract](CNN_FROZEN.md), [CNN results](reports/cnn-final/RESULTS.md), and [LM results](reports/research/RESULTS.md).

Completed experiment reports: [controlled model family](reports/distance-2048-100m-v1/RESULTS.md), [released reference](reports/evo7b-baseline/SUMMARY.md), [original CNN reproduction](reports/cnn-original-v1/SUMMARY.md), and [six-model reassessment](reports/cnn-reassessment-v1/SUMMARY.md). Independent structure measurements and evaluator transfer validation remain necessary before making biological-quality or scaling claims.

## Commands used now

From the project root:

```bash
# Defaults to CPU. Input contains uppercase ACGT continuations, excluding prompts.
python scripts/score_cnn.py --input generated.jsonl --output scores.csv

# Natural and generated files must have matching unique ids and equal lengths.
python scripts/evaluate_project_cnn.py \
  --natural artifacts/cnn-v1/evaluation-prompts.jsonl \
  --generated reports/cnn-final/test-generated-5m.jsonl \
  --output reports/checks/5m-evaluation.json

# Engineering verification, not certification of biological quality.
python scripts/verify_project_cnn.py
python -m unittest discover -s scripts -p 'test_*.py'

# Inspect existing configurations without allocating a GPU or launching work.
bash scripts/run_model.sh 25m inspect --dry-run
bash scripts/queue_research_gpu7.sh --dry-run 25m 50m 100m
```

Training runs on Allegro, `/home/ksong37/cis6200`, using physical GPU 7. Source `scripts/gpu7_env.sh` before CUDA commands. The local repository is the source of truth for code; Allegro is a checked deployment, not a Git checkout. New model sizes cannot train until their own readiness checks pass. No new training is launched by the commands above.

## Repository map

- `scripts/`: current data, model, training, and scoring commands. [Command index](scripts/README.md).
- `scripts/cnn/`: small interfaces to the frozen implementation; no CNN training code.
- `artifacts/cnn-v1/`: immutable weights, original scoring/evaluation implementation, reference prompts, provenance, and SHA-256 manifest.
- `configs/`: shared data settings and five model-size configurations.
- `reports/`: original experiment results and later engineering checks in separate folders.
- `requirements/`: the actual working Allegro environment snapshot and dependency-code audit.
- `archive/`: completed pilot, training, setup, and recovery code, preserved verbatim. Not active commands.
- `provenance/`: pre-cleanup source snapshot, original-path map, deployment audit, and exported one-off tool code.
- `data/`, `vendor/`, `checkpoints/`, `logs/`, `tmp/`, and `wheels/`: generated or external files excluded from Git. Large data and LM checkpoints remain on Allegro.

See [reproduction and auditing](docs/REPRODUCIBILITY.md), [sampling](SAMPLING.md), [architectures](MODELS.md), and [historical training readiness](TRAINING_READY.md). The frozen artifact and its manifest were not changed during repository cleanup.
