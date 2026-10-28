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

## What we trained and evaluated

Our DNA language models learn next-base prediction: given preceding real DNA during training, predict the next letter. During free generation, each sampled letter is fed back into the model before it predicts the next one. Here, free generation is conditioned on a natural prefix; it is not generation from an empty prompt.

The completed comparison gives every generator the same 100 natural prefixes of 256 bases and samples 2,048 new bases per prefix. Natural references are the actual next 2,048 bases from those sources. All six use A/C/G/T-conditional categorical sampling at temperature 1, without top-k or top-p truncation. The original project CNN scores four 512-base windows; the reproduced-CNN reassessment scores two 1,024-base windows. Both exclude prompt text. These local judgments do not directly test relationships across a whole genome.

## Architectures

### DNA language models

All five small models use the same 24-layer StripedHyena 2 operator schedule: six short-Hyena layers, six medium-Hyena layers, ten long-Hyena layers, and two causal self-attention layers. Hyena layers mix information across sequence positions using learned filters; attention layers make content-dependent comparisons between positions. Width is the number of features used to represent each token. The small family uses RMSNorm and short-Hyena MLP modules.

| Model | Exact parameters | Layers | Width | Attention heads |
|---|---:|---:|---:|---:|
| 5M | 5,413,780 | 24 | 160 | 5 |
| 10M | 10,331,804 | 24 | 224 | 7 |
| 25M | 24,886,444 | 24 | 352 | 11 |
| 50M | 51,927,104 | 24 | 512 | 16 |
| 100M | 97,352,024 | 24 | 704 | 22 |
| Released `evo2_7b_base` | 6,481,059,584 | 32 | 4,096 | 32 |

The released base checkpoint has nine short-, nine medium-, nine long-Hyena layers and five attention layers, with an inner MLP width of 11,008. It belongs to the same hybrid architecture family but is not identical to our small models apart from size. Its weights were downloaded, strictly loaded, and used without fine-tuning. It is the base checkpoint, distinct from the later long-context `evo2_7b` checkpoint. See [architectures](MODELS.md) and [released baseline protocol](docs/RELEASED_BASELINE.md).

Each small model trained from scratch on the same approximately 100M-letter OpenGenome2 subset for 3,051 updates and 99,975,168 processed tokens. Shared settings: 4,096-base training context, effective batch of eight sequences, Adam with learning rate 0.0006 decaying to 0.00006, weight decay 0.1, BF16, and one training seed. Only width and corresponding head/group counts vary within the small family. Matching selected category/composition statistics does not reproduce the released model's training diversity, exposure or schedule.

### CNN discriminators

Both use four-channel one-hot A/C/G/T input, a convolutional stem, four residual blocks, global average pooling over sequence positions, and a two-layer classification head. Each residual block contains two convolutions with batch normalization and ReLU, plus a skip connection. Stem kernel size is 15; residual kernel size is 7; blocks 2 and 4 downsample by a factor of two. The head uses ReLU and dropout 0.2, then outputs one synthetic-DNA logit. This is binary classification: natural DNA is class 0 and generated DNA is class 1. A sigmoid maps the logit to a 0–1 synthetic-class score; a threshold can produce a hard label. AUROC uses continuous scores across examples rather than a single threshold.

| Setting | Original frozen project CNN | Each reproduced CNN |
|---|---|---|
| Parameters | 1,227,905 | 2,757,313 |
| Evaluation chunk length | 512 bases | 1,024 bases |
| Stem channels | 64 | 96 |
| Residual-block output channels | 64, 128, 128, 256 | 96, 192, 192, 384 |
| Block dilations | 1, 2, 4, 8 | 1, 2, 2, 4 |
| Head | 256 → 128 → 1 | 384 → 192 → 1 |

The project CNN was trained on external benchmark examples plus outputs from our 5M/10M models, using AdamW at learning rate 0.001, reverse-complement augmentation, and calibration-based early stopping. Calibration selected epoch 2. Its fresh-test sensitivity requirement failed for 5M; it remains a qualified diagnostic.

The reproduction used the original benchmark only, with natural labeled 0 and synthetic labeled 1. Within each domain, one species/taxonomic tag was held out for testing and the other tags supplied fit/calibration data. Each training example averaged logits from four random 1,024-base chunks of its sequence, trained with binary cross-entropy, Adam at learning rate 0.0003, batch 8, weight decay 0.0001, and 10 epochs. The highest calibration-AUROC checkpoint was selected without test-based selection. The original item-level calibration split was retained and audited; natural/generated counterparts can cross fit/calibration, while test tags remain excluded from both.

The original procedure trains multiple detectors, not one universal checkpoint. Our six-model reassessment fixes the 30 detectors from its first 32-chunk evaluation configuration, averages logits within each domain, and gives the eukaryotic/prokaryotic/viral domains equal total weight. Every generator receives exactly the same ensemble. No detector is retrained, recalibrated, or selected using these reassessment scores. This ensemble has not been independently validated as a universal quality measure. See [CNN reproduction](docs/CNN_REPRODUCTION.md) and [reassessment protocol](docs/CNN_REASSESSMENT.md).

## Results

### Next-base prediction on held-out natural DNA

Lower test loss is better. These single-seed differences are small; no matched next-base test loss has been reported for the released reference.

| Small model | Test loss |
|---|---:|
| 5M | 1.187362 |
| 10M | 1.187922 |
| 25M | 1.185391 |
| 50M | 1.184360 |
| 100M | 1.184974 |

### CNN detection of our saved free generations

**AUROC measures detection, not biological quality.** An AUROC of 0.5 means the fixed detector supplies no useful natural-versus-generated ranking on this panel. It can reflect convincing outputs, missed defects, domain mismatch, or insufficient evidence. Below-chance AUROC can indicate reversed score ordering and must not be interpreted as exceptionally good generation.

All six models use the same 100 prompts from 90 source groups. The reassessment screened natural prefixes/references and all 600 generated continuations against the complete original CNN corpus and the known small-model training text. No reported alignment met the registered threshold of at least 256 query bases at at least 90% identity, and no source group was removed. This is no detected overlap at stated thresholds, not proof of species independence. Released 7B pretraining overlap remains unknown.

| Generator | Original project CNN AUROC | Reproduced ensemble AUROC | Reproduced ensemble 95% interval |
|---|---:|---:|---|
| 5M | 0.6839 | 0.5560 | 0.4912–0.6413 |
| 10M | 0.6874 | 0.6030 | 0.5336–0.6808 |
| 25M | 0.6760 | 0.5630 | 0.4883–0.6215 |
| 50M | 0.6472 | 0.5306 | 0.4451–0.6230 |
| 100M | 0.6850 | 0.5080 | 0.4406–0.6016 |
| Released 7B base | 0.5426 | 0.5837 | 0.5244–0.6638 |

The intervals use 2,000 category-stratified whole-source-group bootstrap resamples, paired across models, seed 6220. They are descriptive and unadjusted for multiple comparisons. The 10M and released-7B aggregate intervals exclude 0.5, so the results are not all indistinguishable from chance. Several paired contrasts also exclude zero, but there is no monotonic size trend across all five small models.

**The ranking changes with the evaluator.** The released reference's advantage under the original CNN does not persist under the reproduced ensemble. Its paired AUROC difference versus 100M is +0.0757, with interval +0.0046 to +0.1420. This does not establish that our 100M model produces biologically better DNA. Some secondary domain-trained detectors reverse score ordering on this mixed panel, and only six eukaryotic groups support 35% of the weighted statistic. Evaluator transfer and independent structure measurements remain unresolved.

Full evidence: [reassessment results](reports/cnn-reassessment-v1/SUMMARY.md), [all paired/category/domain estimates](reports/cnn-reassessment-v1/analysis.json), [overlap audit](reports/cnn-reassessment-v1/overlap-audit.json), and [independent verification](reports/cnn-reassessment-v1/verification-final.json).

### Reproduction on the authors' original benchmark

All 210 registered folds completed and passed independent verification. The following are domain means across held-out tags on the original benchmark, not our project mixture-weighted scores.

| Original benchmark domain | One 1,024-base chunk | 32 chunks averaged |
|---|---:|---:|
| Eukaryotic | 0.8744 | 0.9804 |
| Prokaryotic | 0.6805 | 0.7962 |
| Viral, including the MegaDNA tag | 0.5877 | 0.7512 |

This recovers strong original-benchmark detection, especially for eukaryotes, but does not establish transfer to our short continuations. The benchmark contains much longer sequences, typically with a copied 3,000-base conditioning prefix; near-chance detection inside that prefix is not evidence about new DNA. Missing original commands/RNG states prevent an exact published-number replication claim. Each chunk-count configuration has a separately trained detector, so this table is not a fixed-checkpoint causal ablation of aggregation. [Reproduction report](reports/cnn-original-v1/SUMMARY.md).

## How free generation should be evaluated next

We have already evaluated free generations through the CNNs. A fuller evaluation should keep the same prompts, lengths and sampling settings, and add independent measurements on the saved outputs:

1. **Composition:** GC content, short-pattern (k-mer) distributions, and low-complexity or repeated-letter runs, compared with matched natural continuations.
2. **Structure:** repeat lengths and spacing, measured in equal-length windows at increasing distance from the prompt. A 2,048-base continuation cannot support genome-scale structure claims.
3. **Natural variability:** compare generated-versus-natural differences with differences between separate natural windows from matched categories/sources.
4. **Sensitivity controls:** test natural-versus-natural examples and controlled corruptions at the exact evaluation length, checking which defects each fixed metric detects. Keep development/control-design sources separate from final test sources.
5. **Uncertainty:** report categories, paired source-group intervals and, where feasible, repeated generation seeds. Evaluate prediction loss separately from generated-sequence properties.

Composition/structure generation measurements and the proposed structure-retention score are **not yet completed**. They must be specified before inspecting their model rankings. We should not retrain models, choose a detector, or reverse score direction merely to obtain a desired ranking.

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
