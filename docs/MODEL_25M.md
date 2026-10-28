# 25M training and fixed-CNN distance evaluation

This is the third controlled model size, authorized after the 5M/10M distance experiment. It is an exploratory extension of the same evaluation, with the original prompts, natural references, CNN and 5M/10M evidence unchanged.

## Model and training controls

Measured with the pinned Savanna model builder on physical GPU 7: **24,886,444 parameters**, all trainable. The model has 24 layers, width 352, 11 attention heads of dimension 32, and the existing schedule of six short Hyena, six medium Hyena, ten long Hyena and two attention layers. It retains the original short-Hyena MLP, tied embeddings, RMSNorm and 4,096-token context. See `configs/models/25m.yml` and `reports/25m-parameters.json`.

The full run starts from scratch in `checkpoints/25m/`, separate from the 100-step integration checkpoint in `checkpoints/smoke-25m/`. It uses the same verified bounded 100M-character dataset and tokenized files, data order, seed 6200, Adam settings, BF16 precision, microbatch 1, accumulation 8 and 3,051-step budget as the smaller models: **99,975,168 processed training tokens**. Torch compilation stays disabled, as in both completed runs. The token count is fixed; the computation and runtime increase with model size.

Before training, the queue requires the 100-step smoke run with finite loss, strict checkpoint reload, the correct measured parameter count and finite-logit 32-base generation. `register_25m_readiness.py` checks all existing readiness hashes and exact shared configuration controls before extending readiness to 25M. It preserves a copy of the earlier readiness report. It does not enable 50M/100M.

## Execution and recovery

```sh
bash scripts/queue_25m_project_gpu7.sh
```

The queue protects both existing research and distance locks, waits for physical GPU 7, verifies the frozen CNN, passes smoke/readiness, trains, verifies the final checkpoint, registers the evaluation extension, generates, scores and analyzes. Do not start a second queue while either lock is held. `scripts/run_model.sh 25m train` resumes existing valid training checkpoints through a load-only configuration overlay. The full dataset readiness guard remains active.

- Status and orchestration: `logs/25m-project/status` and `queue.log`.
- Model preparation: `inspect.log`, `smoke.log` and `smoke-verification.log` in that directory.
- Research training: `logs/research/25m-train.log` and `25m-verification.log`.
- Training evidence: `reports/research/25m-complete.json` and `25m-verification.json`.

## Evaluation

After the final training checkpoint is verified, `scripts/distance/register_25m.py` creates `data/distance-2048-v1/extension-25m.json`, binding the 25M checkpoint, configuration and original registered evaluation inputs by hash. It does not rewrite the original manifest. `generate_distance2048.py` uses the same 100 prompts, 256-base prefixes, 2,048-base continuations, batch 20, per-prompt seeds and ACGT-constrained temperature 1 sampling as the smaller models. Five dynamic-length inference probes must agree with the original padded context within the existing BF16 tolerance.

The original frozen CNN scores the four 512-base windows and mean-logit sequence aggregate. Its architecture, checkpoint, scoring implementation and calibration stay unchanged. `analyze_25m.py` imports the existing bootstrap estimator, retains the original category weights and seed 6220, and uses the same 2,000 source-group resamples paired across all models and distances. New descriptive contrasts are 25M-minus-5M, 25M-minus-10M and last-minus-first for 25M. These are unadjusted intervals, with no confirmatory significance claim.

Results go to **`reports/distance-2048-25m-v1/`**, preserving `reports/distance-2048-v1/`. Completion requires `complete.json`, `25m-scores.json`, `distance-curve.png`, both CNN integrity reports, and `verification-final.json` produced by:

```sh
python scripts/distance/validate_25m.py
```

The registered set has 100 prompts from 90 source groups, with only six eukaryotic groups carrying 35% of the weighted statistic. Some groups appeared in the earlier CNN test. The CNN failed its declared 5M sensitivity gate and remains a qualified diagnostic; detectability does not establish biological function. These limitations continue to apply to 25M, regardless of whether the result is better, worse or tied.

A dedicated five-minute heartbeat monitor checks failures, repairs and resumes safely, preserves valid checkpoints and synchronizes code between the local project and Allegro. It reports only completion or an actionable blocker, then deletes itself. Once verified, training/readiness/evaluation reports, extension manifest, generated sequences and logs are copied locally. Checkpoints remain on Allegro with recorded hashes. All issued tool code is retained in the provenance export.

## Verified completion

The detached GPU 7 queue finished at 00:18 EDT on October 26, 2026, while local SSH access was unavailable. A fresh independent validation after connectivity returned passed again. Full training reached step 3,051 with 99,975,168 processed tokens, followed by strict checkpoint reload, finite logits and 32-base generation. The final checkpoint SHA256 is `e1c94161638236fd87268266eb44e3dd161ed08530da94de7244aa54ab8f360b`.

All 100 matched 2,048-base continuations completed. Each of the five dynamic-length probes passed the inherited 0.15 logit-difference tolerance (observed maximum 0.125). Generation took 1,579 seconds including setup and probes. The original CNN checkpoint and implementation hashes passed before and after scoring. The original 5M/10M evaluation and registered prompts remained unchanged.

Aggregate detection AUROC is 0.676 (95% source-group interval 0.560–0.751). The paired differences are −0.0079 versus 5M (−0.0589–0.0292) and −0.0114 versus 10M (−0.0377–0.0209). Every per-window paired difference interval also includes zero. The point estimates are slightly lower for 25M, but these data do not establish a size advantage or a scaling law. The first-to-last increase is 0.0689 (−0.0168–0.1286), so a consistent distance trend is also unresolved. Preserve the CNN sensitivity and limited-source-group qualifications.

Results, uncertainty, plot and copied verification are in `reports/distance-2048-25m-v1/`; training metrics and held-out losses are exported separately to `reports/research-25m/` to preserve the original two-model report. Inputs, generation and logs are saved locally and on Allegro; large model checkpoints remain on Allegro with recorded hashes. The monitor is removed after final copies are verified. No 50M/100M run was started.
