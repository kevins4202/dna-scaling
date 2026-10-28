# Sequential 50M and 100M runs

Both runs are authorized on physical GPU 7 of Allegro. The queue trains 50M first, verifies its checkpoint, then trains and verifies 100M. After both training runs finish, it evaluates 50M and 100M in order on the original distance experiment. The CNN and previously registered evaluation data stay unchanged. No pretrained billion-parameter model is started by this queue.

## Measured architecture

| Label | Exact parameters | Layers | Width | Attention heads | Head dimension |
|---|---:|---:|---:|---:|---:|
| 50M | 51,927,104 | 24 | 512 | 16 | 32 |
| 100M | 97,352,024 | 24 | 704 | 22 | 32 |

Actual GPU instantiation measured these totals, including filters, convolutions, embeddings, norms and biases. All parameters are trainable. Both retain the original sequence of six short, six medium and ten long Hyena layers plus two attention layers, with short-Hyena MLPs and tied embeddings. Separate smoke/research checkpoint directories prevent accidental warm starts from the integration test.

## Fixed training controls

Both use the same verified 100M-character dataset and tokenized files, data order and seed 6200, 4,096-token context, BF16, microbatch 1, accumulation 8, Adam settings and schedule as 5M/10M/25M. Each trains from scratch for 3,051 steps, processing 99,975,168 tokens. Torch compilation remains disabled for the whole family. Increasing parameters at fixed tokens also increases compute; it does not replicate Evo 2's much larger training history.

Each model must separately pass a 100-step tiny-data smoke test, finite loss, strict checkpoint reload and finite-logit 32-base generation before `register_large_model_readiness.py` adds it to the existing readiness report. The registration checks every existing required hash and exact shared configuration controls; it does not bypass dataset readiness. Previous readiness reports are retained under `provenance/`.

## Queue and recovery

```sh
bash scripts/queue_large_models_gpu7.sh
```

The queue holds both research and distance locks. It waits for GPU 7, then executes smoke/readiness/training/verification for 50M and 100M. Both final research checkpoints must be at step 3051 and pass the existing verification. Training automatically resumes existing valid checkpoints through a load-only configuration overlay. If a final checkpoint already exists, the queue verifies it instead of retraining. Completed research reports are preserved. A running queue must never be duplicated.

- Status: `logs/large-models/status`; orchestration: `logs/large-models/queue.log`.
- Model preparation: `<size>-inspect.log`, `<size>-smoke.log`, `<size>-smoke-verification.log` in that directory.
- Research training: `logs/research/<size>-train.log` and `<size>-verification.log`.
- Checkpoints: `checkpoints/<size>/`, with separate `checkpoints/smoke-<size>/`.
- Training evidence: `reports/research/<size>-complete.json` and `<size>-verification.json`.

A dedicated monitor checks progress, repairs recoverable problems without prompting and preserves valid checkpoints. It never uses another GPU or interrupts unrelated work. Transient SSH outages lead to retries, not blind restarts. Local and remote code must stay consistent.

## Unchanged distance experiment

After training, `scripts/distance/family.py <size>` registers a new `extension-<size>.json` in `data/distance-2048-v1/`. It binds the new checkpoint, measured parameter count, config and all previous score/generation evidence to the original registered manifest, prompts and natural references. The original files are not rewritten.

`generate_distance2048.py` generates 100 continuations per model: 256-base prefixes and 2,048 new bases, batch 20, same per-prompt seeds, temperature 1, ACGT-constrained categorical sampling, no top-k/top-p or repetition penalty. Five dynamic-length probes retain the inherited 0.15 BF16 logit-difference tolerance against padded 4,096-position inference. Failures require diagnosis; the threshold must not be silently relaxed.

The immutable CNN release scores four 512-base windows and the mean-logit sequence aggregate in evaluation mode without gradients. No retraining, checkpoint switch or recalibration is allowed. `analyze_family.py <size>` imports the original AUROC estimator and uses the original 2,000 category-stratified whole-source-group resamples, seed 6220, paired across all sizes and distances. It reports cumulative curves, per-size intervals and descriptive unadjusted pairwise/first-last contrasts. It does not automatically claim significance or fit a scaling law.

Results go to `reports/distance-2048-50m-v1/` and `reports/distance-2048-100m-v1/`, preserving every previous experiment. Final validation requires:

```sh
python scripts/distance/validate_family.py 50m
python scripts/distance/validate_family.py 100m
```

Each must retain `complete.json`, its score report, the curve, both CNN integrity reports and `verification-final.json`. Completion requires 100 unique valid continuations, matching prompt/seed/group metadata, finite scores, matching provenance/checkpoint/output hashes and unchanged CNN files. At the end, export all five training/held-out metrics into a new `reports/research-family/` directory, preserve prior summaries, copy results/data/logs locally and remove the monitor.

## Interpretation limits

The same 100 prompts represent 90 source groups, with only six eukaryotic groups supporting 35% of the weighted statistic. Some source groups appeared in the earlier CNN test. The CNN failed its declared 5M sensitivity gate and remains a qualified diagnostic. Missing/rare categories and uncertain species independence remain unchanged. Neither a weak size trend nor a near-chance score establishes biological function; these are exploratory fixed-token comparisons.

## Verified completion

The full queue completed on October 26, 2026 at 13:59 EDT. Both larger models passed their own 100-step smoke/reload tests, trained for 3,051 steps and 99,975,168 processed tokens, and passed strict final checkpoint reload with finite-logit 32-base generation. Each then generated all 100 matched 2,048-base continuations. Original sampling, batch 20, prompts, CNN and source-group bootstrap were retained. All five dynamic-length probes passed for each model, with maximum absolute logit difference 0.125 against the unchanged 0.15 tolerance. No fallback decoding or relaxed threshold was needed. Generation including setup/probes took 2,278 seconds for 50M and 3,219 seconds for 100M.

The 50M final checkpoint SHA256 is `382b1e31ab1e5e327e2b70a953fa3e366189964b1ee494cbfabbb2425faa6c7d`; 100M is `8093fb8ba9bcb368c3c94884ade1cf228c3b558108a053dfb2ad967ed7762f94`. Both evaluations passed fresh independent final validation after connectivity returned. Local copies of data, generations, evidence and logs were verified against recorded hashes; large checkpoints remain on Allegro. Frozen CNN files and the original evaluation manifest are unchanged.

Aggregate weighted detection AUROC is 0.6472 (95% source-group interval 0.5502–0.7404) for 50M and 0.6850 (95% source-group interval 0.6155–0.7595) for 100M. Every paired aggregate interval across all five sizes includes zero. 50M has the lowest point estimate, but 100M returns close to 5M/10M; there is no established monotonic size benefit. Validation/test losses are 1.238016/1.184360 for 50M and 1.238316/1.184974 for 100M. These differences are small, from one training seed per size, and the CNN limitations remain.

Complete results and exact intervals are in `reports/distance-2048-100m-v1/complete.json` and `RESULTS.md`; earlier results are preserved. All five training metrics and held-out losses are in `reports/research-family/`. The completed monitor is removed after final synchronization.
