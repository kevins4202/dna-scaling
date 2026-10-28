# 2,048-base distance experiment v1

This experiment uses the existing step-3051 5M and 10M checkpoints. It never trains a language model or changes the frozen CNN. Run both models sequentially on physical GPU 7:

```sh
bash scripts/queue_distance2048_gpu7.sh
```

The queue takes a nonblocking file lock, waits for GPU 7, verifies the registered inputs, generates and scores 5M then 10M, and runs the paired analysis. It resumes already completed generation records. A final interrupted JSONL record is repaired; other malformed or duplicate records fail. Only restart after checking the queue lock and live processes. An existing completion report alone is not a substitute for the final validation below.

## Data and fixed protocol

`data/distance-2048-v1/manifest.json` is registered before generation/scoring. It binds the prompts, matched natural continuations, both LM checkpoints and frozen CNN to SHA256 hashes. The 100 prompts have 256 bases, followed by 2,048 contiguous uppercase ACGT natural bases. Sources used for LM training or CNN training/calibration are excluded. Minimap2 screens against all CNN development natural/generated/null windows, rejecting alignment spans of at least 256 bases with at least 90% identity. The separate validator also checks exact/reverse-complement matches, duplicate chunks and overlapping windows within source groups.

The set contains 90 source groups. Eukaryotic sequences have only 16 windows from six groups, so that category has limited effective sample size. Organelle sequences receive no slot under the fixed 100-prompt quota rule; promoters cannot provide 2,304 bases. The represented original mixture weights total 100 out of 100.52. The fixed evaluator renormalizes weights across represented categories. Fourteen source groups were previously used for CNN testing: this is an exploratory distance analysis on an existing held-out reference pool, not a newly untouched benchmark.

Generation uses BF16, uncached inference, temperature 1 and ACGT-constrained categorical sampling, without top-k/top-p or repetition penalty. Seed is 621900 plus registered prompt id, matched between sizes. Dynamic inference is compared with 4,096-position padded inference at lengths 256, 768, 1,280, 1,792 and 2,303, with the inherited BF16 maximum absolute logit tolerance of 0.15. Batch size defaults to 20 and is recorded. Successful generation requires strict checkpoint reload, the expected exact parameter count, finite logits at every step, and 100 valid sequences per model.

## Frozen scoring and uncertainty

The queue invokes the immutable CNN scorer, in evaluation mode without gradients. It scores four consecutive 512-base windows, excluding the prompt. Each window averages forward and reverse-complement logits. Sequence aggregation averages the four logits, before any sigmoid. The CNN checkpoint and every frozen implementation file must pass their original hash checks.

The primary statistic is the original mixture-weighted within-category natural-versus-generated detection AUROC. Higher means easier detection. The analysis uses 2,000 category-stratified bootstrap resamples of complete source groups, with identical resamples for both models and all distances. It reports four distance intervals, the sequence aggregate, paired 10M-minus-5M differences and last-minus-first differences. These are descriptive, unadjusted 95% percentile intervals; there is no confirmatory multiple-testing claim. The vectorized bootstrap metric is checked against the frozen evaluator's point estimates and a direct replicated-observation unit test.

The original CNN failed its declared 5M sensitivity gate. It stays frozen as a limited distinguishability diagnostic; chance AUROC is not evidence of biological function.

## Verification and outputs

```sh
python scripts/distance/validate.py
python scripts/test_distance2048.py
# After queue status is VERIFIED_COMPLETE:
python scripts/distance/validate.py --complete
```

- `scripts/distance/prepare.py`: deterministic data registration and similarity screen.
- `scripts/generate_distance2048.py`: resumable generation through the pinned Savanna launcher.
- `scripts/distance/analyze.py`: paired bootstrap, results table and curve.
- `scripts/distance/validate.py`: independent data and completion checks.
- `logs/distance-2048-v1/`: queue status and generation/scoring/analysis logs.
- `reports/distance-2048-v1/`: both score reports, before/after CNN integrity checks, final validation, `complete.json`, `RESULTS.md` and `distance-curve.png`.

The generation and scoring run on Allegro. Registered data, generated sequences, reports and source code are copied to the local project. Large LM checkpoints remain on Allegro and are bound by their hashes. The dedicated monitor retries transient SSH failures, avoids duplicate queues, repairs code/environment issues and checks final evidence before announcing completion.

## Verified completion

Both models finished on physical GPU 7 and passed `validate.py --complete`. Each produced 100 continuations of 2,048 bases. Strict LM checkpoint reload, finite generation/scoring, five dynamic-length inference checks, immutable CNN hashes and copied result hashes all passed. Exact sizes remain 5,413,780 and 10,331,804 parameters. Generation took 821 seconds for 5M and 1,018 seconds for 10M; these invocation times exclude scoring and include model setup and probes. No repair or sampling change was required during generation.

Aggregate fixed-CNN detection AUROC was 0.6839 (95% group-bootstrap interval 0.6123–0.7454) for 5M and 0.6874 (0.5644–0.7560) for 10M. Their paired difference, 10M minus 5M, was 0.0035 (−0.0533–0.0381). Every per-distance paired model-difference interval also includes zero. There is no clear size advantage here.

Detection rises after the second window, but the curve is nonmonotonic. Last-minus-first detection increased by 0.0581 for 5M (−0.0769–0.1488) and 0.0743 for 10M (0.0075–0.1667). The latter is a descriptive, unadjusted contrast with only six eukaryotic groups supporting 35% of the weighted statistic; it does not establish a general distance law or biological decline. Preserve the qualified scope of the frozen CNN.

Results and source-group intervals are in `reports/distance-2048-v1/RESULTS.md`, full paired contrasts in `complete.json`, plot in `distance-curve.png`, and independent final evidence in `verification-final.json`. Copies of registered inputs, generated sequences, reports and logs are retained locally; immutable LM checkpoints remain on Allegro with hashes recorded in the manifest.
