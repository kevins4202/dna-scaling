# Verified six-generator reassessment

All six generators were reevaluated with the same fixed domain-balanced ensemble of 30 reproduced CNN checkpoints. No language model or detector was retrained, and all original generations and prior results remain unchanged.

## Fairness and overlap checks

All 100 original prompts from 90 source groups passed the registered screening rules; no IDs were removed or substituted. The screen covered each natural 256-base prefix plus its 2048-base continuation and all 600 saved generated continuations. Targets were the full original CNN benchmark corpus, including heldout examples as well as training/calibration examples, and the known training text shared by the five small language models. Both minimap2 passes (asm10 and sensitive sr k11/w5) reported zero matches meeting the declared threshold of at least256 query bases at at least90% identity. Exact whole-record/reverse-complement equality and small-LM source-row exclusions were checked too. The screening FASTA was independently verified to match all 29,045 registered small-LM training text records.

This establishes no detected sequence overlap at the stated thresholds against the available known training files. It does not establish species/phylogenetic independence or exclude overlap with the released 7B model's much larger pretraining corpus.

## Results

Higher AUROC means higher scores assigned to generated than natural DNA. Chance is 0.5. Below-chance values in secondary diagnostics indicate reversed score ordering, not superior biological quality. These two-window transfer scores must not be equated with the original long-sequence 32-chunk benchmark.

| Generator | Reproduced ensemble | 95% group-bootstrap interval | Original CNN, identical IDs |
|---|---:|---|---:|
| 5m | 0.5560 | 0.4912 to 0.6413 | 0.6839 |
| 10m | 0.6030 | 0.5336 to 0.6808 | 0.6874 |
| 25m | 0.5630 | 0.4883 to 0.6215 | 0.6760 |
| 50m | 0.5306 | 0.4451 to 0.6230 | 0.6472 |
| 100m | 0.5080 | 0.4406 to 0.6016 | 0.6850 |
| released_7b | 0.5837 | 0.5244 to 0.6638 | 0.5426 |

## Interpretation

The ranking is evaluator-sensitive. Under the reproduced ensemble, 100M has the lowest point AUROC and 10M the highest among our small models. This is not a monotonic size trend across all five. The released 7B reference no longer has the lowest detection score. Its paired aggregate difference versus 100M is +0.0757 (95% interval +0.0046 to +0.1420). Several small-model paired contrasts also exclude zero, but these are exploratory unadjusted intervals on a reused panel, not a validated biological-quality ranking.

The original benchmark reproduction recovered strong detection on its own long sequences. Transferring those detectors to our short mixed-source continuations produces much weaker separation and disagrees with our original project CNN. The viral-trained secondary ensemble averages below 0.5 on the mixed panel for our small models, showing score-direction transfer problems. The domain-balanced ensemble itself was fixed before inference but has not been independently validated as a universal quality evaluator. These findings argue against treating either detector ranking as a stable measure of biological realism.

Only six eukaryotic source groups still carry 35% of the category-weighted statistic; rare categories and one training/generation seed add uncertainty. Released 7B differs in pretraining history and used the previously documented full-prefix generation fallback. We preserved those saved outputs and did not alter decoding.

## Scoring and verification

Each continuation contributed exactly two nonoverlapping 1024-base windows, excluding its prompt. Checkpoints were strictly loaded, evaluated in FP32 without augmentation/recalibration, and retained unchanged. Equal total weight went to the eukaryotic, prokaryotic and viral detector domains. All six models used identical weights, IDs, window positions and category weighting. We report 2000 paired category-stratified whole-source-group bootstrap resamples, seed 6220. Independent validation recomputed the ensemble from raw per-detector logits and point AUROCs with sklearn, verified the overlap exclusions and all registered source/input/checkpoint hashes.

## Next step

Preserve all results. Before using CNN scores for a scaling or RL claim, separately register a transfer/sensitivity study with independent sources and matched window positions, and include an independent sequence-based diagnostic. Do not select a detector or flip score directions based on which result favors a generator. This reassessment does not by itself justify retraining the language models.

Evidence: design.json,protocol.json,overlap-audit.json,lm-training-corpus-verification.json,scores.json,detector-scores/,analysis.json,comparison.png,verification-final.json and complete.json.
