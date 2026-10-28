# Frozen CNN v1 results

One 1,227,905-parameter DilatedResNet1D checkpoint is frozen for consistent use across project models. It is a **fixed diagnostic with limited sensitivity**, not a validated sole quality measure or RL reward. No retraining or checkpoint reselection was performed after the fresh test.

## Fresh held-out project test

Both generators use the same 160 natural prompts and 512-base continuations, from 135 source groups not used in classifier training or checkpoint selection. The fresh source pool excludes source rows used by LM training and the previous CNN pilot. All old pilot examples are development/training data only.

The primary score averages category-specific detection AUROCs using the fixed OpenGenome2 mixture weights, renormalized over represented categories. Intervals resample complete source groups within category. Higher AUROC means easier detection; chance is 0.5.

| Generator | Weighted detection AUROC | 95% source-group interval |
|---|---:|---:|
| 5M | 0.5452 | 0.4941 to 0.5982 |
| 10M | 0.5847 | 0.5355 to 0.6353 |

The 5M interval includes chance; 10M detection is weak. These results do not establish an ordering of biological quality. The natural-versus-adjacent-natural control is 0.4885, with interval 0.3861 to 0.5595, consistent with chance.

The prespecified sensitivity gate required both generator intervals to exclude chance. **That gate failed.** The original report and failed-gate status are preserved in `validation.json`. The checkpoint is therefore frozen as a qualified diagnostic, without being relabeled as a validated standalone quality measure. Its weights will stay fixed rather than training repeatedly against these test results. Near-chance detection does not establish biological realism or prove a defective detector.

## Training

The final pool contains 8,860 training window pairs: 8,060 from the external published benchmark and 400 project prompts with one generated example from each model. Sampling gives external and project examples equal expected weight, with equal generator weights and fixed project category weights. Calibration uses 3,368 external pairs and 80 project prompts per generator. No test examples select checkpoints.

One initialization, seed 6211, uses AdamW at learning rate 0.001 and weight decay 0.0001, reverse-complement augmentation, 4096 sampled pairs per epoch, a 40-epoch cap, and patience 8. Calibration selected epoch 2; training stopped after epoch 10. Fresh test results were evaluated only afterward. Exact and reverse-complement overlap and reported >=90%-identity alignments spanning >=256 bases were screened across folds. No project pairs were removed in this final screen.

The external-only development candidate was not adopted because its earlier reused-pilot diagnostic did not establish transfer. It is retained separately under `artifacts/cnn-development-external/`.

## Engineering verification

- Strict checkpoint reload passed; all checkpoint tensors and inference logits were finite.
- Frozen standalone evaluation reproduced both primary AUROCs and intervals exactly.
- Source groups are disjoint across folds, with zero exact/reverse-complement sequence duplicates crossing folds.
- Reverse-complement symmetry and aggregation of four 512-base windows passed.
- The tested CPU/GPU logit difference was zero; the default CPU scoring command was also exercised on the complete 5M test generations.
- Six data/scoring unit tests passed.

Checkpoint SHA-256: `33621e4a90aafbb6c925ff52f7b6c45d74301a66f902cc561a05b83bc3da14b5`.

The artifact manifest, scoring code, architecture, data provenance, and checkpoint are verified by SHA-256 on load. `verification.json` records engineering checks separately from the failed sensitivity gate.

## Permanent use

The fixed artifact is `artifacts/cnn-v1/`, including its standalone scorer, evaluator, and retained reference prompts. `scripts/score_cnn.py` delegates to that artifact. Larger language models use the same weights and scoring rules. No automatic retraining, fine-tuning, recalibration, or replacement is allowed. Any replacement requires explicit user approval, a new version, and rescoring all models being compared.

Source grouping and local alignment screening do not establish species independence. The eligible set omits promoters, organelles in the final test, and short ncRNAs; its eukaryotic source diversity is limited. Category weighting corrects proportions, not missing biological coverage. A 512-base CNN measures local distinguishability, not biological function or correlations across a full long sequence. See `CNN_FROZEN.md` for the complete contract.
