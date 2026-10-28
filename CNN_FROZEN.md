# Fixed project CNN v1

Use one checkpoint and one scoring procedure for every language model in this project. This is a fixed diagnostic with limited sensitivity, not a validated standalone quality measure. The earlier pilot ensemble and external-only development candidate are historical diagnostics, not the project evaluator. Final validation and independent verification are in `reports/cnn-final/`.

## Architecture

The architecture is the published DilatedResNet1D from Synthetic Genomes Benchmark commit `2e2ab3808f4f55db1fbcf22df39610a17f4da1dc`, with **1,227,905 parameters**. Four one-hot A/C/G/T channels enter a 64-channel convolutional stem, four residual blocks with dilations 1/2/4/8 and channels 64/128/128/256, global average pooling, and a binary head. Dropout and batch normalization are used during training; inference always uses evaluation mode and fixed running statistics.

## Data and final validation

The training pool combines the published natural/generated benchmark at https://zenodo.org/records/18826945 with matched outputs from both of our completed 5M and 10M models. Training gives 50% expected weight to the external benchmark and 50% to our project examples. Generator sizes receive equal weight. Project categories use the original OpenGenome2 mixture weights rather than raw source-row counts; external tags have equal expected weight.

The external archive's MD5 is `353488a6529ce72f8d0f435d5d761e1d`. Conditioning prefixes are excluded using generation metadata and a common-prefix check, with a conservative minimum exclusion of 4096 bases. Multi-contig pairs without unambiguous contig pairing are excluded. Original and generated windows use matched positions. All windows are exactly 512 ACGT bases. Entire external taxonomic tags stay within their original development split.

An external-only candidate detected the published benchmark but did not establish transfer to our generators on the already-examined pilot. It is retained under `artifacts/cnn-development-external/`; it was not adopted as the final evaluator. Its test results are development evidence, not the final project's blind test.

Fresh project examples come from `data/reference-independent`. Source rows used by LM training or the earlier CNN pilot are excluded. Before similarity screening, the new pool contains 300 training windows from 218 source groups, 80 calibration windows from 58 source groups, and 160 final-test windows from 135 source groups. Every source group belongs to one fold. All 100 old pilot examples are development/training data only. Fresh prompts receive matched 512-base continuations from each generator using the same 256-base natural prefixes, temperature 1, ACGT-constrained categorical sampling, and matched per-prompt seeds.

Exact and reverse-complement overlap is screened across folds, followed by minimap2 screening for reported alignments of at least 256 bases and at least 90% identity. Affected pairs are removed. The two generators retain the same surviving project prompts. Final retained counts and exclusions are in `data/cnn-final-mixed/manifest.json` and the release's `data-manifest.json`.

The final training configuration is fixed: seed 6211, AdamW, learning rate 0.001, weight decay 0.0001, matched balanced classes, reverse-complement augmentation, 4096 sampled pairs per epoch, a 40-epoch cap, and patience 8. Checkpoint selection uses calibration data only: 75% weight to the average of the two project generators' mixture-weighted category AUROCs and 25% to external calibration macro AUROC. No hyperparameter or seed search follows the fresh project test.

The prespecified sensitivity gate required both project test AUROC intervals to exclude chance in the detection direction. That gate was not met: 5M AUROC is 0.5452 (95% interval 0.4941 to 0.5982); 10M is 0.5847 (0.5355 to 0.6353). The adjacent-natural control is 0.4885 (0.3861 to 0.5595), consistent with chance. The original failed-gate report is preserved unchanged. At the user's request to keep one CNN, the calibration-selected checkpoint is frozen as a diagnostic, with the failed sensitivity gate explicitly recorded. It is not certified as a sole project quality measure or RL reward. No retraining or checkpoint reselection follows the held-out results. Near-chance detection does not establish biological realism or prove a defective detector. Frozen CLI scoring is checked separately for checkpoint reload, CPU/GPU agreement, strand invariance, and multi-window aggregation.

## Permanent scoring contract

- Inputs contain continuations only. Conditioning prompts are separate fields and are never scored.
- Input length is a positive multiple of 512, using uppercase ACGT only. Invalid or partial windows are rejected, never padded or silently replaced.
- Each window receives the mean of forward and reverse-complement logits. A sequence logit is the arithmetic mean of its window logits. Synthetic score is the sigmoid of that mean; natural score is one minus synthetic score. These are classifier scores, not calibrated biological probabilities.
- The primary generator metric is category-specific natural-versus-generated AUROC, averaged with the original fixed mixture weights and renormalized over represented categories. Higher AUROC means easier detection; chance is 0.5. Raw pooled AUROC and per-category results are also reported.
- Confidence intervals resample complete source groups within category, preserving fixed category weights. Missing categories are explicitly reported. Weighting corrects sample proportions, not missing biological coverage or limited source diversity.
- Use the same prompts, natural reference, continuation length, generation settings, and CNN version when comparing models. The release includes `evaluation-prompts.jsonl` for the retained final 512-base test.
- For later 2048-base continuations, use four unchanged 512-base windows and report their aggregate and positions relative to the prompt. New matched long natural references can be registered without updating CNN weights. Local CNN scores do not directly measure correlations across the full 2048 bases.

```bash
python scripts/score_cnn.py --input generated.jsonl --output scores.csv --device cpu
python scripts/evaluate_project_cnn.py --natural artifacts/cnn-v1/evaluation-prompts.jsonl --generated matched-test-generations.jsonl --output evaluation.json --device cpu
```

Evaluation inputs must contain the same unique `id` values. Natural records need `text` or `natural`, `category`, and an actual dependency `group_id`. Generated records need `generated` or `text`. Both sides must have matching continuation lengths. The evaluator rejects missing, extra, or duplicate ids. A different id for each correlated window does not make those windows independent.

## Freeze policy

`artifacts/cnn-v1/` contains the checkpoint, architecture, standalone scoring and evaluation tools, fixed reference prompts, data provenance, protocol, validation report, and SHA-256 manifest. Loading verifies the artifact hashes. The project scoring and evaluation implementations must match the released code. Score exports record model and input hashes. Training refuses to overwrite an existing final selected checkpoint or release.

Do not retrain, fine-tune, recalibrate, change the checkpoint, or alter these scoring rules when 25M/50M/100M models are added. Score them with the same v1 weights. Any future replacement requires explicit user approval, a new version, and rescoring every model being compared. Report weak detection or domain mismatch instead of silently modifying the evaluator.

The source grouping and local sequence screening do not prove species or phylogenetic independence. Eligibility excludes short promoters and ncRNAs, and the test has limited eukaryotic source diversity. The fixed detector is an operational distinguishability benchmark, not a guarantee that all future defects will be detected or a measure of biological function. If it is later used as an RL reward, improving that reward alone is not independent evidence of biological improvement.
