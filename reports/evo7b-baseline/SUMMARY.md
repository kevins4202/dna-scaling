# Verified released Evo 2 7B baseline

Completed all 100 registered continuations of 2,048 bases on physical GPU 7. Exact instantiated parameter count: 6,481,059,584. Strict released-checkpoint reload, finite generation/scoring, deterministic smoke replay, immutable inputs, and before/after frozen CNN integrity passed. Generation took 55,569.8 seconds (15.44 hours).

The final validation failure was an ID-ordering bug: the immutable evaluator sorts string IDs lexicographically, while prompts are stored in numeric order. All 100 unique IDs, source groups and categories match. The corrected verifier joins records by ID; the original registered validator, protocol, sequences and score files remain unchanged. `validation-correction.json` records both validator hashes.

## Results

Higher AUROC means generated DNA is easier for the frozen CNN to distinguish from natural DNA.

| Model | Aggregate detection AUROC | 95% interval |
|---|---:|---|
| 5m | 0.6839 | 0.6123 to 0.7454 |
| 10m | 0.6874 | 0.5644 to 0.7560 |
| 25m | 0.6760 | 0.5602 to 0.7512 |
| 50m | 0.6472 | 0.5502 to 0.7404 |
| 100m | 0.6850 | 0.6155 to 0.7595 |
| released_7b | 0.5426 | 0.4306 to 0.6441 |

The released model is harder for this CNN to detect than every small model in the aggregate comparison. Each descriptive paired aggregate interval excludes zero; these intervals are unadjusted and exploratory. The released-minus-100M difference is -0.1424 (95% interval -0.2418 to -0.0727). This supports a coarse distinction between our fixed-budget small models and a mature pretrained reference. It does not establish a size scaling law because pretraining data, training budget, architecture and numerical inference differ.

The largest point-estimate category gap is eukaryotic genic DNA: released AUROC 0.5352 versus 0.8320 for 100M, based on only six source groups. GTDB also differs (0.5087 versus 0.6806). Metagenomes remain near chance for both (0.4824 versus 0.4648), so this category provides little ranking signal from the current CNN. Small category counts and the six eukaryotic groups still limit interpretation.

No clear distance degradation is detected for the released baseline: final-minus-first window difference 0.0022, interval -0.0712 to 0.1039. The frozen CNN has limited validated sensitivity; near-chance detection is not proof of biological functionality. Evaluation-source overlap with released-model pretraining is unknown.

## Computation mode

The recurrent cache exceeded the predeclared probability-difference check. Tolerances were retained, failed evidence archived, and the baseline used full-prefix reference forward for every generated base. BF16 weights, original prefixes, seeds and ACGT conditional categorical sampling were preserved. `protocol.json`, `cache-diagnosis.json`, and docs/RELEASED_BASELINE.md record the change.

## Next project step

Keep the CNN frozen and expand the heldout evaluation with more independently screened source groups, prioritizing eukaryotic coverage. Keep that panel separately registered before scoring. The earlier expanded-panel preparation did not meet screened group targets and is still pending. Do not treat the old panel as an independent confirmation or start more model training automatically.
