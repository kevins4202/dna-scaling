# Research training readiness

The 100M-character subset remains hash-verified and tokenized on Allegro. All five model sizes completed their smoke tests, controlled research training, strict checkpoint reload and same-prompt distance evaluation. Complete comparisons are in `reports/research-family/` and `reports/distance-2048-100m-v1/`.

| Model | Exact parameters | Layers | Width | Attention heads |
|---|---:|---:|---:|---:|
| 5m | 5,413,780 | 24 | 160 | 5 |
| 10m | 10,331,804 | 24 | 224 | 7 |
| 25m | 24,886,444 | 24 | 352 | 11 |
| 50m | 51,927,104 | 24 | 512 | 16 |
| 100m | 97,352,024 | 24 | 704 | 22 |

All five use six short, six medium and ten long Hyena layers plus two attention layers with short-Hyena MLPs, head dimension32, context4,096, BF16 and no FP8. Each passed its 100-step smoke test and final step3,051 checkpoint reload with finite-logit DNA generation.

## Data and quality checks

- Train: 100,000,000 source characters; validation and test: 1,000,000 each. Source case and ambiguous characters are preserved.
- Exact normalized pretraining category budgets are retained after cleanup.
- Independent reference: 24M characters total, including 8M from 16 disjoint GTDB shards and 2M per other category. Reference and training source coordinates do not overlap.
- Initial eukaryotic GC mismatch was 15.85 percentage points. Sampling across all 40 shards with bounded 4096-character windows reduced it to 0.47 points. Trimer total variation fell from 23.88% to 0.99%.
- All nine categories pass screening thresholds: absolute GC difference <=5 percentage points, trimer total variation <=10%, non-ACGT fraction difference <=2 points, homopolymer >=8 fraction difference <1 point.
- GC window quantiles, document lengths, lowercase fractions and repeat statistics are reported. Document length reflects source stitching/storage, so it is not treated as an organism-level length distribution.
- One original exact evaluation duplicate and high-similarity sequences were found. Evaluation samples were rebuilt from upstream evaluation sources. Candidate screening excluded 89 validation and 197 test candidates for similarity or duplicate content.
- Final exact-record duplicate screen (>=128 characters): zero. Final minimap2 asm10 primary-alignment checks: zero reported matches for validation/train, test/train and test/validation. This is not a proof of phylogenetic independence.

## Training budget

Each model uses one GPU, microbatch 1, accumulation 8 and 3,051 steps: 99,975,168 processed tokens. This matches budgets between models. Preprocessing adds one EOD token per document:
- train: 100,029,045 encoded dataset tokens.
- valid: 1,000,319 encoded dataset tokens.
- test: 1,000,315 encoded dataset tokens.

## Launch on Allegro

```bash
cd /home/ksong37/cis6200
source scripts/gpu7_env.sh
bash scripts/run_model.sh 5m train
# After the first run finishes:
bash scripts/run_model.sh 10m train
```

The launcher uses configs/data100m.yml for full training and keeps smoke-test data separate. A readiness guard verifies source/config/data hashes before starting. Do not launch both jobs simultaneously on GPU 7.

## Limits

This is an efficient, bounded-source approximation to the OpenGenome2 pretraining mixture. Matching the checked properties does not establish equal full-population or species distributions. Species metadata was inspected, but many processed records lack usable identifiers and contig IDs do not directly match assembly/taxonomy IDs. The independent reference is disjoint in source coordinates, but may contain related organisms. Screening thresholds are engineering rules, not formal statistical equivalence tests.

Machine-readable evidence is in reports/training-readiness.json, distribution-audit.json, taxonomy-coverage.json and overlap/.
