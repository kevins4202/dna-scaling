# Original benchmark CNN reproduction: verified results

All 210 registered folds completed: 180 chunk-count folds across 30 heldout taxonomic tags and 30 distance folds. Independent verification strictly reloaded every checkpoint, checked all registered input and frozen-project-CNN hashes, reproduced every saved test AUROC, and recomputed aggregate means. The full original archive contains 1,358 natural/generated pairs. Total recorded training/evaluation time across folds was 5,126 seconds (85.4 minutes), excluding orchestration/loading overhead.

## Main result

Higher detection AUROC means easier distinction between natural and synthetic DNA. These are unweighted means across the heldout tags within each original domain, not our project's mixture-weighted statistic.

| Domain | One 1024-base chunk | Mean over 32 chunks |
|---|---:|---:|
| Eukaryotic | 0.8744 | 0.9804 |
| Prokaryotic | 0.6805 | 0.7962 |
| Viral, including MegaDNA benchmark tag | 0.5877 | 0.7512 |

The released procedure recovers strong detection on the original benchmark, especially for eukaryotes. This supports its qualitative result; missing original commands, RNG states and weights prevent an exact numerical replication claim. Each chunk-count configuration has its own trained/calibration-selected detector as in upstream, so this sweep is not a fixed-checkpoint causal ablation of chunk count.

## Distance and conditioning-prefix audit

At absolute sequence-start positions 0, 256 and 2048, mean eukaryotic AUROC is approximately 0.50. The independent prefix audit finds a median exact shared prefix of 3,000 bases in every domain. These near-chance windows predominantly lie inside the copied conditioning prompt; they are not evidence of high-quality newly generated DNA.

For eukaryotes, detection AUROC is 0.7535 at start 4,096, 0.7990 at 20,000, 0.9016 at 100,000, and 0.9378 at 290,000. This shows an increasing descriptive distance trend in this benchmark. Prokaryotic detection rises more modestly, from 0.5243 at 4,096 to 0.6786 at 290,000. Viral results vary; two later positions represent only five of six tags because short sequences lack eligible windows. The exact grid was registered by us because the released launcher requires it but does not supply the original command. Positions are from sequence start, not prompt end.

## Why comparison with our earlier experiment differs

Our project CNN has 1.23M parameters and 512-base inputs; this released-launcher CNN has 2.76M parameters and 1024-base inputs. The original method trains separately within each domain, holding one tag out. Our method trains one mixed detector using both benchmark and project-generator examples, with different optimization and data weighting. The original benchmark also samples across much longer generated sequences and averages up to 32 chunks, whereas our experiment scores only the first 2,048 newly generated bases. Generator checkpoint variants, prompts, biological coverage and sampling settings also differ.

We therefore cannot attribute the entire gap to our detector or claim our 7B near-chance result proves realism. The reproduction demonstrates that the original setup detects its own benchmark's synthetic artifacts; it does not yet demonstrate transfer to our short continuation task. Our five small models do not need automatic retraining based on this result.

## Limitations and next step

For faithful procedural reproduction, original FASTA concatenation, ambiguous-base handling, prompt inclusion and item-level fit/calibration splitting were retained. The latter can place natural/generated counterparts of a source pair on opposite sides of fit and calibration; this is recorded in each split audit. Heldout test tags are excluded from both fit and calibration. These methodological limitations remain and should not be described as independent validation of biological function.

Next, register matched inference comparisons using the saved heldout checkpoints and sources to isolate evaluation position and amount of sequence observed. Comparisons involving a new evaluation panel must respect training-source exclusions and use the same fixed evaluation rules for every generator. Preserve current models and frozen CNN. Any later detector replacement requires a separate version and a comparison across all generators.

Evidence: protocol.json, summary.json, RESULTS.md, reproduction.png, prefix-audit.json, verification-final.json and evidence-sha256.json. All selected weights, splits, predictions and metrics are retained under folds/000 through folds/209.
