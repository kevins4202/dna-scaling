# Current execution plan

The original proposal remains unchanged in `proposal/main.tex`. This document records current implementation status and the user's narrower focus on one fixed CNN. Additional structure scores, released billion-parameter models, and reinforcement learning are deferred rather than silently added to the core experiment.

## Completed

- Prepare and screen a bounded 100M-character OpenGenome2 training subset, with separate 1M-character validation/test sets and an independent reference sample. Preserve the sampling and taxonomic-coverage limitations.
- Train 5,413,780- and 10,331,804-parameter StripedHyena 2 models for 3,051 optimizer steps each: 99,975,168 processed tokens, the same data/order, character tokenizer, 4,096-base context, optimizer, and schedule. Both used `TORCH_COMPILE_DISABLE=1`; this remains the family-wide setting.
- Verify saved checkpoints, strict reload, finite logits, and DNA sampling. Record held-out loss as supporting evidence. The two runs' held-out losses are nearly tied; this is not a scaling conclusion.
- Freeze one CNN v1 checkpoint and scoring implementation. The main fresh test uses 160 matched 512-base continuations from 135 source groups. The declared sensitivity gate failed for 5M; it remains a qualified diagnostic. Do not retrain, recalibrate, or switch its checkpoint to improve future results.

## Milestone 1: the proposal's distance experiment

Completed and independently verified for 5M then 10M on physical GPU 7. Both generated 100 matched 2,048-base continuations from 90 source groups with the unchanged CNN. Aggregate detection AUROCs were 0.684 and 0.687; the paired difference interval includes zero, so this experiment provides no clear advantage for either size. The distance curve is nonmonotonic. Protocol, coverage limits and completed evidence are recorded in `docs/DISTANCE_2048.md` and `reports/distance-2048-v1/`.

Prepare a versioned evaluation set of 100 held-out natural prefixes of 256 bases with matched natural continuations of 2,048 bases. Register ids, source groups, category weights, sampling settings, and exclusions before scoring. Exclude CNN training/calibration sources and screen close sequence matches. Longer-window eligibility and missing categories must be explicit. The old frozen 512-base reference cannot simply be relabeled as a 2,048-base reference.

Generate 2,048 bases from both existing models using identical prompts and sampling settings. Score four successive 512-base windows with the unchanged CNN and also report the sequence aggregate. Bootstrap complete source groups, retaining paired prompts across models and distances. Four windows from one source are not four independent samples.

Deliverable: reproducible detection-versus-distance curves for 5M and 10M, uncertainty intervals, runtime, and the exact evaluation manifest. A CNN score near chance is reported as limited distinguishability; it is not proof of biological function, nor a reason to retrain the evaluator.

## Milestone 2: complete the controlled model family

All five model sizes are trained and independently verified on GPU 7. Exact totals are 5,413,780; 10,331,804; 24,886,444; 51,927,104; and 97,352,024 parameters. Each completed 3,051 steps and 99,975,168 processed tokens, strict final-checkpoint reload, finite logits and DNA generation. All five completed the same 100-prompt, 2,048-base frozen-CNN distance evaluation.

Aggregate detection AUROCs are 0.684, 0.687, 0.676, 0.647 and 0.685 in size order. Every paired aggregate 95% source-group interval includes zero; the point estimates do not form a monotonic size trend. Held-out losses improve slightly through 50M and rise slightly at 100M. These observations do not establish a scaling law. Preserve the fixed CNN's limited sensitivity and the six eukaryotic source groups supporting 35% of the weighted statistic.

Complete five-model results, curve and paired intervals are in `reports/distance-2048-100m-v1/`; training/held-out summaries are in `reports/research-family/`. Earlier experiments are preserved. Protocol and checkpoint evidence are recorded in `docs/LARGE_MODELS.md`, with verified copies retained locally. The training/evaluation queue is complete and the monitor is removed after final verification.

## Milestone 3: answer the research question

Plot CNN distinguishability against distance for all five sizes. Fit a simple size relationship only after the full family has been evaluated, with uncertainty across source groups and repeated generations. Keep conclusions proportional to the CNN's measured sensitivity; do not force a scaling law from weak separation or two sizes. Additional LM seeds can quantify training variation without changing the CNN.

Then run the proposal's diagnosis: natural-prefix versus generated-prefix continuations, with equal prefix lengths, matched source regions, and identical sampling. Context shortening is a follow-up controlled comparison using the same frozen evaluator.

Deliverable: the size/distance comparison, the prefix-history comparison, and an honest account of effects the detector could not resolve.

## Deferred

RL fine-tuning is not ready: the frozen CNN has limited sensitivity and is not validated as a sole reward. The proposal also requires independent evaluation beyond the reward itself. Do not add that phase automatically. The current project remains the controlled model-family and CNN distance experiment.

## Released 7B reference on the original prompts

The authorized `evo2_7b_base` reference is queued on physical GPU 7, with the original 100 registered prompts and frozen CNN. Installation is isolated; released checkpoint checksums, strict reload, cached/full numerical probes and reproducible sampling precede generation. A five-minute heartbeat monitors download, generation, scoring and final verification. See [RELEASED_BASELINE.md](RELEASED_BASELINE.md). This reference has a different pretraining history and unknown evaluation-source overlap. It does not extend the controlled fixed-100M-token scaling experiment. The expanded heldout panel remains a separate pending study.

## Released baseline verified complete

The released Evo 2 7B base reference completed the original 100-prompt experiment with the CNN frozen. Its aggregate detection AUROC is 0.5426 (95% interval 0.4306 to 0.6441), compared with 0.6472 to 0.6874 for our five small models. All descriptive paired aggregate released-minus-small-model intervals exclude zero, with unadjusted exploratory scope. Full-prefix inference replaced a numerically rejected cache before generation; all original prompts, seeds and weights were retained. The next priority remains a separately registered heldout panel with more independent source groups, especially eukaryotic groups. The earlier expanded-panel collection still needs to meet screened group targets. See `reports/evo7b-baseline/SUMMARY.md` and `docs/RELEASED_BASELINE.md`.

## Reproduce original benchmark CNN before further model comparisons

Authorized October 27, 2026. A separate reproduction now runs on physical GPU7 using all 1,358 original benchmark pairs, the unchanged upstream training/evaluation functions, and released experiment-launcher settings. It performs domain-specific leave-one-tag-out training across 30 tags, six chunk aggregation counts, and a registered distance analysis (210 folds). The existing frozen CNN and earlier experiments remain unchanged.

The launcher's CNN has 2,757,313 parameters, versus 1,227,905 for our adapted detector. Settings, provenance, limitations of exact replication, checkpoint/prediction retention, and differences from our project setup are documented in `docs/CNN_REPRODUCTION.md`. Outputs: `reports/cnn-original-v1/`; status: `logs/cnn-original-v1/status`. A ten-minute monitor will verify completion, synchronize evidence, and report results. Do not interpret partial folds as a completed reproduction or retune on their heldout results.

## Original benchmark CNN reproduction verified complete

All 210 folds completed on GPU7 and passed independent verification. The original procedure recovers strong detection: mean 32-chunk AUROC 0.9804 eukaryotic, 0.7962 prokaryotic, 0.7512 viral. The earlier project CNN remains byte-identical. The original data typically copy a 3000-base prompt; distance coordinates and copied-prefix windows must be distinguished from newly generated DNA. Full evidence and interpretation: `reports/cnn-original-v1/SUMMARY.md`. Next: separately register matched evaluation comparisons that isolate position and sequence aggregation while retaining source exclusions. Do not automatically retrain language models or replace our frozen evaluator.

## Six-model reproduced-CNN reassessment completed

The authorized transfer comparison scored all six saved generation sets with a fixed domain-balanced ensemble of 30 reproduced detectors, two 1024-base new-sequence windows each. All 100 prompts from 90 groups passed screening against the complete original CNN corpus and known small-LM training text; zero reported alignments met the registered256-base / 90% thresholds. Primary AUROCs:5M 0.5560,10M 0.6030,25M 0.5630,50M 0.5306,100M 0.5080,released 7B 0.5837. The evaluator-dependent ranking and secondary direction reversals limit biological-quality claims. Independent integrity/ensemble/metric checks passed. Preserve prior CNN and all generation outputs; next priority is separately registered transfer validation and independent diagnostics, not automatic language-model retraining. Full evidence: reports/cnn-reassessment-v1/SUMMARY.md.
