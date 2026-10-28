# Released Evo 2 7B baseline

The authorized reference is `evo2_7b_base`, the released 8K base model. It is an external pretrained reference, not an additional member of the controlled 100M-token training experiment. Its pretraining overlap with the evaluation sources is unknown.

## Immutable comparison

Use the existing 100 prompts in `data/distance-2048-v1/prompts.jsonl`, each with a 256-base prefix and a 2,048-base natural continuation. Keep the registered manifest, references, small-model generations, scores and CNN unchanged. Generate exactly 2,048 new A/C/G/T bases per prefix, temperature 1, conditional categorical sampling over ACGT, no BOS, top-k, top-p or repetition modification, and the original seed `621900 + id`. The released model uses batch size one with the official recurrent cache; the small models previously used batch size 20.

Checkpoint source: `arcinstitute/evo2_7b_base`, pinned revision `074097e9dc788e8bfe045d6495b9f6153a7c6bfc`. Expected checkpoint: 13,006,429,947 bytes, SHA256 `d8a0e775a5d849921b8725837c6a3cbc71fa15e712f4189a2ed52ef955aad29b`. Preserve resumable downloads and verify the remote LFS checksum. Standard HTTP transport is used because Xet stalled before transferring data on Allegro.

Official code pins are recorded in `scripts/baseline/setup.sh`. The dedicated `.venv-evo2` inherits the existing Torch 2.6 CUDA runtime but installs inference dependencies locally. It does not change the Savanna/CNN environment. Disable Transformer Engine imports only inside the released inference process, selecting the official supported BF16 fallback for the A6000. Long-filter numerical parameters retain the precision chosen by the official loader. Optional Triton inference kernels remain disabled.

Before generation, `scripts/baseline/common.py` registers checkpoint, source, input and prior-score hashes. Strict official model state loading rejects missing model keys and unexpected tensor keys; only unused FP8 metadata and a redundant tied output embedding may be omitted. The redundant output tensor must exactly equal the loaded input embedding, and the official configuration must enable tied embeddings. Smoke verification compares five 32-position teacher-forced cached/full inference probes, with predeclared maximum ACGT logit difference 0.25 and conditional-probability L1 difference 0.03. Repeated 32-base generation must replay exactly. Failed checks require diagnosis rather than relaxed tolerances. Exact instantiated parameter count is recorded after loading.

Generation writes each completed prompt atomically and resumes only records matching the registered prompt, seed, model and protocol hashes. Scoring uses the immutable CNN forward/RC average, four nonoverlapping 512-base windows, and mean-logit aggregate. Analysis reproduces the original 2,000 paired category-stratified whole-source-group bootstrap resamples, seed 6220, with category results and released-minus-small-model contrasts. Higher detection AUROC means generated sequences are easier for this CNN to distinguish from natural sequences.

## Reproduction on Allegro GPU 7

From `/home/ksong37/cis6200`:

```bash
bash scripts/baseline/setup.sh
source scripts/gpu7_env.sh
.venv-evo2/bin/python scripts/baseline/download.py
bash scripts/baseline/queue.sh
```

The queue holds both existing research and distance locks and waits for physical GPU 7 to be idle. It never uses another GPU. Logs/status are under `logs/evo7b-baseline/`. Results, protocol, dependency receipt and verification are under `reports/evo7b-baseline/`; generations are under `data/evo7b-baseline-v1/`. The heartbeat monitor repairs failures, retains valid outputs, and checks until `scripts/baseline/validate.py` writes `verification-final.json`.

## Interpretation limits

The released model has a different training budget, corpus and native architecture. Its result can check whether the frozen evaluator distinguishes a mature reference from our small models, but cannot establish a scaling law. The original panel has 90 source groups, including only six eukaryotic groups, and reuses CNN test groups. The CNN has limited validated sensitivity. Pairwise confidence intervals are exploratory and unadjusted. A high or low CNN score does not establish biological function or sequence safety. Keep the pending expanded heldout study separate from this original-prompt reference experiment.

## Current status

The released checkpoint download and SHA256 verification finished. The initial strict reload guard rejected a redundant `unembed.weight` that the official tied-embedding model does not register separately. The repaired guard requires exact equality with `embedding_layer.weight` and rejects unequal or untied output weights. Three CPU guard tests pass. The failed protocol and smoke log are retained under `reports/evo7b-baseline/pre-generation-recovery-1/`; no continuations existed at re-registration. The GPU 7 queue and monitor remain active; no baseline result is verified yet.

### Attention compatibility recovery

After tied-embedding verification, the official Vortex forward wrapper expected four return values from an installed FlashAttention 2.6.3 kernel that returns eight. `scripts/baseline/compat.py` adapts only the process-local tuple to `(out, softmax_lse, mask, rng_state)`, selecting positions 0, 5, 6 and 7 from the installed documented implementation. It does not change CUDA kernels or the shared environment. Cached attention delegates unchanged. This repair is deployed on Allegro and four CPU guard tests pass. The second failed pre-generation protocol and smoke log are preserved under `reports/evo7b-baseline/pre-generation-recovery-2/`. The lock-protected queue has resumed and retains the original numerical probe limits. No full continuation results are verified yet.

### Cache numerical rejection and reference inference fallback

The fixed cached/full criteria rejected prompt 20: maximum ACGT logit difference 0.08203125 and probability L1 0.05373761. The diagnostic `diagnose_cache.py` compared each cached step with an exact-length full prefix and separately measured full-forward sequence-shape effects. Cached versus exact-prefix probability L1 still reached 0.05073868; full-forward shape effects reached 0.08984955. The original cache thresholds were not relaxed. The baseline now uses the full-prefix forward reference for every generated base, with no recurrent cache. This retains the released weights, BF16 precision, original prompts and conditional ACGT distribution, at higher compute cost. Smoke verification checks finite full-prefix logits and exact repeated full-prefix replay at five 32-position teacher-forced probes, plus deterministic 32-base sampling replay. These checks do not claim that the rejected recurrent cache passed. Failed cache evidence and protocol are archived before any full continuations are generated. The new protocol explicitly records this computation-mode change.

## Verified completion

All 100 continuations, frozen-CNN scores and paired analyses are complete. `scripts/baseline/validate_final.py` verified the results and saved `verification-final.json`. The original registered `validate.py` remains byte-identical: it assumed identical row order, while the evaluator sorts string IDs. The corrected verifier joins unique IDs and retains a hash-bound correction receipt. No regeneration or rescoring was needed. Exact model size is 6,481,059,584 parameters. Aggregate released-model AUROC is 0.5426 (95% interval 0.4306 to 0.6441), versus 0.6850 for 100M. See `reports/evo7b-baseline/SUMMARY.md` for interpretation and limitations. Results and recovery evidence are saved on Allegro and locally.
