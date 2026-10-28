# Original benchmark CNN reproduction

This is a separate authorized experiment using the published benchmark's original full natural/generated FASTA pairs. Our frozen CNN, generated sequences, and prior results are preserved.

Source: https://github.com/Georgakopoulos-Soares-lab/Synthetic_genomes_benchmark at `2e2ab3808f4f55db1fbcf22df39610a17f4da1dc`. Data: https://zenodo.org/records/18826945, archive MD5 `353488a6529ce72f8d0f435d5d761e1d`.

We execute the upstream training, model selection, and evaluation functions without modifying their code. Settings follow `run_all_experiments.py`: 1024-base chunks, 96 base channels, dilations 1/2/2/4, Adam at 0.0003, weight decay 0.0001, batch 8, four training chunks, 10 epochs, 10% calibration. Leave-one-tag-out evaluation trains within each original domain, with all 20 eukaryotic, four prokaryotic, and six viral tags. Chunk-count experiments use 1/2/4/8/16/32 chunks with separately selected checkpoints, as upstream does. We run 32 first to obtain the principal long-sequence comparison before the remaining sweep. Total: 180 chunk-count folds plus 30 distance folds.

The authors' complete original commands, global RNG states, and trained weights were not available in the checked release. Direct model defaults differ from the experiment launcher's defaults. Consequently, this is a faithful reproduction of the released launcher procedure, not a claim of bitwise reproduction or exact published-number matching. We record Python, NumPy, Torch and PYTHONHASHSEED explicitly; upstream does not seed all global random initialization. The published environment does not pin Torch/NumPy versions; our actual versions are recorded without mutating shared dependencies.

Upstream requires an explicit distance grid but does not supply the original command. We register the chosen grid before training: euk/prok 0,256,2048,4096,10000,20000,50000,100000,200000,290000; virus 0,256,2048,4096,10000,20000,40000. Distances are absolute sequence-start coordinates, as the upstream code implements. They must not be mislabeled as distances beyond the conditioning prompt. Unavailable positions remain missing with observed sample counts.

The original FASTA concatenation, conditioning-prefix inclusion, ambiguous-base handling, and item-level calibration split are retained for fidelity. Natural/generated counterparts can occur on opposite sides of the fit/calibration split; the held-out test tag remains excluded from both. These limitations are audited, not silently corrected. Dataset staging only repairs root/case path references; sequence bytes stay identical and all pairs are required. The original archive and manifests remain unchanged.

Observation hooks retain each selected checkpoint, strict reload evidence, split identities, original predictions, per-fold metrics and timing. No test performance selects configurations. Completed folds are verified and resumed without rerunning. Source/data hashes and the existing frozen CNN's hashes are checked before and after. Execution is sequential on physical GPU7 with the existing research/distance locks.

Run: `bash scripts/reproduce_cnn/queue.sh`. Status/logs: `logs/cnn-original-v1/`. Registered protocol, dataset inventory, environment, fold checkpoints/predictions, result table and plots: `reports/cnn-original-v1/`. Staged path-only manifests: `data/cnn-original-v1/`.

Primary question: does the released training/evaluation procedure recover strong detection on the original benchmark? A discrepancy with our adapted project setup is not automatically a software bug or a causal attribution. More controlled follow-ups can vary sequence position, chunk length/count, architecture and training distribution after this reproduction.

Verified launch: 2,757,313 parameters, all 1,358 original pairs, 210 folds, GPU7. The first fold is eukaryotic Aedes_aegypti held out with 32 evaluation chunks. The monitor is `monitor-original-benchmark-cnn-reproduction`, every ten minutes, quiet during routine progress.

## Verified completion

All 210 folds completed and passed independent strict checkpoint reload, saved-prediction AUROC reproduction, heldout-tag exclusion checks, registered input hashes and frozen CNN integrity. Detection using 32 chunks averages 0.9804 for eukaryotes, 0.7962 for prokaryotes and 0.7512 for viruses. Read `reports/cnn-original-v1/SUMMARY.md` for interpretation and the conditioning-prefix audit. Near-chance scores in the first approximately 3000 bases mostly concern copied prompt text, not new generation. The next step is a separately registered matched evaluation comparison, not automatic retraining of our language models.
