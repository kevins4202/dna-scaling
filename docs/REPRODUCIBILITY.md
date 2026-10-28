# Reproduction, provenance, and deployment

## What is preserved

`provenance/executed-code.jsonl` contains tool code issued in available sessions belonging to this project, including inline Python, shell commands, patches, setup attempts, and failed attempts. It contains no conversational messages or tool outputs. Its receipt records the source-session snapshots, cutoff, checksum, deduplication, and any secret-pattern redactions. This is an audit record, **not a script to replay**: old commands include installation, training, and historical repairs.

`provenance/source-before-organization.tar.gz` preserves the pre-cleanup scripts/configurations and remote-only helpers. `provenance/source-layout.json` maps each archived file to its original path and SHA-256. `archive/` exposes the historical code as readable files. Relative paths and original imports are preserved verbatim; the archive is not a second supported command tree. To reconstruct an old stage, restore its original layout in an isolated workspace with the corresponding data and environment. Do not run repair scripts against the current frozen project.

The active source is formatted and indexed in `scripts/README.md`. Formatting and wrapper cleanup do not change the frozen artifact, datasets, LM checkpoints, or original validation reports.

## External source and environment

Savanna is pinned by `SAVANNA_COMMIT`; the published CNN source is pinned by `CNN_BENCHMARK_COMMIT`. `scripts/setup/restore_vendor.sh` restores those checkouts in a fresh workspace. It refuses to overwrite an existing exported directory or silently reset a different checkout. Allegro currently has an exported subset rather than Git metadata. The source audit records omitted upstream files and verifies that shared vendor files match local source. No vendor source patch was found.

`requirements/allegro-environment.json` records the working Python/package/Conda environment and checks installed Python files against distribution RECORD hashes for the runtime libraries. The checked dependency files had no unrecorded edits. `requirements/allegro-pip.txt` is the package-version snapshot; `requirements/cnn-inference.txt` lists the smaller inference dependency set at those recorded versions. These are evidence of the working Linux/Python 3.12 environment, not a claim that a fresh cold install has been tested. Native extension builds also require the matching CUDA/compiler setup; do not blindly install the Linux snapshot on macOS.

The original installation/retry recipes and wheel prefetch code are in `archive/initial-setup/` and `archive/recovery/`. They explain the successful sequence as well as abandoned attempts. The active Allegro environment is `/home/ksong37/miniconda3/envs/cis6200`; CUDA is `/usr/local/cuda-12.4`. `scripts/gpu7_env.sh` derives the project path, selects physical GPU 7, provides the required library paths, and disables torch.compile consistently with the completed research runs. It installs nothing.

## Routine checks

```bash
python -m unittest discover -s scripts -p 'test_*.py'
python scripts/verify_project_cnn.py --integrity-only
python scripts/verify_project_cnn.py --device cpu
python scripts/audit/repository_audit.py --output reports/code-audit/local.json
bash scripts/run_model.sh 25m inspect --dry-run
bash scripts/queue_research_gpu7.sh --dry-run 25m 50m 100m
```

Full historical CNN training-data checks require the data retained on Allegro:

```bash
source scripts/gpu7_env.sh
python scripts/verify_project_cnn.py --full-data --device cpu
```

These checks write new receipts under `reports/checks/`; they do not overwrite the original research validation or retrain a model. The frozen manifest's sensitivity-gate result remains false even when engineering checks pass.

## Refresh the code archive

From this Mac, with the local session history available:

```bash
python scripts/audit/export_executed_code.py \
  --thread-id 01a16fb4-13f6-76d1-9525-b2171a58fd6d \
  --output provenance/executed-code.jsonl
```

A new export captures a new explicit cutoff. It includes issued code, not a guarantee that every attempted line completed successfully. Relevant outcomes are in the experiment reports and logs. The source snapshot and executed-code archive complement each other: the snapshot preserves runnable-file contents at the old relative paths, while the tool archive retains one-off commands and intervening edits.

## Source-only deployment

The local Git workspace is the canonical source; Allegro is a deployment copy. `scripts/audit/apply_source_release.py` applies a staged release after comparing existing files with a captured baseline. It rejects unexpected remote edits and protects artifacts, checkpoints, datasets, vendor source, logs, and original results. Archived old entry points are removed only after their exact originals have been preserved. The command arguments and applied release are recorded in the tool archive and source-audit receipts.

`reports/code-audit/` records final checks. A source audit distinguishes the deliberately smaller remote vendor export from actual edits to shared code. Do not use a blind synchronization with deletion against the whole project.
