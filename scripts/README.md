# Current command index

Run commands from the repository root. Nothing here automatically trains the CNN.

| Area | Entry points | Purpose |
|---|---|---|
| Frozen CNN | `score_cnn.py`, `evaluate_project_cnn.py` | Delegate to the original immutable implementation in `artifacts/cnn-v1/`. `score_project_cnn.py` is a compatibility alias. |
| CNN checks | `verify_project_cnn.py`, `test_project_cnn.py`, `cnn/` | Hashes, strict reload, finite scores, strand symmetry, aggregation, and optional historical data separation. CPU default. |
| Data sampling | `sample_opengenome.py` | Weighted sampler used for tiny/full-sized bounded subsets. Do not rerun sampling against the frozen experiment directories. |
| Data validation | `validate_opengenome.py`, `audit_dataset.py`, `broaden_gtdb_reference.py`, `repair_euk_sampling.py`, `clean_eval_overlap.py` | Original dataset audit/repair tools. They can change datasets; use isolated/versioned outputs and revalidate before any new experiment. |
| Preprocessing | `preprocess_gpu.sh` | Produce Savanna indexed data; defaults to tiny smoke data. Select output/data directories deliberately. |
| Model configuration | `build_models.py`, `test_models.py`, `inspect_model_gpu.py` | Family definitions, shared-control checks, and actual GPU parameter counting. The builder preserves verified completion labels. |
| Readiness | `check_training_ready.py`, `finalize_training_readiness.py` | Verify the recorded artifacts and model readiness. Current readiness covers 5M/10M only. |
| LM training | `run_model.sh`, `queue_research_gpu7.sh` | GPU 7, torch.compile disabled, readiness guard, checkpoint resume, sequential queue. Both support safe dry-run review. |
| Distance experiment | `queue_distance2048_gpu7.sh`, `generate_distance2048.py`, `distance/`, `test_distance2048.py` | Matched 100-prompt 2,048-base evaluation, frozen CNN, paired source-group intervals. See `docs/DISTANCE_2048.md`. |
| 25M extension | `queue_25m_project_gpu7.sh`, `register_25m_readiness.py`, `distance/register_25m.py`, `distance/analyze_25m.py`, `distance/validate_25m.py` | Smoke/readiness -> controlled training -> same-prompt frozen-CNN evaluation. See `docs/MODEL_25M.md`. |
| Larger family | `queue_large_models_gpu7.sh`, `register_large_model_readiness.py`, `distance/family.py`, `distance/analyze_family.py`, `distance/validate_family.py`, `test_distance_family.py` | Sequential 50M/100M training, then unchanged-prompt frozen-CNN evaluations. See `docs/LARGE_MODELS.md`. |
| LM results | `verify_research_checkpoint.py`, `check_research_completion.py`, `export_research_results.py` | Strict reload/generation and retained training metrics. |
| Environment | `gpu7_env.sh`, `setup/restore_vendor.sh` | Use the existing Allegro environment and restore pinned external source in a fresh workspace. |
| Audit | `audit/repository_audit.py`, `audit/capture_environment.py`, `audit/export_executed_code.py` | Source/artifact integrity, installed environment, and one-off code provenance. |
| One-time cleanup/deployment | `audit/organize_repository.py`, `audit/apply_source_release.py` | Preserved maintenance tools with overwrite/preflight guards; not experiment commands. |
| Tests | `test_sampler.py`, `test_models.py`, `test_project_cnn.py` | Discover with `python -m unittest discover -s scripts -p 'test_*.py'`. |

Completed CNN training code is archived by phase in `archive/cnn-pilot/`, `archive/cnn-external/`, and `archive/cnn-final/`. Setup/recovery recipes are archived separately. The original source layout is preserved in the provenance tarball. Importing current scoring code does not import a trainer or create a training output directory.
