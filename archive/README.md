# Historical source only

These are the exact scripts used during completed stages. They are retained for provenance and review, not as current commands. Original relative paths/imports are unchanged, so do not execute them in place against the live project.

- `cnn-pilot/`: 100-prompt pilot, three-seed detector, controls, and scoring.
- `cnn-external/`: external benchmark preparation and external-only development candidate.
- `cnn-final/`: fresh project generation, mixed-data training, sensitivity gate, qualified freeze, and original verification/scoring entry points.
- `initial-setup/`: initial bacterial subset, abandoned full-shard sampler, installation, and smoke training.
- `recovery/`: scripts recovered from Allegro, including environment/config repairs and wheel prefetching.

`provenance/source-layout.json` records original paths and checksums. The pre-cleanup source tarball restores the original `scripts/` layout for an isolated historical reconstruction. Current entry points are indexed in `scripts/README.md`; the deployed CNN is immutable and has no automatic training path.
