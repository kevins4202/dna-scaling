# OpenGenome2 sampling

## Commands

Python 3 standard library only. Run from the project root. Each output directory must be new; failed runs remain marked incomplete and must not be used.

Tiny integration test, already run successfully:

```bash
python3 scripts/sample_opengenome.py --mode smoke --train-tokens 100000 --eval-tokens 10000 --output data/mixture-smoke
python3 scripts/validate_opengenome.py data/mixture-smoke
```

Full 100M-character training subset, to run on the GPU host, NOT downloaded locally:

```bash
python3 scripts/sample_opengenome.py --mode full --train-tokens 100000000 --eval-tokens 1000000 --output data/mixture-100m
python3 scripts/validate_opengenome.py data/mixture-100m
bash scripts/preprocess_gpu.sh data/mixture-100m
```

Full mode streams ALL matching compressed shards sequentially. It retains a bounded candidate sample rather than the entire corpus on disk. Network transfer is much larger than 100 MB and can reach terabytes because gzip shards must be scanned to avoid prefix bias. This implementation prioritizes inspecting the whole population, not cheap source access. Plan network/time on the GPU host before running. The optional `--shards-per-category N` selects N shuffled whole shards per category to reduce transfer, but then represents only a random shard subset. It must not be presented as verified population-wide sampling.

## Target mixture

Source: https://huggingface.co/datasets/arcinstitute/opengenome2#training-data-composition

Pinned revision: 84d2a7e690c8bb395d1f4868822dfa031ea561c6

Use initial-pretraining weights, not later context-extension weights or the unrelated mixture in some Savanna configs. The published weights total 100.52; divide by 100.52 and allocate integer character budgets by largest remainder.

| Category | Published weight |
|---|---:|
| GTDB + IMG/PR | 18 |
| Metagenomes | 24 |
| IMG/VR | 3 |
| Stitched mRNA | 9 |
| mRNA | 9 |
| Eukaryotic genic windows | 35 |
| Organelles | 0.5 |
| ncRNA | 2 |
| Promoters | 0.02 |

A token budget here means ASCII characters of the upstream JSONL `text`. Original case, ambiguous characters, and special text are preserved. Savanna's character tokenizer encodes characters; `--append-eod` adds one extra separator token per output record. Thus 100M means 100M source characters, not exactly 100M encoded tokens after separators. Models must subsequently train for the same measured encoded-token budget.

## Sampling and checks

- Preserve upstream train/valid/test splits. Do not randomly split adjacent sequence windows.
- Shuffle shard order with a fixed seed; in full mode inspect complete selected shards.
- Assign independent random priorities to records. Keep the lowest-priority records until their combined length covers the category budget, then shuffle them and crop a random substring of the final record to meet the exact budget. Short records are retained without a minimum-length filter. No artificial concatenation or ACGT-only filtering is applied.
- This is approximate record-based sampling with exact category budgets. Random record selection is not an exact uniform nucleotide sample; the variable-length stopping rule and final cropping have finite-sample effects. It does not guarantee organism proportions.
- Smoke mode reads only prefixes of one randomly chosen shard per category. It tests download/parsing/allocation/output only, not representativeness. Tiny categories may contain very short final fragments.
- Manifest records source paths, row numbers, offsets, random seed, pinned dataset revision, scanned reference statistics, selected statistics, scanned-stream hashes, and output hashes.
- Validator checks quotas, upstream split provenance, file integrity, GC summaries, and exact cross-split duplicate records of at least 128 characters. It reports duplicates rather than silently changing the distribution.
- Homology, reverse-complement matches, organism coverage, and offset substring overlap are NOT checked yet. No claim of homology-separated evaluation or identical within-category distribution is justified. Before research training, review GC and category diagnostics and add organism/homology checks appropriate to the available metadata.

## Verification

`python3 scripts/test_sampler.py` exercises exact budget rounding, deterministic selection, complete shard scans, preservation of case and non-ACGT characters, and failure on insufficient input. The live tiny test covers all nine categories and three splits. It passed budget/hash/provenance checks with zero identical cross-split output records of at least 128 characters.
