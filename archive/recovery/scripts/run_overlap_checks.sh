#!/usr/bin/env bash
set -euo pipefail
cd /home/ksong37/cis6200
export PATH=/home/ksong37/miniconda3/envs/cis6200/bin:$PATH
mkdir -p reports/overlap
minimap2 -x asm10 -t 4 -d data/mixture-100m-bounded/train.mmi data/mixture-100m-bounded/train.fasta
minimap2 -x asm10 -c --secondary=no -t 4 data/mixture-100m-bounded/train.mmi data/mixture-100m-bounded/valid.fasta > reports/overlap/valid-vs-train.paf
minimap2 -x asm10 -c --secondary=no -t 4 data/mixture-100m-bounded/train.mmi data/mixture-100m-bounded/test.fasta > reports/overlap/test-vs-train.paf
minimap2 -x asm10 -c --secondary=no -t 4 data/mixture-100m-bounded/valid.fasta data/mixture-100m-bounded/test.fasta > reports/overlap/test-vs-valid.paf
printf 'ALIGNMENTS_COMPLETE\n'
