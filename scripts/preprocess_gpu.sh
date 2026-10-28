#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
dataset="${1:-data/mixture-smoke}"
[[ "$dataset" = /* ]] || dataset="$root/$dataset"
python3 scripts/validate_opengenome.py "$dataset"
if [[ -n "${CIS6200_ENV_BIN:-}" ]]; then
    export PATH="$CIS6200_ENV_BIN:$PATH"
else
    source .venv-gpu/bin/activate
fi
output="${2:-data/tokenized}"
[[ "$output" = /* ]] || output="$root/$output"
mkdir -p "$output"
cd vendor/savanna
for split in train valid test; do
    PYTHONPATH=. python tools/preprocess_data.py --input "$dataset/$split.jsonl" --output-prefix "$output/$split" --tokenizer-type CharLevelTokenizer --append-eod --workers 2
done
