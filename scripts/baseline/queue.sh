#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gpu7_env.sh
mkdir -p logs/evo7b-baseline reports/evo7b-baseline data/evo7b-baseline-v1
exec 9>logs/research/queue.lock
flock -n 9 || { echo 'Research queue already active'; exit 1; }
exec 8>logs/distance-2048-v1/queue.lock
flock -n 8 || { echo 'Distance queue already active'; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$1" > logs/evo7b-baseline/status; }
trap 'status "FAILED line=$LINENO exit=$?"' ERR
status WAITING_FOR_VERIFIED_RELEASED_CHECKPOINT
while ! python -c 'import json;assert json.load(open("reports/evo7b-baseline/download-manifest.json"))["status"]=="verified"' 2>/dev/null; do sleep 30; done
status WAITING_FOR_GPU_7
while true; do
    read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( memory < 1000 && utilization < 10 )); then break; fi
    sleep 30
done
python scripts/verify_project_cnn.py --integrity-only --output reports/evo7b-baseline/cnn-integrity-before.json
python scripts/baseline/common.py
export PYTHONPATH="$PWD/vendor/evo2:$PWD/vendor/vortex"
if [[ ! -f reports/evo7b-baseline/smoke-verification.json ]]; then
    status VERIFYING_STRICT_RELOAD_AND_CACHED_INFERENCE
    .venv-evo2/bin/python -u scripts/baseline/generate.py --smoke >> logs/evo7b-baseline/smoke.log 2>&1
fi
status GENERATING_RELEASED_7B_100_BY_2048
.venv-evo2/bin/python -u scripts/baseline/generate.py >> logs/evo7b-baseline/generation.log 2>&1
status SCORING_WITH_FROZEN_CNN
python scripts/evaluate_project_cnn.py --natural data/distance-2048-v1/natural.jsonl --generated data/evo7b-baseline-v1/generated-evo2_7b_base.jsonl --output reports/evo7b-baseline/released-7b-scores.json --device cuda >> logs/evo7b-baseline/scoring.log 2>&1
status PAIRED_ANALYSIS
python scripts/baseline/analyze.py > logs/evo7b-baseline/analysis.log 2>&1
python scripts/verify_project_cnn.py --integrity-only --output reports/evo7b-baseline/cnn-integrity-after.json
python scripts/baseline/validate_final.py > logs/evo7b-baseline/validation.log 2>&1
status RELEASED_7B_BASELINE_VERIFIED_COMPLETE
