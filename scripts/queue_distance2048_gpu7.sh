#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu7_env.sh
mkdir -p logs/distance-2048-v1 reports/distance-2048-v1
exec 9>logs/distance-2048-v1/queue.lock
flock -n 9 || { echo 'Distance queue already running'; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$1" > logs/distance-2048-v1/status; }
trap 'status "FAILED line=$LINENO exit=$?"' ERR
status CHECKING_INPUTS
python scripts/distance/prepare.py
python scripts/verify_project_cnn.py --integrity-only --output reports/distance-2048-v1/cnn-integrity-before.json
status WAITING_FOR_GPU_7
while true; do
    read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( memory < 1000 && utilization < 10 )); then break; fi
    sleep 30
done
for size in 5m 10m; do
    status "GENERATING $size"
    (cd vendor/savanna; python launch.py ../../scripts/generate_distance2048.py -d ../../configs data100m.yml "models/$size.yml") >> "logs/distance-2048-v1/$size-generation.log" 2>&1
    status "SCORING $size FROZEN_CNN"
    python scripts/evaluate_project_cnn.py --natural data/distance-2048-v1/natural.jsonl --generated "data/distance-2048-v1/generated-$size.jsonl" --output "reports/distance-2048-v1/$size-scores.json" --device cuda >> "logs/distance-2048-v1/$size-scoring.log" 2>&1
done
status PAIRED_ANALYSIS
python scripts/distance/analyze.py > logs/distance-2048-v1/analysis.log 2>&1
python scripts/verify_project_cnn.py --integrity-only --output reports/distance-2048-v1/cnn-integrity-after.json
status VERIFIED_COMPLETE
