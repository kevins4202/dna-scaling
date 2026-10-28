#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
sizes=("$@")
if [[ ${#sizes[@]} == 0 ]]; then sizes=(5m 10m); fi
for size in "${sizes[@]}"; do
    case "$size" in 5m|10m|25m|50m|100m) ;; *) echo 'Unknown model size' >&2; exit 1 ;; esac
done
if "$dry_run"; then
    for size in "${sizes[@]}"; do
        if [[ -f "reports/research/$size-complete.json" ]]; then
            echo "SKIP $size: completion report already exists"
        else
            echo "PLAN $size: GPU 7 -> readiness guard -> resumable training -> strict checkpoint verification"
        fi
    done
    exit 0
fi
source scripts/gpu7_env.sh
mkdir -p logs/research reports/research
exec 9>logs/research/queue.lock
flock -n 9 || { echo 'Research queue already running'; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$1" > logs/research/status; }
trap 'status "FAILED line=$LINENO exit=$?; inspect logs/research"' ERR
for size in "${sizes[@]}"; do
    if [[ -f "reports/research/$size-complete.json" ]]; then continue; fi
    status "WAITING_FOR_GPU_7 $size"
    while true; do
        read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
        if (( memory < 1000 && utilization < 10 )); then break; fi
        sleep 30
    done
    status "CHECKING_READINESS $size"
    python scripts/check_training_ready.py "$size"
    status "TRAINING $size 3051_STEPS"
    bash scripts/run_model.sh "$size" train >> "logs/research/$size-train.log" 2>&1
    status "VERIFYING_CHECKPOINT $size"
    (
        cd vendor/savanna
        python launch.py ../../scripts/verify_research_checkpoint.py -d ../../configs data100m.yml "models/$size.yml"
    ) > "logs/research/$size-verification.log" 2>&1
    python scripts/check_research_completion.py "$size"
    status "COMPLETED $size"
done
status 'REQUESTED_MODELS_VERIFIED_COMPLETE'
