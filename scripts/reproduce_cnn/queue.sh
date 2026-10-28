#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gpu7_env.sh
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
mkdir -p logs/cnn-original-v1 logs/research logs/distance-2048-v1
exec 9>logs/research/queue.lock
flock -n 9 || { echo 'Research queue already active'; exit 1; }
exec 8>logs/distance-2048-v1/queue.lock
flock -n 8 || { echo 'Distance queue already active'; exit 1; }
exec 7>logs/cnn-original-v1/queue.lock
flock -n 7 || { echo 'Original CNN queue already active'; exit 1; }
trap 'printf "FAILED line=%s exit=%s\n" "$LINENO" "$?" > logs/cnn-original-v1/status' ERR
printf 'WAITING_FOR_GPU_7\n' > logs/cnn-original-v1/status
while true; do
    read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( memory < 1000 && utilization < 10 )); then break; fi
    sleep 30
done
python -u scripts/reproduce_cnn/prepare.py
python -u scripts/reproduce_cnn/run.py
