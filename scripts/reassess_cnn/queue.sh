#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gpu7_env.sh
export MPLBACKEND=Agg
mkdir -p logs/cnn-reassessment-v1 logs/research logs/distance-2048-v1
exec 9>logs/research/queue.lock
flock -n 9 || { echo 'Research queue already active'; exit 1; }
exec 8>logs/distance-2048-v1/queue.lock
flock -n 8 || { echo 'Distance queue already active'; exit 1; }
exec 7>logs/cnn-reassessment-v1/queue.lock
flock -n 7 || { echo 'Reassessment queue already active'; exit 1; }
trap 'printf "FAILED line=%s exit=%s\n" "$LINENO" "$?" > logs/cnn-reassessment-v1/status' ERR
printf 'TESTING\n' > logs/cnn-reassessment-v1/status
python scripts/reassess_cnn/test_reassessment.py
printf 'SCREENING_KNOWN_TRAINING_OVERLAP\n' > logs/cnn-reassessment-v1/status
python -u scripts/reassess_cnn/prepare.py
printf 'WAITING_FOR_GPU7\n' > logs/cnn-reassessment-v1/status
while true; do
    read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( memory < 1000 && utilization < 10 )); then break; fi
    sleep 30
done
printf 'SCORING_FIXED_REPRODUCED_CNN_ENSEMBLE\n' > logs/cnn-reassessment-v1/status
python -u scripts/reassess_cnn/score.py
printf 'PAIRED_ANALYSIS\n' > logs/cnn-reassessment-v1/status
python -u scripts/reassess_cnn/analyze.py
python -u scripts/reassess_cnn/validate.py
printf 'CNN_REASSESSMENT_VERIFIED_COMPLETE\n' > logs/cnn-reassessment-v1/status
