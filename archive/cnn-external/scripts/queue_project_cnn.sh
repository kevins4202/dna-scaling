#!/usr/bin/env bash
set -euo pipefail
cd /home/ksong37/cis6200
mkdir -p logs/cnn-final
exec 9>logs/cnn-final/queue.lock
flock -n 9 || exit 0
source scripts/gpu7_env.sh
export TORCH_COMPILE_DISABLE=1
gpu_uuid=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
    echo 'blocked: GPU 7 is occupied' > logs/cnn-final/status
    exit 1
fi
echo preparing > logs/cnn-final/status
python scripts/test_project_cnn.py > logs/cnn-final/tests.log 2>&1
echo training > logs/cnn-final/status
if python -u scripts/train_project_cnn.py > logs/cnn-final/train.log 2>&1; then
    echo verifying > logs/cnn-final/status
    python scripts/score_project_cnn.py --input data/cnn-pilot/generated-5m.jsonl --output reports/cnn-v1-training/rescored-5m.csv --device cuda > logs/cnn-final/score.log 2>&1
    echo complete > logs/cnn-final/status
else
    echo 'failed: inspect train.log; no automatic retraining' > logs/cnn-final/status
    exit 1
fi
