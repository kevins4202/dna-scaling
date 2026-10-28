#!/usr/bin/env bash
set -euo pipefail
cd /home/ksong37/cis6200
exec 9>logs/cnn-final/queue.lock
flock 9
source scripts/gpu7_env.sh
export TORCH_COMPILE_DISABLE=1
if test -d artifacts/cnn-v1; then exit 0; fi
if ! grep -Fxq generation-complete logs/cnn-final/project-status; then
    echo 'blocked: generation did not finish' > logs/cnn-final/final-status
    exit 1
fi
gpu_uuid=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
    echo 'blocked: GPU 7 is occupied' > logs/cnn-final/final-status
    exit 1
fi
trap 'echo "failed at line $LINENO; no automatic retraining" > /home/ksong37/cis6200/logs/cnn-final/final-status' ERR
echo preparing > logs/cnn-final/final-status
python scripts/build_final_cnn_data.py > logs/cnn-final/mixed-data.log 2>&1
echo training > logs/cnn-final/final-status
python -u scripts/train_final_project_cnn.py > logs/cnn-final/final-train.log 2>&1
echo verifying > logs/cnn-final/final-status
python scripts/verify_project_cnn.py > logs/cnn-final/final-verification.log 2>&1
echo complete > logs/cnn-final/final-status
