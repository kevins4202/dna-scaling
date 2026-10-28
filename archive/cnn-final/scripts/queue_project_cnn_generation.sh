#!/usr/bin/env bash
set -euo pipefail
cd /home/ksong37/cis6200
exec 9>logs/cnn-final/queue.lock
flock -n 9 || exit 0
source scripts/gpu7_env.sh
export TORCH_COMPILE_DISABLE=1
export CNN_DATA_DIR=data/cnn-project-final
export CNN_BATCH_SIZE=48
gpu_uuid=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
    echo 'blocked: GPU 7 is occupied' > logs/cnn-final/project-status
    exit 1
fi
trap 'echo "failed at line $LINENO" > /home/ksong37/cis6200/logs/cnn-final/project-status' ERR
for size in 5m 10m; do
    echo "generating $size" > logs/cnn-final/project-status
    (
        cd vendor/savanna
        python launch.py ../../scripts/generate_project_cnn_samples.py -d ../../configs data100m.yml "models/$size.yml"
    ) > "logs/cnn-final/generation-$size.log" 2>&1
done
echo generation-complete > logs/cnn-final/project-status
