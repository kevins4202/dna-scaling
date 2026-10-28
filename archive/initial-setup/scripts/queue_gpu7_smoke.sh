#!/usr/bin/env bash
set -euo pipefail
cd /home/ksong37/cis6200
mkdir -p logs
exec 9>logs/gpu7-smoke.lock
flock -n 9 || { echo 'This queue is already running'; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$1" > logs/gpu7-smoke.status; }
trap 'status "FAILED line=$LINENO exit=$?; inspect logs/gpu7-smoke.log and logs/setup-retry.log"' ERR
export CUDA_VISIBLE_DEVICES=7
export CIS6200_ENV_BIN=/home/ksong37/miniconda3/envs/cis6200/bin
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH=/home/ksong37/miniconda3/envs/cis6200/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/home/ksong37/miniconda3/envs/cis6200/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PATH="/home/ksong37/miniconda3/envs/cis6200/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/ksong37/cis6200/vendor/savanna:${PYTHONPATH:-}"
export TMPDIR=/home/ksong37/cis6200/tmp
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.6
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
export PIP_RESUME_RETRIES=10
if [[ ! -f logs/setup.complete ]]; then
    status 'INSTALLING'
    bash scripts/setup_allegro.sh > logs/setup-retry.log 2>&1
fi
status 'WAITING_FOR_GPU_7'
while true; do
    read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( memory < 1000 && utilization < 10 )); then break; fi
    sleep 30
done
status 'INSPECTING_5M'
bash scripts/run_model.sh 5m inspect > logs/5m-inspect.log 2>&1
status 'PREPROCESSING_TINY_DATA'
bash scripts/preprocess_gpu.sh data/mixture-smoke > logs/preprocess-smoke.log 2>&1
status 'TRAINING_5M_100_STEPS'
bash scripts/run_model.sh 5m smoke > logs/5m-smoke.log 2>&1
status 'TRAINING_COMMAND_FINISHED; checkpoint and sampling validation still required'
