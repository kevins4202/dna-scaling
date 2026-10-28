#!/usr/bin/env bash
# Source this on Allegro before CUDA commands. It does not install or train anything.
CIS6200_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=7
export CIS6200_ENV_BIN="${CIS6200_ENV_BIN:-/home/ksong37/miniconda3/envs/cis6200/bin}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CIS6200_ENV_BIN:$CUDA_HOME/bin:$PATH"
CIS6200_ENV_LIB="$(dirname "$CIS6200_ENV_BIN")/lib"
export LD_LIBRARY_PATH="$CIS6200_ENV_LIB:$CIS6200_ENV_LIB/python3.12/site-packages/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CIS6200_PROJECT_ROOT/vendor/savanna:${PYTHONPATH:-}"
export TMPDIR="$CIS6200_PROJECT_ROOT/tmp"
export TORCH_COMPILE_DISABLE=1
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.6
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
export PIP_RESUME_RETRIES=10
