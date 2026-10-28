#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null || { echo 'Requires Linux with an NVIDIA CUDA GPU.'; exit 1; }
command -v nvcc >/dev/null || { echo 'Requires CUDA toolkit (nvcc).'; exit 1; }
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
cd vendor/savanna
make setup-env
