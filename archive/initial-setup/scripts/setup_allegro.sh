#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH="$CUDA_HOME"
export PATH="/home/ksong37/miniconda3/envs/cis6200/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=/home/ksong37/cis6200/tmp
export PIP_NO_CACHE_DIR=1
export CPATH=/home/ksong37/miniconda3/envs/cis6200/lib/python3.12/site-packages/nvidia/cudnn/include
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.6
cd /home/ksong37/cis6200/vendor/savanna
python -m pip install --upgrade pip 'setuptools<81' wheel packaging ninja
python -m pip install --no-deps /home/ksong37/cis6200/wheels/*.whl
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://pypi.org/simple
python -m pip install -r requirements/requirements.txt
python -m pip install PyYAML einops lazy_import_plus omegaconf opt-einsum pybind11 regex requests scipy sentencepiece 'tokenizers>=0.20.1' wandb boto3 arrow 'pydantic<=2.8.2' 'numpy<=2.2.6' tensorboard pandas tqdm ninja packaging pytest
python -m pip install --no-build-isolation deepspeed causal-conv1d 'flash-attn>=2.1.1,<=2.6.3'
python -m pip install --no-build-isolation ring_flash_attn
python -m pip install 'transformer-engine[pytorch]==1.13.0' --no-build-isolation
cd savanna/data
make
cd /home/ksong37/cis6200/vendor/savanna
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0)); from savanna.arguments import GlobalConfig; print("SAVANNA_IMPORT_OK")'
touch /home/ksong37/cis6200/logs/setup.complete
