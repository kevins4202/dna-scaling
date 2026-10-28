#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gpu7_env.sh
mkdir -p vendor logs/evo7b-baseline reports/evo7b-baseline
clone_pin() {
    local url="$1" destination="$2" revision="$3"
    if [[ ! -d "$destination/.git" ]]; then git clone "$url" "$destination"; fi
    if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
        git -C "$destination" fetch origin "$revision"
        git -C "$destination" checkout --detach "$revision"
    fi
    [[ -z "$(git -C "$destination" status --porcelain --untracked-files=no)" ]] || { echo "Vendor source modified: $destination"; exit 1; }
}
clone_pin https://github.com/ArcInstitute/evo2.git vendor/evo2 53f195997257c56c00e5ef8d33a54f5baad143a6
clone_pin https://github.com/Zymrael/vortex.git vendor/vortex 8b00afebeac745d1f31e7e2788f0e0e39fa47637
if [[ ! -x .venv-evo2/bin/python ]]; then "$CIS6200_ENV_BIN/python" -m venv --system-site-packages .venv-evo2; fi
.venv-evo2/bin/python -m pip install --no-deps biopython==1.85 rich==13.9.4 markdown-it-py==3.0.0 mdurl==0.1.2 einops==0.8.1
.venv-evo2/bin/python -m pip install --no-deps --no-build-isolation -e vendor/vortex -e vendor/evo2
.venv-evo2/bin/python -m pip freeze --local > reports/evo7b-baseline/requirements-local.txt
PYTHONPATH="$PWD/vendor/evo2:$PWD/vendor/vortex" .venv-evo2/bin/python -c 'import sys; sys.modules["transformer_engine"]=None; import torch,evo2,vortex; print("IMPORTS_OK",torch.__version__)'
