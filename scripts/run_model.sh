#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
size="${1:?Specify 5m, 10m, 25m, 50m, or 100m}"
mode="${2:-inspect}"
dry_run="${3:-}"
case "$size" in 5m|10m|25m|50m|100m) ;; *) echo 'Unknown model size' >&2; exit 1 ;; esac
case "$mode" in
    inspect) program=../../scripts/inspect_model_gpu.py; config="models/$size.yml" ;;
    smoke) program=train.py; config="models/${size}_smoke.yml" ;;
    train) program=train.py; config="models/$size.yml" ;;
    *) echo 'Mode must be inspect, smoke, or train' >&2; exit 1 ;;
esac
[[ -z "$dry_run" || "$dry_run" == --dry-run ]] || { echo 'Unknown argument' >&2; exit 1; }
data_config=data.yml
[[ "$mode" != train ]] || data_config=data100m.yml
if [[ "$dry_run" == --dry-run ]]; then
    echo "GPU 7; torch.compile disabled; mode=$mode; model=$size"
    [[ "$mode" != train ]] || echo 'Execution requires the data/model readiness guard and resumes an existing checkpoint.'
    printf 'cd %q\n' "$PWD/vendor/savanna"
    printf 'python launch.py %q -d ../../configs %q %q\n' "$program" "$data_config" "$config"
    exit 0
fi
source scripts/gpu7_env.sh
[[ "$CUDA_VISIBLE_DEVICES" == 7 ]] || { echo 'This project uses physical GPU 7' >&2; exit 1; }
[[ -x "$CIS6200_ENV_BIN/python" ]] || { echo 'Selected Allegro environment is missing; see docs/REPRODUCIBILITY.md' >&2; exit 1; }
if [[ "$mode" == train ]]; then
    python scripts/check_training_ready.py "$size"
fi
cd vendor/savanna
extra_configs=()
if [[ "$mode" == train && -f "../../checkpoints/$size/latest" ]]; then
    printf '{"load":"../../checkpoints/%s"}\n' "$size" > "../../configs/resume_$size.yml"
    extra_configs+=("resume_$size.yml")
fi
python launch.py "$program" -d ../../configs "$data_config" "$config" "${extra_configs[@]}"
