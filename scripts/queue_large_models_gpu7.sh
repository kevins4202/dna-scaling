#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu7_env.sh
mkdir -p logs/large-models logs/research reports/research reports/distance-2048-50m-v1 reports/distance-2048-100m-v1
exec 9>logs/research/queue.lock
flock -n 9 || { echo 'Research queue is already running'; exit 1; }
exec 8>logs/distance-2048-v1/queue.lock
flock -n 8 || { echo 'Distance queue is already running'; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$1" > logs/large-models/status; }
trap 'status "FAILED line=$LINENO exit=$?"' ERR
wait_gpu() {
    status WAITING_FOR_GPU_7
    while true; do
        read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
        if (( memory < 1000 && utilization < 10 )); then break; fi
        sleep 30
    done
}
for size in 50m 100m; do
    wait_gpu
    if [[ ! -f checkpoints/$size-parameters.json ]]; then
        status "INSPECTING $size"
        bash scripts/run_model.sh "$size" inspect >> "logs/large-models/$size-inspect.log" 2>&1
    fi
    if [[ ! -f reports/research/smoke-$size-verification.json ]]; then
        if [[ ! -s checkpoints/smoke-$size/global_step100/mp_rank_00_model_states.pt ]]; then
            status "SMOKE_TRAINING $size 100_STEPS"
            bash scripts/run_model.sh "$size" smoke >> "logs/large-models/$size-smoke.log" 2>&1
        fi
        status "VERIFYING_SMOKE $size"
        (cd vendor/savanna; python launch.py ../../scripts/verify_research_checkpoint.py -d ../../configs data.yml "models/${size}_smoke.yml") >> "logs/large-models/$size-smoke-verification.log" 2>&1
    fi
    status "REGISTERING_READINESS $size"
    python scripts/register_large_model_readiness.py "$size"
    if [[ ! -f reports/research/$size-complete.json ]]; then
        if [[ ! -s checkpoints/$size/global_step3051/mp_rank_00_model_states.pt ]]; then
            status "TRAINING $size 3051_STEPS"
            bash scripts/run_model.sh "$size" train >> "logs/research/$size-train.log" 2>&1
        fi
        status "VERIFYING_RESEARCH_CHECKPOINT $size"
        (cd vendor/savanna; python launch.py ../../scripts/verify_research_checkpoint.py -d ../../configs data100m.yml "models/$size.yml") >> "logs/research/$size-verification.log" 2>&1
        python scripts/check_research_completion.py "$size"
    fi
done
for size in 50m 100m; do
    wait_gpu
    out="reports/distance-2048-$size-v1"
    status "REGISTERING_DISTANCE $size"
    python scripts/distance/family.py "$size"
    if [[ -f "$out/verification-final.json" ]]; then
        python scripts/distance/validate_family.py "$size"
        continue
    fi
    python scripts/verify_project_cnn.py --integrity-only --output "$out/cnn-integrity-before.json"
    status "GENERATING $size 100_BY_2048_BASES"
    (cd vendor/savanna; python launch.py ../../scripts/generate_distance2048.py -d ../../configs data100m.yml "models/$size.yml") >> "logs/large-models/$size-generation.log" 2>&1
    status "SCORING $size FROZEN_CNN"
    python scripts/evaluate_project_cnn.py --natural data/distance-2048-v1/natural.jsonl --generated "data/distance-2048-v1/generated-$size.jsonl" --output "$out/$size-scores.json" --device cuda >> "logs/large-models/$size-scoring.log" 2>&1
    status "PAIRED_FAMILY_ANALYSIS $size"
    python scripts/distance/analyze_family.py "$size" > "logs/large-models/$size-analysis.log" 2>&1
    python scripts/verify_project_cnn.py --integrity-only --output "$out/cnn-integrity-after.json"
    python scripts/distance/validate_family.py "$size"
done
status BOTH_MODELS_AND_DISTANCE_EVALUATIONS_VERIFIED_COMPLETE
