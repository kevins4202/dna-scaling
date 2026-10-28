#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu7_env.sh
mkdir -p logs/25m-project reports/distance-2048-25m-v1 logs/research
exec 9>logs/research/queue.lock
flock -n 9 || { echo 'A research job already holds the queue lock'; exit 1; }
exec 8>logs/distance-2048-v1/queue.lock
flock -n 8 || { echo 'A distance experiment already holds its lock'; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$1" > logs/25m-project/status; }
trap 'status "FAILED line=$LINENO exit=$?"' ERR
status WAITING_FOR_GPU_7
while true; do
    read -r memory utilization < <(nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( memory < 1000 && utilization < 10 )); then break; fi
    sleep 30
done
python scripts/verify_project_cnn.py --integrity-only --output reports/distance-2048-25m-v1/cnn-integrity-before.json
if [[ ! -f checkpoints/25m-parameters.json ]]; then
    status INSPECTING_25M
    bash scripts/run_model.sh 25m inspect >> logs/25m-project/inspect.log 2>&1
fi
if [[ ! -f reports/research/smoke-25m-verification.json ]]; then
    if [[ ! -s checkpoints/smoke-25m/global_step100/mp_rank_00_model_states.pt ]]; then
        status SMOKE_TRAINING_25M_100_STEPS
        bash scripts/run_model.sh 25m smoke >> logs/25m-project/smoke.log 2>&1
    fi
    status VERIFYING_SMOKE_25M
    (cd vendor/savanna; python launch.py ../../scripts/verify_research_checkpoint.py -d ../../configs data.yml models/25m_smoke.yml) >> logs/25m-project/smoke-verification.log 2>&1
fi
status REGISTERING_25M_READINESS
python scripts/register_25m_readiness.py
if [[ ! -f reports/research/25m-complete.json ]]; then
    status TRAINING_25M_3051_STEPS
    bash scripts/run_model.sh 25m train >> logs/research/25m-train.log 2>&1
    status VERIFYING_RESEARCH_CHECKPOINT_25M
    (cd vendor/savanna; python launch.py ../../scripts/verify_research_checkpoint.py -d ../../configs data100m.yml models/25m.yml) >> logs/research/25m-verification.log 2>&1
    python scripts/check_research_completion.py 25m
fi
status REGISTERING_DISTANCE_EXTENSION_25M
python scripts/distance/register_25m.py
status GENERATING_25M_100_BY_2048_BASES
(cd vendor/savanna; python launch.py ../../scripts/generate_distance2048.py -d ../../configs data100m.yml models/25m.yml) >> logs/25m-project/generation.log 2>&1
status SCORING_25M_WITH_FROZEN_CNN
python scripts/evaluate_project_cnn.py --natural data/distance-2048-v1/natural.jsonl --generated data/distance-2048-v1/generated-25m.jsonl --output reports/distance-2048-25m-v1/25m-scores.json --device cuda >> logs/25m-project/scoring.log 2>&1
status ANALYZING_THREE_SIZES
python scripts/distance/analyze_25m.py > logs/25m-project/analysis.log 2>&1
python scripts/verify_project_cnn.py --integrity-only --output reports/distance-2048-25m-v1/cnn-integrity-after.json
python scripts/distance/validate_25m.py
status VERIFIED_COMPLETE
