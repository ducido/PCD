#!/usr/bin/env bash
# Baselines and ablations for pi0 on SIMPLER.
# Usage (from anywhere): bash pizero/scripts/run_ablations.sh [num_gpus] [result_root]
#   vanilla pi0                               (Tables II, IV, V)
#   pi0 on masked / inpainted observations    (Table IV)
#   pi0 on randomly masked observations       (Table V)
#   SDN with inpainting negatives             (Table VII)
#   SDN stage 1 only / stage 2 only           (ablation)
# Add e.g. `--opts knn_k 10 top_m 5 long_horizon 5` below to match per-task settings.
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_GPUS=${1:-1}
RESULT_ROOT=${2:-./results/pizero}
N_TRAJS=50

TASKS=(
    google_robot_close_drawer
    google_robot_move_near
    google_robot_open_drawer
    google_robot_pick_coke_can
    google_robot_place_apple_in_closed_top_drawer
    widowx_carrot_on_plate
    widowx_put_eggplant_in_basket
    widowx_spoon_on_towel
    widowx_stack_cube
)

run() {  # run <method> <task> [opts...]
    local method=$1 task=$2; shift 2
    python -m pizero.evaluate --method "$method" --task "$task" --num-gpus "$NUM_GPUS" \
        --n-trajs "$N_TRAJS" --result-root "$RESULT_ROOT" "$@"
}

for task in "${TASKS[@]}"; do
    run vanilla "$task"
    run vanilla_perturbed "$task" --opts negative_mode zeros_bbox
    run vanilla_perturbed "$task" --opts negative_mode inpaint
    run vanilla_perturbed "$task" --opts negative_mode random_zeros_bbox
    run sdn "$task" --opts negative_mode inpaint
    run sdn_grounding "$task" --opts negative_mode zeros_bbox
    run sdn_smooth "$task"
done
