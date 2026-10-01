#!/usr/bin/env bash
# Run pi0 + SDN (or a baseline / ablation) on SIMPLER.
# Usage (from anywhere): bash scripts/sdn.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .venv/bin/activate ] && source .venv/bin/activate

# cluster modules (skipped if `module` is unavailable); ffmpeg is needed for the episode GIFs
if command -v module >/dev/null 2>&1; then
    module load gcc/13.2.0
    module load ffmpeg/7.0.2
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# vanilla | vanilla_perturbed | sdn | sdn_grounding | sdn_smooth
method="sdn"
num_gpus=4
n_trajs=50
result_root="./eval_logs/sdn"
checkpoint="pretrained/open-pi-zero"

# key value pairs; comma-separated values are grid-searched, e.g. "knn_k 6,10 top_m 3,5 long_horizon 4,5,10"
# negative_mode: zeros_bbox | inpaint | random_zeros_bbox
# by: grounded_sam_tracking | gt | point_tracking | box_tracking
search_opts="by grounded_sam_tracking negative_mode zeros_bbox num_samples 12 knn_k 6 top_m 3 long_horizon 4"

tasks=(
    "google_robot_close_drawer"
    "google_robot_move_near"
    "google_robot_open_drawer"
    "google_robot_pick_coke_can"
    "google_robot_place_apple_in_closed_top_drawer"
    "widowx_carrot_on_plate"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_stack_cube"
)

for task in "${tasks[@]}"; do
    echo "Running $method for pizero on $task"

    python -m pizero.evaluate \
        --method "$method" \
        --n-trajs "$n_trajs" \
        --num-gpus "$num_gpus" \
        --result-root "$result_root" \
        --checkpoint "$checkpoint" \
        --task "$task" \
        --search-opts $search_opts
done
