source .venv/bin/activate

module load gcc/13.2.0
module load ffmpeg/7.0.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
# BASE_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/PCD/simpler_env/policies/pizero"
# export PYTHONPATH="$BASE_DIR/open_pi_zero:$BASE_DIR:$(dirname "$0"):$PYTHONPATH"

M_action_horizon=4
num_gpus=1
n_trajs=1
result_root="./results_4gpu_new/openvla/sdn"

search_opts="by grounded_sam_tracking alpha 0.8 num_repeats 36 knn_k 10 top_k 5"


policies=("openvla")
checkpoints=("pretrained/openvla-7b")
tasks=(
    "google_robot_pick_coke_can"
    # "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    # "widowx_put_eggplant_in_basket"
    # "widowx_spoon_on_towel"
    # "widowx_carrot_on_plate"
    # "widowx_stack_cube"
    # "google_robot_place_apple_in_closed_top_drawer"
)

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        python text_ag_parallel_inference.py \
            --contrast \
            --knn-topK-motion \
            --M-action-horizon $M_action_horizon \
            --n-trajs $n_trajs \
            --num-gpus $num_gpus \
            --result-root $result_root \
            --policy ${policies[$i]} \
            --checkpoint ${checkpoints[$i]} \
            --task $task \
            --search-opts $search_opts
    done
done