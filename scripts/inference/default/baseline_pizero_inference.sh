
source .venv/bin/activate

module load gcc/13.2.0
module load ffmpeg/7.0.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

BASE_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/PCD/simpler_env/policies/pizero"
export PYTHONPATH="$BASE_DIR/open_pi_zero:$BASE_DIR:$(dirname "$0"):$PYTHONPATH"

num_gpus=1
result_root="./results_4gpu/default/baseline_paraphrased_text"

policies=("pizero")
checkpoints=("pretrained/open-pi-zero")
tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_carrot_on_plate"
    "widowx_stack_cube"
    "google_robot_place_apple_in_closed_top_drawer"
)

# /projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/PCD/simpler_env/policies/pizero/open_pi_zero/config/eval/fractal.yaml

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        python parallel_inference_paraphrased.py \
            --num-gpus $num_gpus \
            --result-root $result_root \
            --policy ${policies[$i]} \
            --checkpoint ${checkpoints[$i]} \
            --task $task
    done
done
