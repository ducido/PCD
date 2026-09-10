source .venv/bin/activate

module load gcc/13.2.0
module load ffmpeg/7.0.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/PCD/simpler_env/policies/pizero"
export PYTHONPATH="$BASE_DIR/open_pi_zero:$BASE_DIR:$(dirname "$0"):$PYTHONPATH"

M_action_horizon=4
opts="by grounded_sam_tracking alpha 0.2 num_repeats 36 knn_k 3 top_k 3"

python benchmark_step_latency.py \
    --policy pizero \
    --checkpoint pretrained/open-pi-zero \
    --task google_robot_pick_coke_can \
    --contrast \
    --steps knn_topK_motion_step \
    --M-action-horizon $M_action_horizon \
    --num-warmup 20 \
    --num-iters 50 \
    --output ./outputs/latency/knn_topK_motion_step_M${M_action_horizon}.json \
    --opts $opts
