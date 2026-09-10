
source .venv/bin/activate
module load gcc/13.2.0
module load cuda/12.6.2

TASKS=(
    # google_robot_pick_coke_can
    google_robot_move_near
    # widowx_stack_cube
    # widowx_carrot_on_plate
)

# algo=base
# algo=knn_topK_long_motion
algo=knn
# algo=pcd


for task in "${TASKS[@]}"; do
    for ep in {1..2}; do

        python gen_iss_mask.py \
            data.root="results_4gpu_explain/default/knn_topK_long_delta_motion_4_shape/open-pi-zero/by\=gt--alpha\=0.2--num_repeats\=12--knn_k\=3--top_k\=3/${task}/" \
            data.task=$task \
            data.episode=$ep \
            policy.checkpoint_dir=pretrained/open-pi-zero \
            policy.algo=$algo \
            output.dir=outputs/$task/$algo/$ep



        python compute_nmr_k.py \
            data.root="results_4gpu_explain/default/knn_topK_long_delta_motion_4_shape/open-pi-zero/by\=gt--alpha\=0.2--num_repeats\=12--knn_k\=3--top_k\=3/${task}/" \
            data.task=$task \
            data.episode=$ep \
            output.dir=outputs/$task/$algo/$ep


        python vis_iss_nmr_k.py \
            data.root="results_4gpu_explain/default/knn_topK_long_delta_motion_4_shape/open-pi-zero/by\=gt--alpha\=0.2--num_repeats\=12--knn_k\=3--top_k\=3/${task}/" \
            data.task=$task \
            data.episode=$ep \
            output.dir=outputs/$task/$algo/$ep
    done
done