source .venv/bin/activate

module load gcc/13.2.0
module load ffmpeg/7.0.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# bash scripts/inference/default/baseline_octo_inference.sh
# bash scripts/inference/default/baseline_openvla_inference.sh
# bash scripts/inference/default/baseline_pizero_inference.sh

# bash scripts/inference/default/contrast_octo_inference.sh
# bash scripts/inference/default/contrast_openvla_inference.sh
bash scripts/inference/default/contrast_pizero_inference.sh