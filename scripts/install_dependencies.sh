#!/usr/bin/env bash
# Install the environment for pi0 + SDN on SIMPLER (Python 3.10, CUDA 11.8).
# Run from the repository root inside a fresh environment, e.g.
#   conda create -n sdn python=3.10 && conda activate sdn
#   bash scripts/install_dependencies.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# PyTorch / TensorFlow (TensorFlow is used by the SIMPLER image preprocessing)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install "tensorflow[and-cuda]==2.15.0"

# SIMPLER simulator
pip install ruckig --only-binary=:all:
pip install -e third_party/ManiSkill2_real2sim

# Grounding-DINO + SAM2 (negative-set construction)
pip install -e third_party/grounded_sam_2
pip install --no-build-isolation -e third_party/grounded_sam_2/grounding_dino

# LaMa inpainting (only needed for negative_mode=inpaint)
pip install -r third_party/inpaint_anything/lama/requirements.txt

# everything else
pip install -r requirements.txt
