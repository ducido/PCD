#!/usr/bin/env bash
# Download all checkpoints into ./pretrained (run from anywhere).
# PaliGemma is gated: accept its license on Hugging Face and run `huggingface-cli login` first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p pretrained

# pi0 (open-pi-zero re-implementation, SIMPLER Bridge/Fractal checkpoints)
huggingface-cli download allenzren/open-pi-zero --local-dir pretrained/open-pi-zero
# PaliGemma tokenizer / processor used by pi0
huggingface-cli download google/paligemma-3b-pt-224 --local-dir pretrained/paligemma-3b-pt-224

# Grounding-DINO + SAM2 for detecting and tracking task-relevant objects
huggingface-cli download IDEA-Research/grounding-dino-base --local-dir pretrained/grounding-dino-base
if [ ! -f pretrained/sam2.1_hiera_large.pt ]; then
    wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

# LaMa (only needed for negative_mode=inpaint): download the 'big-lama' folder from
# https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg
# and place it at pretrained/big-lama
if [ ! -d pretrained/big-lama ]; then
    echo "NOTE: pretrained/big-lama not found; download it manually for negative_mode=inpaint."
fi
