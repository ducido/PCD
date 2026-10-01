# Selected Diffusion Noise (SDN)

Official implementation of **"Test-Time Improvement of VLA Policies via Selected Diffusion Noise for Spurious-Robust Action Smoothing"**.

SDN is a training-free, test-time action-selection method for diffusion / flow-matching VLA policies.
At every step it samples several candidate action chunks from different initial noises and selects one in two stages:

1. **Grounding filter.** It also samples a negative set from a counterfactual observation in which the task-relevant objects are masked out.
   It then keeps the top-M candidates that lie in dense regions of the positive set and in sparse regions of the negative set (k-NN density ratio).
2. **Kinematic refinement.** Among those M candidates, it executes the smoothest chunk.

This repository contains the π0 experiments on SIMPLER. The GR00T N1.5 / N1.6 experiments will be added under [`gr00t/`](gr00t/).

## Repository structure

```
sdn/                       backbone-agnostic SDN core, shared by all backbones
  selection.py             k-NN grounding score, smoothness score, two-stage selection
  negatives/               negative observations: Grounding-DINO + SAM2 masks, zero-bbox / LaMa inpainting
pizero/                    π0 on SIMPLER
  sdn_policy.py            π0 wrapper that samples candidates and calls sdn.selection
  config.py                default hyperparameters
  evaluate.py              multi-GPU SIMPLER evaluation
  scripts/                 run_sdn.sh (main results), run_ablations.sh (baselines / ablations)
gr00t/                     GR00T integration (coming soon)
simpler_env/               SIMPLER environments + open-pi-zero model code (vendored)
third_party/               ManiSkill2_real2sim, Grounded-SAM-2, LaMa (vendored)
tests/                     unit tests for sdn.selection
```

## Installation

```bash
conda create -n sdn python=3.10
conda activate sdn
bash scripts/install_dependencies.sh
```

```bash
huggingface-cli login
```

```bash
bash scripts/download_pretrained_weights.sh
```

`huggingface-cli login` is needed for the gated PaliGemma weights. LaMa weights (`pretrained/big-lama`) are only required for `negative_mode inpaint` and must be downloaded manually; see the script.

## Running

All commands are run from the repository root. Workers are launched on the GPUs listed in `CUDA_VISIBLE_DEVICES` (or all visible GPUs).

Evaluate π0 + SDN on one task:

```bash
python -m pizero.evaluate --method sdn --task widowx_spoon_on_towel --num-gpus 4 --opts knn_k 6 top_m 3 long_horizon 4
```

Reproduce the π0 tables:

```bash
bash pizero/scripts/run_sdn.sh 4
```

```bash
bash pizero/scripts/run_ablations.sh 4
```

`run_sdn.sh` gives Table II. `run_ablations.sh` gives the vanilla / masked / inpainted / random-mask baselines (Tables IV, V), SDN with inpainting negatives (Table VII) and the single-stage ablations.

Results are written to `<result-root>/<method>/<options>/<task>/`: one GIF per episode (original and negative view side by side) and `000_success_<rate>.log`.

### Methods

| `--method` | Description |
|---|---|
| `vanilla` | π0 on the original observation |
| `vanilla_perturbed` | π0 on the perturbed observation (object masked, inpainted or randomly masked) |
| `sdn` | full SDN (grounding filter + kinematic refinement) |
| `sdn_grounding` | stage 1 only: execute the most grounded candidate |
| `sdn_smooth` | stage 2 only: execute the smoothest candidate (no negative set) |

### Options

Pass options as `--opts key value ...`, or as a grid with `--search-opts key v1,v2 ...`. Unknown keys raise an error.

| Key | Default | Meaning |
|---|---|---|
| `num_samples` | 12 | N, candidates per observation (for both the positive and the negative set) |
| `knn_k` | 6 | k of the k-NN grounding score |
| `top_m` | 3 | M, candidates kept by the grounding filter |
| `long_horizon` | `None` | length of the extended chunks scored by the smoothness stage (`None`: π0's chunk length, 4) |
| `lambda_energy` | 0.05 | weight of the energy term in the smoothness score |
| `ignore_gripper` | `True` | exclude the gripper dimension from the smoothness score |
| `negative_mode` | `zeros_bbox` | `zeros_bbox`, `inpaint` (LaMa) or `random_zeros_bbox` (control) |
| `by` | `grounded_sam_tracking` | object masks from Grounding-DINO + SAM2 tracking; `gt` uses simulator segmentation |
| `bbox_pad` | 3 | padding (px) of the zeroed bounding box |

## Using SDN with another backbone

`sdn/selection.py` depends only on PyTorch. Given candidate chunks from the original observation, `actions` with shape `[N, T, D]`, and from the negative observation, `neg_actions` with shape `[C, T, D]`:

```python
from sdn.selection import sdn_select

idx = sdn_select(actions, neg_actions, knn_k=6, top_m=3, grounding_horizon=exec_horizon)
chunk_to_execute = actions[idx, :exec_horizon]
```

Run the unit tests with `python -m pytest tests/`.

## Acknowledgements

This code builds on [Policy Contrastive Decoding (PCD)](https://github.com/Koorye/PCD), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [open-pi-zero](https://github.com/allenzren/open-pi-zero), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) and [Inpaint-Anything / LaMa](https://github.com/geekyutao/Inpaint-Anything). We thank the authors for releasing their code.

## Citation

```bibtex
@inproceedings{sdn2027,
  title     = {Test-Time Improvement of VLA Policies via Selected Diffusion Noise for Spurious-Robust Action Smoothing},
  author    = {Anonymous},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2027}
}
```
