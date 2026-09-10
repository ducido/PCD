"""Generate ISS heatmaps for one prepared X-ICM episode."""

from __future__ import annotations


'''
source .venv/bin/activate
module load gcc/13.2.0
module load cuda/12.6.2
'''


import json
import os
import numpy as np
import imageio.v2 as imageio

import hydra
from iss_nmr_toolkit.constants import DEFAULT_VIEWS
from iss_nmr_toolkit.core.iss import compute_iss
from iss_nmr_toolkit.io.artifacts import save_iss_npz
from iss_nmr_toolkit.runner_utils import (
    iss_path,
    output_dir,
    quiet_third_party_logs,
    resolve_episode_dir,
    resolve_prompt,
)
from omegaconf import DictConfig, OmegaConf
import os
import torch

from contrast_policies.pizero_contrast import PiZeroContrastInference


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    quiet_third_party_logs()
    episode_dir = resolve_episode_dir(cfg)
    out_dir = output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)


    policy_cfg = dict(
        cfg_dir='simpler_env/policies/pizero/open_pi_zero/config/eval',
        use_ddp=False,
        use_naive=False,
        use_torch_compile=True,
        checkpoint_path=cfg.policy.checkpoint_dir,
        num_repeats=24
    )
    policy = PiZeroContrastInference(**policy_cfg)

    video_path = os.path.join(cfg.data.root, f"episode_{cfg.data.episode}.gif")
    contrast_video_path = os.path.join(cfg.data.root, f"episode_{cfg.data.episode}_contrast.gif")
    ooi_path = os.path.join(cfg.data.root, f"episode_{cfg.data.episode}_ooi.gif")
    state_path = os.path.join(cfg.data.root, f"episode_{cfg.data.episode}_states.npy")

    rgb_frames = imageio.mimread(video_path)
    contrast_frames = imageio.mimread(contrast_video_path)
    ooi_frames = imageio.mimread(ooi_path)
    states = np.load(state_path, allow_pickle=True)

    assert len(rgb_frames) == len(contrast_frames) == len(ooi_frames) == len(states)

    observations = []
    for step in range(len(rgb_frames)):
        obs = {}
        obs['image'] = rgb_frames[step][:,:,:3]
        obs['contrast_image'] = contrast_frames[step][:,:,:3]
        obs['states'] = states[step]['states']
        obs['prompt'] = states[step]['prompt']
        observations.append(obs)


    DEFAULT_VIEWS = {
        "front": "image",
        # "wrist": "wrist_image",
        # "overhead": "overhead_image",
    }


    heatmaps = compute_iss(
        policy,
        observations,
        views=DEFAULT_VIEWS,
        n_masks=int(cfg.iss.n_masks),
        p_keep=float(cfg.iss.p_keep),
        grid_size=(int(cfg.iss.grid_size[0]), int(cfg.iss.grid_size[1])),
        blur_sigma=float(cfg.iss.blur_sigma),
        time_stride=int(cfg.iss.time_stride),
        batch_size=int(cfg.iss.batch_size),
        seed=int(cfg.iss.seed),
        algo=cfg.policy.algo
    )

    path = save_iss_npz(
        iss_path(cfg),
        heatmaps,
        meta={
            "episode_dir": str(episode_dir.resolve()),
            "policy": "pi05",
            "config_name": cfg.policy.config_name,
            "n_masks": int(cfg.iss.n_masks),
            "p_keep": float(cfg.iss.p_keep),
            "grid_size": [int(cfg.iss.grid_size[0]), int(cfg.iss.grid_size[1])],
            "blur_sigma": float(cfg.iss.blur_sigma),
            "time_stride": int(cfg.iss.time_stride),
            "seed": int(cfg.iss.seed),
            "views": DEFAULT_VIEWS,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )
    summary = {
        "iss_heatmaps": str(path),
        "episode_dir": str(episode_dir),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (out_dir / "iss_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved ISS heatmaps: {path}")


if __name__ == "__main__":
    main()
