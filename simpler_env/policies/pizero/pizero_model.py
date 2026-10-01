import hydra
import os.path as osp
import torch
from omegaconf import OmegaConf

import sys
sys.path.append(osp.join(osp.dirname(__file__), 'open_pi_zero'))
sys.path.append(osp.dirname(__file__))
from src.model.vla.pizero import PiZero

from .. import setup_torch_seed

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
_CHECKPOINTS = {
    ('widowx_bridge', 'beta'): 'bridge_beta_step19296_2024-12-26_22-30_42.pt',
    ('widowx_bridge', 'uniform'): 'bridge_uniform_step19296_2024-12-26_22-31_42.pt',
    ('google_robot', 'beta'): 'fractal_beta_step29576_2024-12-29_13-10_42.pt',
    ('google_robot', 'uniform'): 'fractal_uniform_step29576_2024-12-31_22-26_42.pt',
}
_CONFIGS = {'widowx_bridge': 'bridge.yaml', 'google_robot': 'fractal.yaml'}


def resolve_path(path):
    """Resolve a path relative to the repository root (absolute paths are unchanged)."""
    return path if osp.isabs(path) else osp.join(REPO_ROOT, path)


def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)


class PiZeroInference:
    def __init__(self,
                 cfg_dir,
                 checkpoint_path,
                 policy_setup="widowx_bridge",
                 flow_sampling='beta',
                 use_ddp=False,
                 use_naive=False,
                 use_torch_compile=False,
                 seed=0):
        self.use_naive = use_naive

        if (policy_setup, flow_sampling) not in _CHECKPOINTS:
            raise ValueError(f"Unsupported policy_setup={policy_setup} / flow_sampling={flow_sampling}")
        cfg = OmegaConf.load(osp.join(resolve_path(cfg_dir), _CONFIGS[policy_setup]))
        checkpoint_path = osp.join(resolve_path(checkpoint_path), _CHECKPOINTS[(policy_setup, flow_sampling)])
        adapter_cfg = cfg.env.adapter
        adapter_cfg.dataset_statistics_path = resolve_path(adapter_cfg.dataset_statistics_path)
        adapter_cfg.pretrained_model_path = resolve_path(adapter_cfg.pretrained_model_path)

        cfg.flow_sampling = flow_sampling
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda')
        self.model = PiZero(cfg, use_ddp=use_ddp)
        load_checkpoint(self.model, checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        
        self.base_model = self.model  # un-compiled module, for changing attributes
        if use_torch_compile:
            self.model = torch.compile(self.model, mode='default')
        self.model.eval()

        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)
        self.env_adapter.reset()
        
    def reset(self, instruction, seed=None):
        self.env_adapter.reset()
        if seed is not None:
            setup_torch_seed(seed)
    
    def step(self, image, instruction, proprio, *args, **kwargs):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        raw_actions = self.forward_actions(inputs)
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions

    def preprocess_inputs(self, image, instruction, proprio):
        inputs = self.env_adapter.preprocess(image, instruction, proprio)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = \
            (self.model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=self.dtype))
        image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(causal_mask)
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(self.dtype),
        }

        if self.use_naive:
            inputs.update({"causal_mask": causal_mask})
        else:
            inputs.update({
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
            })
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def forward_actions(self, inputs):
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_action_naive(**inputs)
            else:
                actions = self.model.infer_action(**inputs)
        return actions
