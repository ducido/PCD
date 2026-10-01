"""Default configurations for pi0 + SDN on SIMPLER.

Any key below can be overridden from the command line with `--opts key value ...`.
"""

import copy

# Evaluation methods.
#   vanilla            : pi0 on the original observation.
#   vanilla_perturbed  : pi0 on the perturbed observation (Tables IV and V).
#   sdn                : full SDN (grounding filter + smoothness refinement).
#   sdn_grounding      : SDN stage 1 only.
#   sdn_smooth         : SDN stage 2 only (no negative set needed).
METHODS = ['vanilla', 'vanilla_perturbed', 'sdn', 'sdn_grounding', 'sdn_smooth']
METHODS_WITH_NEGATIVE_IMAGE = ['vanilla_perturbed', 'sdn', 'sdn_grounding']
SDN_METHODS = ['sdn', 'sdn_grounding', 'sdn_smooth']

PIZERO_CONFIG = dict(
    cfg_dir='simpler_env/policies/pizero/open_pi_zero/config/eval',
    flow_sampling='beta',
    use_ddp=False,
    use_naive=False,
    use_torch_compile=True,
)

SDN_CONFIG = dict(
    num_samples=12,      # N
    knn_k=6,             # k of the k-NN grounding score
    top_m=3,             # M, candidates kept by the grounding filter
    long_horizon=None,   # extended chunk length for the smoothness stage (None = pi0's horizon)
    lambda_energy=0.05,
    ignore_gripper=True,
)

NEGATIVE_IMAGE_CONFIG = dict(
    camera_name=None,
    by='grounded_sam_tracking',  # Grounding-DINO + SAM2; 'gt' uses the simulator segmentation
    negative_mode='zeros_bbox',  # 'zeros_bbox' | 'inpaint' | 'random_zeros_bbox'
    inpaint_mode='lama',
    bbox_pad=3,
    random_bbox_margin=10,
    version=2,
    get_all_parts=False,
)


def _override(config, opts):
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config


def get_policy_setup(task):
    if task.startswith('google_robot'):
        return 'google_robot'
    if task.startswith('widowx'):
        return 'widowx_bridge'
    raise NotImplementedError(f"Unknown robot for task {task}")


def get_policy_config(checkpoint, task, opts, method):
    config = copy.deepcopy(PIZERO_CONFIG)
    config['checkpoint_path'] = checkpoint
    config['policy_setup'] = get_policy_setup(task)
    if method in SDN_METHODS:
        config.update(copy.deepcopy(SDN_CONFIG))
    return _override(config, opts)


def get_negative_image_config(opts):
    return _override(copy.deepcopy(NEGATIVE_IMAGE_CONFIG), opts)


def check_opts(opts):
    known = set(PIZERO_CONFIG) | set(SDN_CONFIG) | set(NEGATIVE_IMAGE_CONFIG)
    unknown = sorted(set(opts) - known)
    if unknown:
        raise ValueError(f"Unknown options {unknown}; valid options are {sorted(known)}")


def build_policy(checkpoint, task, opts, method):
    config = get_policy_config(checkpoint, task, opts, method)
    if method in SDN_METHODS:
        from pizero.sdn_policy import PiZeroSDNInference
        return PiZeroSDNInference(**config), config
    from simpler_env.policies.pizero.pizero_model import PiZeroInference
    return PiZeroInference(**config), config
