OPEN_PIZERO_CONFIG = dict(
    cfg_dir='simpler_env/policies/pizero/open_pi_zero/config/eval',
    use_ddp=False,
    use_naive=False,
    use_torch_compile=True,
)

CONTRAST_IMAGE_CONFIG = dict(
    camera_name=None,
    by="gt",
    inpaint_mode="lama",
    color="auto",
    sigma=5,
    version=2,
    get_all_parts=False,
)


CONTRAST_OPEN_PIZERO_CONFIG = dict(
    num_repeats=12,
    knn_k=6,
    top_M=3,
    long_ah=4
)


def get_policy_config(policy, checkpoint, task, opts, algo):
    if policy == 'rt1':
        config = RT1_CONFIG
        config['saved_model_path'] = checkpoint
    elif policy == 'octo':
        config = OCTO_CONFIG
        config['model_type'] = checkpoint
    elif policy == 'openvla':
        config = OPENVLA_CONFIG
        config['saved_model_path'] = checkpoint
    elif policy == 'pizero':
        config = OPEN_PIZERO_CONFIG
        config['checkpoint_path'] = checkpoint
    else:
        raise NotImplementedError()
    
    # select policy setup based on task
    if task.startswith('google_robot'):
        config['policy_setup'] = 'google_robot'
    elif task.startswith('widowx'):
        config['policy_setup'] = 'widowx_bridge'
    else:
        raise NotImplementedError

    # update config if contrast policy is used
    if algo in ['grounding_and_smooth', 'grounding', 'smooth']:
        from properties import CONTRAST_OCTO_CONFIG, CONTRAST_OPENVLA_CONFIG
        config.update(CONTRAST_OPEN_PIZERO_CONFIG)

    # update opts
    for k, v in opts.items():
        if k in config:
            config[k] = v
    
    return config


def get_contrast_image_generator_config(opts):
    config = CONTRAST_IMAGE_CONFIG
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config
