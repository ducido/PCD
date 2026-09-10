"""
Measure the wall-clock latency (ms) of policy `*_step` methods with dummy inputs.

No SimplerEnv environment is built: images / proprio are synthesized, so only the
policy forward path is timed (preprocess + model forward + contrast decoding +
postprocess), exactly as it is called in `text_ag_parallel_inference.py`.

The arguments of each step method are filled by name from a pool of dummy values,
so any method of the policy can be timed by name, e.g.:

    python benchmark_step_latency.py \
        --policy pizero \
        --checkpoint pretrained/open-pi-zero \
        --task google_robot_pick_coke_can \
        --contrast \
        --steps knn_topK_motion_step base_best_of_N_smooth_step \
        --M-action-horizon 4 \
        --num-warmup 10 \
        --num-iters 50 \
        --opts by grounded_sam_tracking alpha 0.2 num_repeats 36 knn_k 10 top_k 5
"""

import argparse
import inspect
import json
import numpy as np
import os
import os.path as osp
import time

import torch

from utils import parse_opts


# arguments of the step methods are resolved by parameter name
DUMMY_INSTRUCTION = "pick coke can"
DUMMY_NEGATIVE_INSTRUCTION = "do nothing, just stand still"


def build_dummy_inputs(image_size, M_action_horizon):
    """
    Build a pool of dummy inputs, keyed by the parameter names used by the
    `*_step` methods of the policies.
    """
    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, size=(*image_size, 3), dtype=np.uint8)
    contrast_image = rng.randint(0, 256, size=(*image_size, 3), dtype=np.uint8)

    # obs['agent']['eef_pos']: [xyz (3), quat wxyz (4), gripper (1)]
    proprio = np.array([0.35, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    return {
        'image': image,
        'contrast_image': contrast_image,
        'instruction': DUMMY_INSTRUCTION,
        'task_description': DUMMY_INSTRUCTION,
        'negative_instruction': DUMMY_NEGATIVE_INSTRUCTION,
        'negative_prompt': DUMMY_NEGATIVE_INSTRUCTION,
        'proprio': proprio,
        'contrast_proprio': torch.zeros_like(torch.tensor(proprio)),
        'M_action_horizon': M_action_horizon,
    }


def bind_step_args(step_fn, dummy_inputs):
    """
    Map the signature of `step_fn` onto the dummy input pool.
    Return the kwargs to call it with.
    """
    signature = inspect.signature(step_fn)
    kwargs = {}
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in dummy_inputs:
            kwargs[name] = dummy_inputs[name]
        elif param.default is not param.empty:
            continue  # optional and unknown, let the policy use its default
        else:
            raise ValueError(
                f"No dummy input for required argument '{name}' of "
                f"{step_fn.__qualname__}; add it to build_dummy_inputs()."
            )
    return kwargs


def set_gpu(gpu_id):
    """ Same as ParallelRunner._set_gpu: must be called before building the policy. """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # list_physical_devices can avoid cuda error, don't know why
    import tensorflow as tf
    tf.config.list_physical_devices("GPU")


def build_policy(args):
    """ Build the policy exactly like ParallelRunner._build_policy, minus the logger. """
    from properties import get_policy_config

    config = get_policy_config(args.policy,
                               args.checkpoint,
                               args.task,
                               parse_opts(args.opts),
                               args.contrast,
                               ag=args.ag,
                               cd_knn=args.cd_knn,
                               # the "my contrast" config is a superset of the plain
                               # contrast one (adds ag_weight / knn_k / top_k), so ask
                               # for it whenever a contrast policy is benchmarked
                               knn_topK_motion=args.contrast)
    print("Policy config:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    from contrast_policies import get_policy
    return get_policy(args.policy, args.contrast, config)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_step(policy, step_name, dummy_inputs, num_warmup, num_iters):
    """
    Time one step method:
        1. reset the policy
        2. run `num_warmup` steps without recording (compilation / cudnn autotune /
           lazy CUDA init happen here)
        3. run `num_iters` steps, recording the elapsed time of each one
    Return:
        - latencies: list of per-call latencies in ms
    """
    step_fn = getattr(policy, step_name, None)
    if step_fn is None or not callable(step_fn):
        raise AttributeError(f"Policy {type(policy).__name__} has no step method '{step_name}'.")

    kwargs = bind_step_args(step_fn, dummy_inputs)
    print(f"\n=== {step_name}({', '.join(kwargs)}) ===")

    policy.reset(DUMMY_INSTRUCTION, seed=0)

    print(f"Warming up {num_warmup} steps...")
    for i in range(num_warmup):
        step_fn(**kwargs)
        synchronize()

    print(f"Measuring {num_iters} steps...")
    latencies = []
    for i in range(num_iters):
        synchronize()
        start = time.perf_counter()
        step_fn(**kwargs)
        synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def stat_latencies(latencies):
    latencies = np.asarray(latencies, dtype=np.float64)
    return {
        'num_iters': int(latencies.size),
        'mean_ms': float(latencies.mean()),
        'std_ms': float(latencies.std()),
        'median_ms': float(np.median(latencies)),
        'min_ms': float(latencies.min()),
        'max_ms': float(latencies.max()),
        'p90_ms': float(np.percentile(latencies, 90)),
        'fps': float(1000.0 / latencies.mean()),
    }


def print_table(results):
    headers = ['step', 'iters', 'mean(ms)', 'std(ms)', 'median(ms)', 'min(ms)', 'max(ms)', 'p90(ms)', 'calls/s']
    rows = [[name,
             f"{s['num_iters']}",
             f"{s['mean_ms']:.2f}",
             f"{s['std_ms']:.2f}",
             f"{s['median_ms']:.2f}",
             f"{s['min_ms']:.2f}",
             f"{s['max_ms']:.2f}",
             f"{s['p90_ms']:.2f}",
             f"{s['fps']:.2f}"] for name, s in results.items()]

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = '  '.join(h.ljust(w) for h, w in zip(headers, widths))
    print('\n' + line)
    print('-' * len(line))
    for row in rows:
        print('  '.join(cell.ljust(w) for cell, w in zip(row, widths)))


def main(args):
    set_gpu(args.gpu_id)
    policy = build_policy(args)
    dummy_inputs = build_dummy_inputs(args.image_size, args.M_action_horizon)

    results = {}
    latencies = {}
    for step_name in args.steps:
        latencies[step_name] = time_step(policy, step_name, dummy_inputs,
                                         args.num_warmup, args.num_iters)
        results[step_name] = stat_latencies(latencies[step_name])
        print(f"{step_name}: {results[step_name]['mean_ms']:.2f} +- "
              f"{results[step_name]['std_ms']:.2f} ms")

    print_table(results)

    if args.output is not None:
        if osp.dirname(args.output):
            os.makedirs(osp.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'policy': args.policy,
                'checkpoint': args.checkpoint,
                'task': args.task,
                'contrast': args.contrast,
                'M_action_horizon': args.M_action_horizon,
                'num_warmup': args.num_warmup,
                'opts': parse_opts(args.opts),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'results': results,
                'latencies_ms': latencies,
            }, f, indent=2)
        print(f"\nSaved timings to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="pizero")
    parser.add_argument("--checkpoint", type=str, default="pretrained/open-pi-zero")
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--contrast", action="store_true",
                        help="build the contrast policy (needed for every *_step except step/baseline_step)")
    parser.add_argument("--ag", action="store_true", help="only affects which config is loaded")
    parser.add_argument("--cd-knn", action="store_true", help="only affects which config is loaded")
    parser.add_argument("--steps", nargs="+", default=["knn_topK_motion_step"],
                        help="names of the policy methods to time, e.g. knn_topK_motion_step knn_de_step")
    parser.add_argument("--M-action-horizon", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 640],
                        help="H W of the dummy camera image, resized by the policy anyway")
    parser.add_argument("--output", type=str, default=None, help="optional path of a json report")
    parser.add_argument("--opts", nargs="+", default=[])
    args = parser.parse_args()
    main(args)
