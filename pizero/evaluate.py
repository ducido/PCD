"""Evaluate pi0 (+ SDN) on SIMPLER.

Example (run from the repository root):
    python -m pizero.evaluate --method sdn --task google_robot_pick_coke_can --num-gpus 4 \
        --opts knn_k 6 top_m 3 long_horizon 4

Episodes are split across GPUs; each GPU process builds its own environment, policy and
negative-image generator. Results (per-episode GIFs and a summary log) are written to
<result-root>/<method>/<opts>/<task>/.
"""

import argparse
import gc
import itertools
import multiprocessing
import os
import os.path as osp
import queue
import shutil
import traceback

import numpy as np
import torch

from pizero.config import (METHODS, METHODS_WITH_NEGATIVE_IMAGE, build_policy, check_opts,
                           get_negative_image_config)
from pizero.eval_utils import (Logger, convert_numpy_or_torch_to_python, parse_opts, reset_logging,
                               stat_final, stat_first, stat_info, summarize, tile_images, write_video)

gpu_lock = multiprocessing.Lock()


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    if camera_name is None:
        if "google_robot" in env.unwrapped.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


def get_visible_gpus():
    """GPU ids to launch workers on, respecting CUDA_VISIBLE_DEVICES if it is set."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    lines = os.popen("nvidia-smi --query-gpu=index --format=csv,noheader").readlines()
    return [line.strip() for line in lines if line.strip()]


class ParallelRunner:
    def __init__(self,
                 method,
                 task,
                 checkpoint,
                 num_gpus=1,
                 result_root='./results',
                 n_trajs=50,
                 reload_every=8,
                 save_video=True,
                 opts=None):
        self.method = method
        self.task = task
        self.checkpoint = checkpoint
        self.num_gpus = num_gpus
        self.result_root = result_root
        self.n_trajs = n_trajs
        self.reload_every = reload_every
        self.save_video = save_video
        self.opts = dict(opts or {})
        self.use_negative_image = method in METHODS_WITH_NEGATIVE_IMAGE

    def run(self):
        if not self._set_result_dir():
            return
        self._build_logger()

        gpu_ids = get_visible_gpus()[:max(1, self.num_gpus)]
        if len(gpu_ids) == 0:
            raise RuntimeError("No GPU found")

        if len(gpu_ids) == 1:
            infos = self.run_episodes(gpu_ids[0], range(self.n_trajs), show_detail=True)
        else:
            infos = self._run_in_parallel(gpu_ids)

        info = stat_info(infos)
        # workers write to the same log file, so re-open it before writing the summary
        self._build_logger(mode='a')
        self.logger.infos("Results", info)
        os.rename(osp.join(self.result_dir, '000.log'),
                  osp.join(self.result_dir, f"000_success_{round(info['success'], 4)}.log"))

    def _run_in_parallel(self, gpu_ids):
        info_queue = multiprocessing.Queue()
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            episodes = list(range(i, self.n_trajs, len(gpu_ids)))
            self.logger.info(f"Allocating episodes for GPU {gpu_id}: {episodes}.")
            process = multiprocessing.Process(target=self.run_episodes,
                                              args=(gpu_id, episodes, info_queue, i == 0))
            process.start()
            processes.append(process)

        # drain the queue before joining, otherwise workers can block on a full pipe
        infos = []
        while len(infos) < self.n_trajs:
            try:
                infos.append(info_queue.get(timeout=60))
            except queue.Empty:
                failed = [p.pid for p in processes if p.exitcode not in (None, 0)]
                if failed:
                    for p in processes:
                        p.terminate()
                    raise RuntimeError(f"Worker processes {failed} failed, see {self.result_dir}")
        for process in processes:
            process.join()
        return infos

    def run_episodes(self, gpu_id, episodes, info_queue=None, show_detail=False):
        env, policy = self.build_episode(gpu_id, show_detail)
        infos = []
        for i, episode in enumerate(episodes):
            self.logger.info(f"Running episode {episode} on GPU {gpu_id}.")
            try:
                info = self.run_episode(env, policy, episode, show_detail=show_detail)
            except Exception as e:
                self.logger.error(f"Episode {episode} failed with error: {e}.")
                self.logger.error(traceback.format_exc())
                self._write_error(episode, e)
                raise
            infos.append(info)
            if info_queue is not None:
                info_queue.put(info)

            # rebuilding pi0 periodically avoids GPU memory growth over long runs
            gc.collect()
            torch.cuda.empty_cache()
            if self.reload_every > 0 and (i + 1) % self.reload_every == 0 and i + 1 < len(episodes):
                del policy
                gc.collect()
                torch.cuda.empty_cache()
                with gpu_lock:
                    policy = self._build_policy()
        return infos

    def run_episode(self, env, policy, episode, show_detail=False):
        obs, _ = env.reset(seed=episode)
        instruction = env.unwrapped.get_language_instruction()
        is_final_subtask = env.unwrapped.is_final_subtask()

        policy.reset(instruction, seed=episode)
        if self.use_negative_image:
            self.negative_image_generator.reset()
        if show_detail:
            self.logger.info(f"Initial instruction: {instruction}")

        predicted_terminated, truncated = False, False
        timestep = 0
        frames = []
        step_infos = []

        image = get_image_from_maniskill2_obs_dict(env, obs)
        neg_image = self._get_negative_image(obs, instruction)
        frames.append(self._frame(image, neg_image))

        while not (predicted_terminated or truncated):
            proprio = obs['agent']['eef_pos']
            if self.method == 'vanilla':
                _, actions = policy.step(image, instruction, proprio=proprio)
            elif self.method == 'vanilla_perturbed':
                _, actions = policy.step(neg_image, instruction, proprio=proprio)
            elif self.method == 'sdn':
                _, actions = policy.sdn_step(image, neg_image, instruction, proprio=proprio)
            elif self.method == 'sdn_grounding':
                _, actions = policy.grounding_step(image, neg_image, instruction, proprio=proprio)
            elif self.method == 'sdn_smooth':
                _, actions = policy.smooth_step(image, instruction, proprio=proprio)
            else:
                raise ValueError(f"Unknown method {self.method}")

            # execute the whole action chunk
            for action in actions:
                obs, reward, success, truncated, info = env.step(
                    np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
                image = get_image_from_maniskill2_obs_dict(env, obs)

                is_final_subtask = env.unwrapped.is_final_subtask()
                timestep += 1
                step_infos.append(convert_numpy_or_torch_to_python(info))
                predicted_terminated = bool(action["terminate_episode"][0] > 0)
                if show_detail:
                    self.logger.info(f"Step {timestep}: {step_infos[-1]}")

                if predicted_terminated and not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

                new_instruction = env.unwrapped.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                    if show_detail:
                        self.logger.info(f"New instruction: {instruction}")

                neg_image = self._get_negative_image(obs, instruction)
                frames.append(self._frame(image, neg_image))

        info = summarize(step_infos)
        info.update(stat_first(step_infos))
        info.update(stat_final(step_infos))
        success = info['success']
        self.logger.info(f"Episode {episode} finished with success {success}.")
        if self.save_video:
            write_video(frames, f"{self.result_dir}/episode_{episode}_success_{success}.gif")
        return info

    def _get_negative_image(self, obs, instruction):
        if not self.use_negative_image:
            return None
        return self.negative_image_generator.generate(obs, instruction)

    @staticmethod
    def _frame(image, neg_image):
        return image if neg_image is None else tile_images([image, neg_image])

    def build_episode(self, gpu_id, show_detail):
        self._set_gpu(gpu_id)
        import simpler_env
        env = simpler_env.make(self.task)
        if self.use_negative_image:
            self._build_negative_image_generator(env, show_detail)
        with gpu_lock:
            policy = self._build_policy(show_detail)
        return env, policy

    def _set_gpu(self, gpu_id):
        """Must be called before any CUDA initialisation in the worker."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # listing devices early avoids a CUDA initialisation error in TensorFlow
        import tensorflow as tf
        tf.config.list_physical_devices("GPU")

    def _build_policy(self, show_detail=False):
        policy, config = build_policy(self.checkpoint, self.task, self.opts, self.method)
        reset_logging()
        self._build_logger(mode='a')
        if show_detail:
            self.logger.infos("Policy Config", config)
        return policy

    def _build_negative_image_generator(self, env, show_detail=False):
        from sdn.negatives import NegativeImageGenerator
        config = get_negative_image_config(self.opts)
        self.negative_image_generator = NegativeImageGenerator(env=env, **config)
        reset_logging()
        self._build_logger(mode='a')
        if show_detail:
            self.logger.infos("Negative Image Config", config)

    def _build_logger(self, mode='w'):
        self.logger = Logger(osp.join(self.result_dir, '000.log'), mode)

    def _set_result_dir(self):
        """Create the result dir. Returns False if this configuration has already finished."""
        self.result_dir = osp.join(self.result_root, self.method)
        if len(self.opts) > 0:
            self.result_dir = osp.join(self.result_dir, '--'.join(f'{k}={v}' for k, v in self.opts.items()))
        self.result_dir = osp.join(self.result_dir, self.task)

        if osp.exists(self.result_dir):
            logs = [f for f in os.listdir(self.result_dir) if f.startswith('000') and f.endswith('.log')]
            if any(f.startswith('000_success_') for f in logs):
                print(f"{self.result_dir} already finished, skipping.")
                return False
            print(f"{self.result_dir} exists but did not finish, starting over.")
            shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)
        return True

    def _write_error(self, episode, error):
        with open(osp.join(self.result_dir, f"000_episode_{episode}_error.log"), 'w') as f:
            f.write(str(error))
            traceback.print_exc(file=f)


def iterate_search_opts(search_opts):
    """{'a': [1, 2], 'b': 3} -> {'a': 1, 'b': 3}, {'a': 2, 'b': 3}"""
    keys = list(search_opts)
    values = [v if isinstance(v, list) else [v] for v in search_opts.values()]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def main(args):
    opts = parse_opts(args.opts)
    search_opts = parse_opts(args.search_opts)
    check_opts({**opts, **search_opts})

    grid = iterate_search_opts(search_opts) if search_opts else [{}]
    for search in grid:
        runner = ParallelRunner(method=args.method,
                                task=args.task,
                                checkpoint=args.checkpoint,
                                num_gpus=args.num_gpus,
                                result_root=args.result_root,
                                n_trajs=args.n_trajs,
                                reload_every=args.reload_every,
                                save_video=not args.no_video,
                                opts={**opts, **search})
        runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, default="sdn")
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--checkpoint", default="pretrained/open-pi-zero",
                        help="directory with the open-pi-zero checkpoints")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--result-root", default="./results")
    parser.add_argument("--n-trajs", type=int, default=50, help="number of rollouts (episode seeds 0..n-1)")
    parser.add_argument("--reload-every", type=int, default=8,
                        help="rebuild the policy every n episodes per GPU (0 disables)")
    parser.add_argument("--no-video", action="store_true", help="do not save per-episode GIFs")
    parser.add_argument("--opts", nargs="+", default=[],
                        help="config overrides as key value pairs, e.g. knn_k 6 top_m 3 negative_mode inpaint")
    parser.add_argument("--search-opts", nargs="+", default=[],
                        help="grid search as key comma-separated-values pairs, e.g. knn_k 6,10 top_m 3,5")
    main(parser.parse_args())
