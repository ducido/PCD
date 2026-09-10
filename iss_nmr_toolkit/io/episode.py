"""Prepared X-ICM episode and semantic-mask loaders."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from iss_nmr_toolkit.constants import DEFAULT_VIEWS, MASK_VIDEO_KEYS, VIEW_NAMES
from iss_nmr_toolkit.core.iss import parse_image


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _find_frame_dir(episode_dir: Path, view: str, obs_key: str) -> Path | None:
    candidates = [
        episode_dir / view,
        episode_dir / f"{view}_rgb",
        episode_dir / obs_key,
        episode_dir / "images" / view,
        episode_dir / "images" / f"{view}_rgb",
        episode_dir / "rgb" / view,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _list_images(frame_dir: Path) -> list[str]:
    paths: list[str] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(str(frame_dir / pattern)))
    paths = [path for path in paths if not Path(path).name.startswith(".")]
    return sorted(paths, key=natural_key)


def _load_state_npz(episode_dir: Path) -> dict[str, Any]:
    for name in ("states.npz", "observations.npz", "episode.npz"):
        path = episode_dir / name
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return {key: data[key] for key in data.files}
    return {}


def load_episode_directory(
    episode_dir: str | os.PathLike[str],
    *,
    views: dict[str, str] = DEFAULT_VIEWS,
    prompt: str | None = None,
    require_state: bool = False,
) -> list[dict[str, Any]]:
    """Load an exported episode directory into policy observations.

    Expected RGB layout can be any of:
    ``front/*.png``, ``front_rgb/*.png``, ``images/front/*.png``, or a folder
    named by the policy observation key such as ``exterior_image_1_left``.
    ``states.npz`` should contain ``joint_position`` and ``gripper_position``.
    ``prompt`` may also be stored there. Pi05 X-ICM uses state through the
    tokenized prompt transform, so set ``require_state=True`` for policy runs
    that should match the web-page ISS computation.
    """
    root = Path(episode_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {root}")

    frame_paths: dict[str, list[str]] = {}
    for view, obs_key in views.items():
        frame_dir = _find_frame_dir(root, view, obs_key)
        if frame_dir is None:
            raise FileNotFoundError(f"Missing RGB frame directory for view '{view}' under {root}")
        paths = _list_images(frame_dir)
        if not paths:
            raise FileNotFoundError(f"No RGB frames found in {frame_dir}")
        frame_paths[view] = paths

    lengths = {view: len(paths) for view, paths in frame_paths.items()}
    total_steps = min(lengths.values())
    if len(set(lengths.values())) != 1:
        print(f"Warning: view frame counts differ; truncating to {total_steps}: {lengths}")

    state = _load_state_npz(root)
    joint_positions = state.get("joint_position")
    gripper_positions = state.get("gripper_position")
    state_prompt = state.get("prompt")
    if require_state and (joint_positions is None or gripper_positions is None):
        raise FileNotFoundError(
            "Missing states.npz with joint_position and gripper_position. "
            "Pi05 X-ICM tokenizes state into the prompt, and the prepared X-ICM "
            "episode state must be aligned with every frame."
        )
    if prompt is None and state_prompt is not None:
        prompt = str(state_prompt.item() if hasattr(state_prompt, "item") else state_prompt)
    if prompt is None:
        prompt = root.parent.name.replace("_", " ")

    observations: list[dict[str, Any]] = []
    for step in tqdm(range(total_steps), desc="Loading episode frames"):
        obs: dict[str, Any] = {}
        for view, obs_key in views.items():
            obs[obs_key] = _read_rgb(frame_paths[view][step])

        if joint_positions is not None and step < len(joint_positions):
            obs["joint_position"] = np.asarray(joint_positions[step])
        else:
            obs["joint_position"] = np.zeros(7, dtype=np.float32)

        if gripper_positions is not None and step < len(gripper_positions):
            gripper = np.asarray(gripper_positions[step])
        else:
            gripper = np.zeros(1, dtype=np.float32)
        if gripper.ndim == 0:
            gripper = gripper[..., np.newaxis]
        obs["gripper_position"] = gripper
        obs["prompt"] = prompt
        observations.append(obs)

    return observations

def _load_episode_video(dataset_root: str | os.PathLike[str], episode: int, video_key: str) -> np.ndarray:
    """Decode ``{root}/videos/chunk-{chunk:03d}/{video_key}/episode_{episode:06d}.mp4``.

    The videos are AV1, which decord's bundled FFmpeg cannot decode, so decoding
    goes through pyav. Returns ``(length, height, width, 3)`` uint8 frames.
    """
    from gr00t.utils.video_utils import get_all_frames

    path = (
        Path(dataset_root)
        / "videos"
        / f"chunk-{int(episode) // 1000:03d}"
        / video_key
        / f"episode_{int(episode):06d}.mp4"
    )
    frames, _ = get_all_frames(str(path), video_backend="pyav")
    return frames


def load_masks(
    dataset_root: str | os.PathLike[str],
    episode: int,
    *,
    views: tuple[str, ...] = VIEW_NAMES,
    mask_video_keys: dict[str, str] = MASK_VIDEO_KEYS,
) -> dict[str, np.ndarray]:
    """Load per-view masks from the LeRobot mask videos of one episode.

    ``episode=1`` reads ``episode_000001.mp4``. Videos are lossy-compressed, so
    frames are reduced to one channel and thresholded back to a binary mask
    (``0=background/nuisance``, ``1=object of interest``).

    Returns one ``(length, height, width)`` uint8 array per view.
    """
    masks: dict[str, np.ndarray] = {}
    for view in views:
        frames = _load_episode_video(dataset_root, episode, mask_video_keys[view])
        masks[view] = (frames.max(axis=-1) > 127).astype(np.uint8)
    return masks


def load_episode_frames(
    dataset_root: str | os.PathLike[str],
    episode: int,
    *,
    views: dict[str, str] = DEFAULT_VIEWS,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Load the RGB frames of one LeRobot episode as per-step observations.

    Each view's observation key (``image``, ``wrist_image``) names the LeRobot
    video key ``observation.images.{obs_key}``. Returns one dict per step, keyed
    by observation key, which is what the ISS visualizers consume.
    """
    frames = {
        obs_key: _load_episode_video(dataset_root, episode, f"observation.images.{obs_key}")
        for obs_key in views.values()
    }
    lengths = {obs_key: len(view_frames) for obs_key, view_frames in frames.items()}
    total_steps = min(lengths.values())
    if len(set(lengths.values())) != 1:
        print(f"Warning: view frame counts differ; truncating to {total_steps}: {lengths}")

    observations: list[dict[str, Any]] = []
    for step in range(total_steps):
        obs: dict[str, Any] = {obs_key: parse_image(frames[obs_key][step]) for obs_key in frames}
        if prompt is not None:
            obs["prompt"] = prompt
        observations.append(obs)
    return observations
