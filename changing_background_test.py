"""Standalone background replacement, adapted from RL-ViGen.

Drop this file into any repo (SimplerEnv, ManiSkill, etc.). Only needs
numpy + opencv.

One frame in, one frame out:

    new_frame = replace_background(frame, background, mask)

Run `python change_background.py` for a working demo with a dummy mask.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

# Where the composited episode video is written (sibling of normal/bbox/inpainting).
OUT_DIR = Path("example_perturbations/bg")


# --------------------------------------------------------------------------- #
# The function
# --------------------------------------------------------------------------- #
def replace_background(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    mask_is_background: bool = True,
    feather: int = 0,
) -> np.ndarray:
    """Replace the background of a single frame.

    Args:
        frame:  uint8 image, (H, W, 3). The scene from your simulator.
        background: uint8 image, (H, W, 3). Auto-resized to the frame if needed.
        mask:   (H, W) array marking the background. Accepts bool, {0,1} or
                {0,255} ints, or a soft float array in [0, 1]. Auto-resized to
                the frame if needed.
        mask_is_background: True  -> mask marks pixels to REPLACE (default).
                            False -> mask marks the robot/objects; it's inverted.
        feather: odd kernel size to blur the mask edge. 0 = hard edge.
                 Try 3-5 at low resolution to avoid a jagged silhouette.

    Returns:
        uint8 (H, W, 3) frame with the background swapped.
    """
    frame = np.asarray(frame)
    assert frame.ndim == 3 and frame.shape[2] == 3, f"frame must be (H,W,3), got {frame.shape}"
    assert frame.dtype == np.uint8, f"frame must be uint8, got {frame.dtype}"

    H, W = frame.shape[:2]

    # --- background: resize to match the frame -----------------------------
    background = np.asarray(background)
    if background.shape[:2] != (H, W):
        background = cv2.resize(background, (W, H), interpolation=cv2.INTER_LINEAR)

    # --- mask: coerce to float (H, W, 1) in [0, 1] -------------------------
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0] if m.shape[-1] == 1 else m[0]
    is_hard = m.dtype == np.bool_ or np.issubdtype(m.dtype, np.integer)
    m = m.astype(np.float32)
    if is_hard and m.max() > 1.0:
        m /= 255.0
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H),
                       interpolation=cv2.INTER_NEAREST if is_hard else cv2.INTER_LINEAR)
    if not mask_is_background:
        m = 1.0 - m
    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    alpha = np.clip(m, 0.0, 1.0)[..., None]

    # --- composite ---------------------------------------------------------
    out = frame.astype(np.float32) * (1 - alpha) + background.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def load_background(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read a background image file as a (H, W, 3) uint8 RGB array.

    Args:
        path: image file (.png/.jpg/...).
        size: optional (H, W) to resize to.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # cv2 reads BGR
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# Example
# --------------------------------------------------------------------------- #
def _dummy_scene(H=256, W=256):
    """Fake simulator output: a 'robot arm' + a 'cube' on a table.

    Returns (frame, bg_mask). Replace this with your real frame + real mask.
    """
    frame = np.full((H, W, 3), 120, np.uint8)          # gray backdrop
    frame[int(H * 0.72):] = (95, 80, 70)               # table

    fg = np.zeros((H, W), np.uint8)                    # foreground mask

    x = int(W * 0.35)
    cv2.rectangle(frame, (x, 40), (x + 26, 190), (200, 200, 205), -1)   # arm
    cv2.rectangle(fg, (x, 40), (x + 26, 190), 1, -1)

    cx = int(W * 0.60)
    cv2.rectangle(frame, (cx, 160), (cx + 45, 205), (30, 90, 200), -1)  # cube
    cv2.rectangle(fg, (cx, 160), (cx + 45, 205), 1, -1)

    # NOTE: here the table counts as foreground; only the backdrop is replaced.
    fg[int(H * 0.72):] = 1

    bg_mask = fg == 0                                  # True = background
    return frame, bg_mask


def _dummy_background(H=256, W=256):
    """Fake background image: a color gradient."""
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    bg = np.zeros((H, W, 3), np.uint8)
    bg[..., 0] = (127 * (1 + np.sin(3 * xx))).astype(np.uint8)
    bg[..., 1] = (127 * (1 + np.sin(3 * yy))).astype(np.uint8)
    bg[..., 2] = (127 * (1 + np.cos(3 * (xx + yy)))).astype(np.uint8)
    return bg

def random_low_frequency_background(
    frame: np.ndarray,
    noise_scale: float = 45.0,
    blur_kernel: int = 51,
    base_strength: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a spatially correlated random background.

    The random field is strongly blurred so that the perturbation
    changes large-scale background appearance rather than individual
    pixels.

    Args:
        frame:
            Original RGB uint8 frame with shape (H, W, 3).

        noise_scale:
            Standard deviation of the random field.

        blur_kernel:
            Gaussian blur kernel size. Larger values produce smoother
            spatial variations.

        base_strength:
            Controls how much the generated random field deviates from
            the original background statistics.

        rng:
            Optional NumPy random generator.

    Returns:
        RGB uint8 background image with shape (H, W, 3).
    """

    if rng is None:
        rng = np.random.default_rng()

    frame_float = frame.astype(np.float32)

    H, W = frame.shape[:2]

    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=(H, W, 3),
    ).astype(np.float32)

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    noise = cv2.GaussianBlur(
        noise,
        (blur_kernel, blur_kernel),
        0,
    )

    random_background = (
        frame_float
        + base_strength * noise
    )

    random_background = np.clip(
        random_background,
        0.0,
        255.0,
    )

    return random_background.astype(np.uint8)

def random_background_appearance(
    frame: np.ndarray,
    brightness_range: Tuple[float, float] = (0.75, 1.25),
    contrast_range: Tuple[float, float] = (0.80, 1.20),
    color_jitter: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Randomize global appearance of the background.

    Brightness, contrast, and per-channel color statistics are changed
    while preserving the spatial structure of the original image.

    Args:
        frame:
            Original RGB uint8 frame.

        brightness_range:
            Multiplicative brightness range.

        contrast_range:
            Multiplicative contrast range.

        color_jitter:
            Maximum relative per-channel color perturbation.

        rng:
            Optional NumPy random generator.

    Returns:
        RGB uint8 randomized background.
    """

    if rng is None:
        rng = np.random.default_rng()

    image = frame.astype(np.float32)

    brightness = rng.uniform(
        brightness_range[0],
        brightness_range[1],
    )

    contrast = rng.uniform(
        contrast_range[0],
        contrast_range[1],
    )

    channel_scale = rng.uniform(
        1.0 - color_jitter,
        1.0 + color_jitter,
        size=(1, 1, 3),
    )

    mean = image.mean(
        axis=(0, 1),
        keepdims=True,
    )

    randomized = (
        (image - mean)
        * contrast
        + mean
    )

    randomized = (
        randomized
        * brightness
        * channel_scale
    )

    randomized = np.clip(
        randomized,
        0.0,
        255.0,
    )

    return randomized.astype(np.uint8)
    
def main():
    H = W = 256
    out_dir = "bg_demo_out"
    os.makedirs(out_dir, exist_ok=True)

    # 1. your frame + your mask  (swap these two lines for the real thing)
    data_root = '/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/Isaac-GR00T/eval_logs/gg_robot/baseline_nenvs1_eps20_ah1/google_robot_pick_coke_can/videos'
    ep = 0
    video_path = os.path.join(data_root, f"episode_{ep}_rgb.gif")
    contrast_video_path = os.path.join(data_root, f"episode_{ep}_ooi_mask.gif")
    rgb_frames = imageio.mimread(video_path)
    contrast_frames = imageio.mimread(contrast_video_path)

    change_bg_frames = []
    for step in range(len(rgb_frames)):

        if step > 3:
            break
        frame = rgb_frames[step][:,:,:3]
        # collapse to (H, W): replace_background can't read a 3-channel mask
        mask = ~contrast_frames[step][:,:,:3].max(axis=-1)

        background = _dummy_background(H, W)

        # 3. the one call
        new_frame = replace_background(frame, background, mask)

        change_bg_frames.append(new_frame)

    # 4. save the composited episode as a video
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"episode_{ep}.gif"
    imageio.mimsave(out_path, change_bg_frames, fps=15, loop=0)
    print(f"Saved {len(change_bg_frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
