"""Extract the first frame of each bbox-perturbation GIF, split into halves.

Each GIF stacks two views vertically: the unperturbed observation on top and the
bbox-perturbed one below. Frame 0 is split and saved as <task_name>.png into
example_perturbations/normal (top) and example_perturbations/bbox (bottom).
"""

import argparse
from pathlib import Path

from PIL import Image

ROOT = Path("results_4gpu_new/default/knn_topK_long_delta_motion_4_bbox/open-pi-zero")
CFG_K10 = "by=grounded_sam_tracking--alpha=0.2--num_repeats=24--knn_k=10--top_k=3"
CFG_K6 = "by=grounded_sam_tracking--alpha=0.2--num_repeats=24--knn_k=6--top_k=3"

DEFAULT_TOP_DST = Path("example_perturbations/normal")
DEFAULT_BOTTOM_DST = Path("example_perturbations/bbox")

# task_name -> (config dir, episode file)
EPISODES = {
    "google_robot_close_drawer": (CFG_K10, "episode_5.gif"),
    "google_robot_move_near": (CFG_K10, "episode_0.gif"),
    "google_robot_open_drawer": (CFG_K10, "episode_2.gif"),
    "google_robot_pick_coke_can": (CFG_K6, "episode_2.gif"),
    "google_robot_place_apple_in_closed_top_drawer": (CFG_K6, "episode_3.gif"),
    "widowx_carrot_on_plate": (CFG_K6, "episode_7.gif"),
    "widowx_put_eggplant_in_basket": (CFG_K6, "episode_31.gif"),
    "widowx_spoon_on_towel": (CFG_K6, "episode_47.gif"),
    "widowx_stack_cube": (CFG_K6, "episode_14.gif"),
}


def split_first_frame(gif_path: Path, top_path: Path, bottom_path: Path) -> None:
    with Image.open(gif_path) as im:
        im.seek(0)
        frame = im.convert("RGB")
        w, h = frame.size
        frame.crop((0, 0, w, h // 2)).save(top_path)
        frame.crop((0, h // 2, w, h)).save(bottom_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT,
                        help="Root holding the <config>/<task>/<episode>.gif tree")
    parser.add_argument("--top-dst", type=Path, default=DEFAULT_TOP_DST,
                        help="Output directory for the top (unperturbed) halves")
    parser.add_argument("--bottom-dst", type=Path, default=DEFAULT_BOTTOM_DST,
                        help="Output directory for the bottom (perturbed) halves")
    args = parser.parse_args()

    args.top_dst.mkdir(parents=True, exist_ok=True)
    args.bottom_dst.mkdir(parents=True, exist_ok=True)

    for task_name, (cfg, episode) in EPISODES.items():
        gif_path = args.src / cfg / task_name / episode
        if not gif_path.exists():
            print(f"MISSING: {gif_path}")
            continue
        top_path = args.top_dst / f"{task_name}.png"
        bottom_path = args.bottom_dst / f"{task_name}.png"
        split_first_frame(gif_path, top_path, bottom_path)
        print(f"{gif_path}\n  top    -> {top_path}\n  bottom -> {bottom_path}")


if __name__ == "__main__":
    main()
