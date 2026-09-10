"""Extract the bottom half of the first frame of each perturbation GIF.

Each GIF stacks two views vertically (original on top, perturbed below). We keep
only the bottom half of frame 0 and save it as <task_name>.png.
"""

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_SRC = Path("results_4gpu_new/default/baseline_inpainting_object/open-pi-zero")
DEFAULT_DST = Path("example_perturbations/inpainting")

# The specific episode picked for each task (task dirs hold many episodes).
EPISODES = {
    "google_robot_close_drawer": "episode_1_success_False.gif",
    "google_robot_move_near": "episode_0_success_False.gif",
    "google_robot_open_drawer": "episode_0_success_False.gif",
    "google_robot_pick_coke_can": "episode_0_success_False.gif",
    "google_robot_place_apple_in_closed_top_drawer": "episode_1_success_False.gif",
    "widowx_carrot_on_plate": "episode_0_success_False.gif",
    "widowx_put_eggplant_in_basket": "episode_0_success_False.gif",
    "widowx_spoon_on_towel": "episode_0_success_False.gif",
    "widowx_stack_cube": "episode_0_success_False.gif",
}


def extract_bottom_first_frame(gif_path: Path, out_path: Path) -> None:
    with Image.open(gif_path) as im:
        im.seek(0)
        frame = im.convert("RGB")
        w, h = frame.size
        bottom = frame.crop((0, h // 2, w, h))
        bottom.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Directory containing one sub-directory per task")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help="Output directory for the cropped frames")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    for task_name, episode in EPISODES.items():
        gif_path = args.src / task_name / episode
        if not gif_path.exists():
            print(f"MISSING: {gif_path}")
            continue
        out_path = args.dst / f"{task_name}.png"
        extract_bottom_first_frame(gif_path, out_path)
        print(f"{gif_path} -> {out_path}")


if __name__ == "__main__":
    main()
