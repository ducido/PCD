"""
Build a folder of paired demo videos for the website:

    example_base_ours/<task>/base_fail.mp4    -- a failure episode from the baseline
    example_base_ours/<task>/ours_success.mp4 -- a success episode from our method

Baseline episode gifs encode success/failure directly in the filename:
    episode_<N>_success_<True|False>.gif

"Ours" episode gifs do not encode success in the filename (episode_<N>.gif); the
per-episode outcome is only recorded in the run log (000_success_<rate>.log) as
lines like "Episode <N> finished with success True.". So for "ours" we parse the
log to find a successful episode index and use that episode's gif.

All gifs are converted to mp4 with imageio (using the bundled imageio-ffmpeg
binary), since there is no system ffmpeg on this machine.
"""
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parent

BASELINE_ROOT = ROOT / "results_4gpu_rebuttal/default/baseline_masking_random_bbox_object/open-pi-zero"
OURS_ROOT = ROOT / (
    "results_4gpu_new/default/knn_topK_long_delta_motion_4_bbox/open-pi-zero/"
    "by=grounded_sam_tracking--alpha=0.2--num_repeats=36--knn_k=6--top_k=3"
)

OUT_ROOT = ROOT / "example_base_ours"

EPISODE_LOG_RE = re.compile(r"Episode (\d+) finished with success (True|False)")
BASELINE_GIF_RE = re.compile(r"episode_(\d+)_success_(True|False)\.gif$")


def get_gif_fps(gif_path: Path, default: float = 10.0) -> float:
    durations = []
    with Image.open(gif_path) as im:
        try:
            frame = 0
            while True:
                im.seek(frame)
                durations.append(im.info.get("duration", 100))
                frame += 1
        except EOFError:
            pass
    if durations:
        avg_ms = sum(durations) / len(durations)
        if avg_ms > 0:
            return 1000.0 / avg_ms
    return default


def gif_to_mp4(gif_path: Path, mp4_path: Path) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    fps = get_gif_fps(gif_path)
    frames = imageio.mimread(str(gif_path), memtest=False)
    writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            if frame.ndim == 2:
                frame = frame[..., None].repeat(3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            writer.append_data(frame)
    finally:
        writer.close()


def find_baseline_failure(task_dir: Path):
    candidates = []
    for gif in task_dir.glob("episode_*_success_False.gif"):
        m = BASELINE_GIF_RE.search(gif.name)
        if m:
            candidates.append((int(m.group(1)), gif))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_ours_success(task_dir: Path):
    success_episodes = set()
    for log_file in sorted(task_dir.glob("*.log")):
        text = log_file.read_text(errors="ignore")
        for m in EPISODE_LOG_RE.finditer(text):
            ep, success = int(m.group(1)), m.group(2) == "True"
            if success:
                success_episodes.add(ep)
    for ep in sorted(success_episodes):
        gif = task_dir / f"episode_{ep}.gif"
        if gif.exists():
            return gif
    return None


def main():
    if not BASELINE_ROOT.is_dir():
        sys.exit(f"Baseline root not found: {BASELINE_ROOT}")
    if not OURS_ROOT.is_dir():
        sys.exit(f"Ours root not found: {OURS_ROOT}")

    baseline_tasks = {d.name for d in BASELINE_ROOT.iterdir() if d.is_dir()}
    ours_tasks = {d.name for d in OURS_ROOT.iterdir() if d.is_dir()}
    common_tasks = sorted(baseline_tasks & ours_tasks)

    print(f"Common tasks ({len(common_tasks)}): {common_tasks}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    report = []
    for task in common_tasks:
        base_dir = BASELINE_ROOT / task
        ours_dir = OURS_ROOT / task

        fail_gif = find_baseline_failure(base_dir)
        success_gif = find_ours_success(ours_dir)

        status = {"task": task}

        if fail_gif is None:
            status["base_fail"] = "MISSING (no failing baseline episode found)"
        else:
            out_path = OUT_ROOT / task / "base_fail.mp4"
            gif_to_mp4(fail_gif, out_path)
            status["base_fail"] = f"{fail_gif.name} -> {out_path.relative_to(ROOT)}"

        if success_gif is None:
            status["ours_success"] = "MISSING (no successful 'ours' episode found)"
        else:
            out_path = OUT_ROOT / task / "ours_success.mp4"
            gif_to_mp4(success_gif, out_path)
            status["ours_success"] = f"{success_gif.name} -> {out_path.relative_to(ROOT)}"

        report.append(status)
        print(f"[{task}] base_fail: {status['base_fail']}")
        print(f"[{task}] ours_success: {status['ours_success']}")

    missing = [r for r in report if "MISSING" in r["base_fail"] or "MISSING" in r["ours_success"]]
    print("\n=== Summary ===")
    print(f"Total tasks: {len(report)}, complete pairs: {len(report) - len(missing)}, incomplete: {len(missing)}")
    if missing:
        print("Incomplete tasks:")
        for r in missing:
            print(f"  - {r['task']}: base_fail={r['base_fail']}, ours_success={r['ours_success']}")


if __name__ == "__main__":
    main()
