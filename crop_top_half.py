"""
Crop every mp4 in example_base_ours/<task>/*.mp4 to keep only the top half
(vertical crop: height // 2, full width), overwriting each file in place.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "example_base_ours"


def crop_top_half(mp4_path: Path) -> None:
    reader = imageio.get_reader(str(mp4_path))
    meta = reader.get_meta_data()
    fps = meta.get("fps", 10)

    tmp_path = mp4_path.with_suffix(".tmp.mp4")
    writer = imageio.get_writer(str(tmp_path), fps=fps, codec="libx264", quality=8)
    try:
        for frame in reader:
            h = frame.shape[0]
            top = frame[: h // 2]
            writer.append_data(top)
    finally:
        writer.close()
        reader.close()

    tmp_path.replace(mp4_path)


def main():
    if not OUT_ROOT.is_dir():
        sys.exit(f"Not found: {OUT_ROOT}")

    mp4_files = sorted(OUT_ROOT.glob("*/*.mp4"))
    print(f"Found {len(mp4_files)} mp4 files")

    for mp4_path in mp4_files:
        crop_top_half(mp4_path)
        print(f"cropped: {mp4_path.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
