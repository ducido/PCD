import os
import random
import requests


OUTPUT_DIR = "./hypersim"
TARGET_IMAGES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()

output_names = list(range(1, TARGET_IMAGES + 1))
random.shuffle(output_names)

downloaded = 0

for xx in range(1, 11):
    for yy in range(1, 11):
        if downloaded >= TARGET_IMAGES:
            break

        source_url = (
            f"https://huggingface.co/datasets/ritianyu/Hypersim/"
            f"resolve/main/ai_{xx:03d}_{yy:03d}/images/"
            f"scene_cam_00_final_preview/frame.0000.color.jpg"
        )

        try:
            response = session.get(
                source_url,
                timeout=30
            )

            if response.status_code == 404:
                print(
                    f"SKIP: ai_{xx:03d}_{yy:03d} "
                    f"(not found)"
                )
                continue

            response.raise_for_status()

            output_name = output_names[downloaded]
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{output_name:03d}.jpg"
            )

            with open(output_path, "wb") as f:
                f.write(response.content)

            downloaded += 1

            print(
                f"[{downloaded:3d}/{TARGET_IMAGES}] "
                f"ai_{xx:03d}_{yy:03d} -> "
                f"{output_name:03d}.jpg"
            )

        except requests.RequestException as e:
            print(
                f"ERROR: ai_{xx:03d}_{yy:03d}: {e}"
            )

    if downloaded >= TARGET_IMAGES:
        break


print()
print(
    f"Done! Downloaded {downloaded} images "
    f"to {OUTPUT_DIR}"
)