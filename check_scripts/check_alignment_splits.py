from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = Path(os.getenv("DATASET_DIR"))

SPLITS = ["train", "valid", "test"]


def check_split(split_name):
    print(f"\n=== Checking split: {split_name} ===")

    img_root = DATASET_DIR / split_name / "images"
    lbl_root = DATASET_DIR / split_name / "labels"
    msk_root = DATASET_DIR / split_name / "masks"

    missing_labels = []
    missing_masks = []
    missing_images_from_masks = []
    missing_images_from_labels = []

    images = sorted(img_root.glob("*.png"))
    labels = sorted(lbl_root.glob("*.txt"))
    masks  = sorted(msk_root.glob("*.png"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    mask_stems  = {p.stem for p in masks}

    # Image -> Label
    for stem in image_stems:
        if stem not in label_stems:
            missing_labels.append(stem)

    # Image -> Mask
    for stem in image_stems:
        if stem not in mask_stems:
            missing_masks.append(stem)

    # Mask -> Image
    for stem in mask_stems:
        if stem not in image_stems:
            missing_images_from_masks.append(stem)

    # Label -> Image
    for stem in label_stems:
        if stem not in image_stems:
            missing_images_from_labels.append(stem)

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")
    print(f"Masks  : {len(masks)}")

    print(f"\nMissing labels for images : {len(missing_labels)}")
    print(f"Missing masks for images  : {len(missing_masks)}")
    print(f"Masks without image       : {len(missing_images_from_masks)}")
    print(f"Labels without image      : {len(missing_images_from_labels)}")

    return {
        "missing_labels": missing_labels,
        "missing_masks": missing_masks,
        "orphan_masks": missing_images_from_masks,
        "orphan_labels": missing_images_from_labels,
    }


def main():
    print("\nChecking dataset alignment after split...\n")

    for split in SPLITS:
        split_path = DATASET_DIR / split
        if not split_path.exists():
            print(f"[WARN] Split folder not found: {split}")
            continue
        check_split(split)


if __name__ == "__main__":
    main()
