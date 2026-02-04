# build_dataset.py
from pathlib import Path
import os
import json
import math
import random
import shutil
import re
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR"))          
DATASET_DIR = Path(os.getenv("DATASET_DIR"))   

SPLIT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

USE_HARDLINKS = False  


def gather_file(src: Path, dst: Path):
    """
    Copy or hardlink a file to the destination path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if USE_HARDLINKS:
        try:
            if dst.exists():
                dst.unlink()
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    img_root = BASE_DIR / "images"
    lbl_root = BASE_DIR / "labels"
    msk_root = BASE_DIR / "masks"

    patients = sorted(
        p.name for p in img_root.iterdir()
        if p.is_dir() and re.match(r"^VS-SEG-\d{3}$", p.name)
    )

    if not patients:
        print("No patients found in base directory.")
        return

    random.seed(SPLIT_SEED)
    random.shuffle(patients)

    n = len(patients)
    n_train = math.floor(n * TRAIN_FRAC)
    n_val = math.floor(n * VAL_FRAC)

    splits = {
        "train": patients[:n_train],
        "valid": patients[n_train:n_train + n_val],
        "test":  patients[n_train + n_val:],
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "splits.json").write_text(json.dumps(splits, indent=2))

    print("Dataset split:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} patients")

    for split, pids in splits.items():
        print(f"\n-- Building {split} set ({len(pids)} patients) --")

        for pid in pids:
            img_dir = img_root / pid
            lbl_dir = lbl_root / pid
            msk_dir = msk_root / pid

            for img in sorted(img_dir.glob("*.png")):
                stem = img.stem

                dst_img = DATASET_DIR / split / "images" / f"{pid}_{img.name}"
                gather_file(img, dst_img)

                lbl = lbl_dir / f"{stem}.txt"
                dst_lbl = DATASET_DIR / split / "labels" / f"{pid}_{stem}.txt"

                if lbl.is_file():
                    gather_file(lbl, dst_lbl)
                else:
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    dst_lbl.write_text("")

                if msk_dir.is_dir():
                    msk = msk_dir / f"{stem}.png"
                    if msk.is_file():
                        dst_msk = DATASET_DIR / split / "masks" / f"{pid}_{img.name}"
                        gather_file(msk, dst_msk)

    print("\nDataset successfully built at:", DATASET_DIR)


if __name__ == "__main__":
    main()
