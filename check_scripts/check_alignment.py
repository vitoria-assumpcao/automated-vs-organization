from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR"))
imgs_root  = BASE_DIR/"images"
msks_root  = BASE_DIR/"masks"
lbls_root  = BASE_DIR/"labels"

missing_labels = defaultdict(list)

for pid_dir in sorted(msks_root.iterdir()):
    if not pid_dir.is_dir(): 
        continue
    pid = pid_dir.name
    for msk in sorted(pid_dir.glob("*.png")):
        stem = msk.stem  
        img_path = imgs_root/pid/f"{stem}.png"
        lbl_path = lbls_root/pid/f"{stem}.txt"
        if not img_path.exists():
            print(f"[WARN] Sem imagem: {img_path}")
        if not lbl_path.exists():
            missing_labels[pid].append(stem)

total_missing = sum(len(v) for v in missing_labels.values())
print(f"\nMISSING labels from {total_missing} masks\n")
for pid, stems in missing_labels.items():
    print(pid, "->", len(stems), "itens")
    for s in stems[:10]:
        print("  -", s)
    if len(stems) > 10:
        print("  ...")


def count_files_in_subdirs(root: Path):
    total = 0
    for sub in root.glob("VS-SEG-*"):
        if sub.is_dir():
            total += len(list(sub.glob("*.*")))
    return total

def main():
    paths = {
        "Images": BASE_DIR / "images",
        "Labels": BASE_DIR / "labels",
        "Masks":  BASE_DIR / "masks",
    }

    print("Counting files in base_vs:\n")
    for name, path in paths.items():
        if not path.exists():
            print(f"{name:<10}: folder not found")
            continue
        count = count_files_in_subdirs(path)
        print(f"{name:<10}: {count}")

if __name__ == "__main__":
    main()