from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR"))
imgs_root  = BASE_DIR/"images"
msks_root  = BASE_DIR/"masks"
lbls_root  = BASE_DIR/"labels"

MIN_POINTS = 3


def count_files_in_subdirs(root: Path):
    total = 0
    for sub in root.glob("VS-SEG-*"):
        if sub.is_dir():
            total += len(list(sub.glob("*.*")))
    return total


def validate_label_file(text: str) -> list:
    """
    Validates a single label file's contents for geometric correctness.
    Returns a list of error descriptions. Empty list means valid.
    """
    errors = []

    for line_num, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue

        coords = parts[1:]

        if len(coords) < MIN_POINTS * 2:
            errors.append(
                f"  Line {line_num}: too few coordinates "
                f"({len(coords)} values, expected at least {MIN_POINTS * 2})"
            )
            continue

        if len(coords) % 2 != 0:
            errors.append(
                f"  Line {line_num}: odd number of coordinate values ({len(coords)})"
            )

        invalid_values = []
        for i, val in enumerate(coords):
            try:
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    invalid_values.append((i, v))
            except ValueError:
                errors.append(
                    f"  Line {line_num}: non-numeric value at position {i}: '{val}'"
                )

        if invalid_values:
            examples = ", ".join(f"pos {i}={v:.4f}" for i, v in invalid_values[:5])
            errors.append(
                f"  Line {line_num}: {len(invalid_values)} coordinate(s) outside [0, 1] "
                f"-- examples: {examples}"
            )

    return errors


def main():
    
    # CHECK 1: File-level consistency                                

    print("CHECK 1: File-level consistency\n")

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
                print(f"[WARN] Missing image: {img_path}")
            if not lbl_path.exists():
                missing_labels[pid].append(stem)

    total_missing = sum(len(v) for v in missing_labels.values())
    print(f"MISSING labels for {total_missing} masks\n")
    for pid, stems in missing_labels.items():
        print(pid, "->", len(stems), "items")
        for s in stems[:10]:
            print("  -", s)
        if len(stems) > 10:
            print("  ...")

    print("\nFile counts per directory:")
    paths = {"Images": imgs_root, "Labels": lbls_root, "Masks": msks_root}
    for name, path in paths.items():
        if not path.exists():
            print(f"{name:<10}: folder not found")
            continue
        count = count_files_in_subdirs(path)
        print(f"{name:<10}: {count}")

    # CHECK 2: Geometric validity of label files                      
       
    print("\nCHECK 2: Geometric validity of label files\n")

    if not lbls_root.exists():
        print(f"ERROR: Labels directory not found: {lbls_root}")
        return

    patient_errors = defaultdict(list)
    total_files = 0
    total_empty = 0
    total_invalid = 0

    patients = sorted(
        p for p in lbls_root.iterdir()
        if p.is_dir() and p.name.startswith("VS-SEG-")
    )

    for patient_dir in patients:
        pid = patient_dir.name
        for lbl in sorted(patient_dir.glob("*.txt")):
            total_files += 1
            text = lbl.read_text().strip()
            if not text:
                total_empty += 1
                continue
            errors = validate_label_file(text)
            if errors:
                total_invalid += 1
                patient_errors[pid].append((lbl.name, errors))

    print(f"Total label files checked : {total_files}")
    print(f"Empty label files (valid) : {total_empty}")
    print(f"Files with annotations    : {total_files - total_empty}")
    print(f"Files with errors         : {total_invalid}")

    if patient_errors:
        print(f"\nERRORS FOUND in {len(patient_errors)} patient(s):\n")
        for pid, file_errors in sorted(patient_errors.items()):
            print(f"[{pid}]")
            for filename, errors in file_errors:
                print(f"  {filename}:")
                for err in errors:
                    print(err)
            print()
    else:
        print("\nAll annotation files passed geometric validation.")
        print("No out-of-range coordinates or degenerate polygons found.")

    print("\nSUMMARY:")
    print(f"File-level consistency : {'PASSED' if total_missing == 0 else 'FAILED'}")
    print(f"Geometric validity     : {'PASSED' if total_invalid == 0 else 'FAILED'}")


if __name__ == "__main__":
    main()
