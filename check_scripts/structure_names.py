from pathlib import Path
import json
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()


CONTOURS_ORIGIN_FOLDER = Path(os.getenv("CONTOURS_SOURCE"))


def list_all_structure_names():
    structure_counter = Counter()
    total_files = 0

    # Iterate over all vs_gk_* folders (t1 and t2)
    for folder in sorted(CONTOURS_ORIGIN_FOLDER.iterdir()):
        if not folder.is_dir():
            continue

        contour_json = folder / "contours.json"
        if not contour_json.is_file():
            continue

        total_files += 1

        try:
            data = json.loads(contour_json.read_text())
        except Exception as e:
            print(f"[ERROR] Failed to read {contour_json}: {e}")
            continue

        for structure in data:
            name = structure.get("structure_name")
            if name:
                structure_counter[name.strip()] += 1

    print(f"\nContours files processed: {total_files}")
    print(f"Unique structure_name entries: {len(structure_counter)}\n")

    # Print sorted by frequency (most common first)
    for name, count in structure_counter.most_common():
        print(f"{name:30s} : {count}")

    return structure_counter


if __name__ == "__main__":
    list_all_structure_names()
