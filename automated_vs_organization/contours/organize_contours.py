from pathlib import Path
import shutil
import re
import os
from dotenv import load_dotenv

load_dotenv()

CONTOURS_ORIGIN_FOLDER = Path(os.getenv("CONTOURS_ORIGIN_FOLDER"))
CONTOURS_SOURCE = Path(os.getenv("CONTOURS_SOURCE"))
CONTOUR_JSON_FILENAME = "contours.json"

# Regular expression to identify T1 and T2 contour folders
CONTOUR_PATTERN = re.compile(r"^vs_gk_(\d+)_(t1|t2)$", re.IGNORECASE)


def organize_contours():
    """
    Organizes T1 and T2 tumour contour files from the TCIA Vestibular Schwannoma dataset.

    The script:
    - Traverses contour folders named as vs_gk_<patient_number>_t1 or vs_gk_<patient_number>_t2
    - Maps them to patient IDs in the format VS-SEG-###
    - Copies the contours.json file into a patient-wise directory structure
    """

    print("== Organizing T1 and T2 contour files by patient ==")
    print(f"Source      : {CONTOURS_ORIGIN_FOLDER}")
    print(f"Destination : {CONTOURS_SOURCE}\n")

    if not CONTOURS_ORIGIN_FOLDER.is_dir():
        print(f"ERROR: source directory not found: {CONTOURS_ORIGIN_FOLDER}")
        return

    CONTOURS_SOURCE.mkdir(parents=True, exist_ok=True)

    copied_files = 0

    for folder in sorted(CONTOURS_ORIGIN_FOLDER.iterdir()):
        if not folder.is_dir():
            continue

        match = CONTOUR_PATTERN.match(folder.name)
        if not match:
            continue

        patient_number = int(match.group(1))
        modality = match.group(2).lower()
        patient_id = f"VS-SEG-{patient_number:03d}"

        patient_dest = CONTOURS_SOURCE / patient_id
        patient_dest.mkdir(parents=True, exist_ok=True)

        source_json = folder / CONTOUR_JSON_FILENAME

        if source_json.is_file():
            try:
                destination_json = patient_dest / f"{modality}_contours.json"
                shutil.copy2(source_json, destination_json)

                print(f"[OK] {folder.name} -> {destination_json}")
                copied_files += 1

            except Exception as e:
                print(f"[ERROR] Failed to copy '{source_json}': {e}")
        else:
            print(
                f"[WARNING] '{CONTOUR_JSON_FILENAME}' not found in: {folder}"
            )

    print("\n== DONE ==")
    print(f"T1 contour files copied: {copied_files}")
    print(f"Output folder          : {CONTOURS_SOURCE}")

if __name__ == "__main__":
    organize_contours()
