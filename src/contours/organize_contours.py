from pathlib import Path
import shutil
import os
from dotenv import load_dotenv

from src.config import CONTOUR_FOLDER_REGEX, PATIENT_ID_TEMPLATE, contour_json_filename

load_dotenv()

CONTOURS_ORIGIN_FOLDER = Path(os.getenv("CONTOURS_ORIGIN_FOLDER"))
CONTOURS_SOURCE = Path(os.getenv("CONTOURS_SOURCE"))
CONTOUR_JSON_FILENAME = "contours.json"


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

        match = CONTOUR_FOLDER_REGEX.match(folder.name)
        if not match:
            continue

        patient_number = int(match.group(1))
        modality = match.group(2).lower()
        patient_id = PATIENT_ID_TEMPLATE.format(patient_number)

        patient_dest = CONTOURS_SOURCE / patient_id
        patient_dest.mkdir(parents=True, exist_ok=True)

        source_json = folder / CONTOUR_JSON_FILENAME

        if source_json.is_file():
            try:
                destination_json = patient_dest / contour_json_filename(modality)
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
    print(f"Contour files copied: {copied_files}")
    print(f"Output folder          : {CONTOURS_SOURCE}")

if __name__ == "__main__":
    organize_contours()
