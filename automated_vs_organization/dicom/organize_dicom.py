from pathlib import Path
import shutil
import re
import os
from dotenv import load_dotenv

load_dotenv()


DICOM_ORIGIN_FOLDER  = Path(os.getenv("DICOM_ORIGIN_FOLDER"))
DICOM_SOURCE = Path(os.getenv("DICOM_SOURCE"))


FILTRO_T1_REGEX = re.compile(r"t1", re.IGNORECASE)
FILTRO_T2_REGEX = re.compile(r"t2", re.IGNORECASE)
DRY_RUN = False

def is_probable_dicom(p: Path) -> bool:
    """
    Heuristic function to identify whether a file is likely a DICOM file.
    Criteria:
    - File extension is '.dcm' or empty
    - Optional check for the 'DICM' signature at byte offset 128
    """
    if not p.is_file():
        return False
    ext = p.suffix.lower()
    if ext == ".dcm" or ext == "":
        try:
            with p.open("rb") as f:
                f.seek(128)
                if f.read(4) == b"DICM":
                    return True
        except Exception:
            return True  
        return True
    if ext in {".xml", ".json", ".txt", ".csv", ".zip", ".gz"}:
        return False
    return False

def organize_dicom():
    """
    Traverses the Vestibular Schwannoma SEG dataset structure,
    identifies T1- and T2-weighted MRI series using directory names,
    and copies the corresponding DICOM files into a patient-wise
    and modality-wise organized directory structure.
    """
    print("== Organizing T1 and T2 DICOM files by patient ==")
    print(f"Source      : {DICOM_ORIGIN_FOLDER}")
    print(f"Destination : {DICOM_SOURCE}\n")

    if not DICOM_ORIGIN_FOLDER.is_dir():
        print(f"ERROR: source directory not found: {DICOM_ORIGIN_FOLDER}")
        return

    DICOM_SOURCE.mkdir(parents=True, exist_ok=True)
    standard_patient = re.compile(r"^VS-SEG-\d{3}$", re.IGNORECASE)

    patients = sorted([
        p for p in DICOM_ORIGIN_FOLDER.iterdir()
        if p.is_dir() and standard_patient.match(p.name)
    ])

    if not patients:
        print("WARNING: No VS-SEG-### patient folders found.")
        return

    total_copy = 0
    total_t1_series = 0
    total_t2_series = 0

    for folder_patient in patients:
        patient_id = folder_patient.name

        patient_dest = DICOM_SOURCE / patient_id
        patient_dest.mkdir(parents=True, exist_ok=True)
        
        subfolders = sorted([
            sp for sp in folder_patient.iterdir() if sp.is_dir()
        ])

        for sub in subfolders:
            # Some patients have an extra directory level for series
            series = sorted([s for s in sub.iterdir() if s.is_dir()]) or [sub]

            for serie in series:
                if FILTRO_T1_REGEX.search(serie.name):
                    modality = "t1"
                    total_t1_series += 1
                elif FILTRO_T2_REGEX.search(serie.name):
                    modality = "t2"
                    total_t2_series += 1
                else:
                    continue

                modality_dest = patient_dest / modality
                modality_dest.mkdir(parents=True, exist_ok=True)

                files = sorted([
                    f for f in serie.iterdir()
                    if f.is_file() and is_probable_dicom(f)
                ])

                if not files:
                    print(
                        f"[{patient_id}] WARNING: "
                        f"{modality.upper()} series without files: {serie}"
                    )
                    continue

                for f in files:
                    final_destination = modality_dest / f.name

                    if final_destination.exists():
                        print(
                            f"[{patient_id}] WARNING: already exists "
                            f"'{final_destination.name}', skipping."
                        )
                        continue

                    if DRY_RUN:
                        print(f"[DRY] {f} -> {final_destination}")
                    else:
                        try:
                            shutil.copy2(f, final_destination)
                            total_copy += 1
                        except Exception as e:
                            print(f"[ERROR] Failed to copy '{f}': {e}")

    print("\n== DONE ==")
    print(f"T1 series found: {total_t1_series}")
    print(f"T2 series found: {total_t2_series}")
    print(f"Copied files   : {total_copy}")
    print(f"Output folder  : {DICOM_SOURCE}")


if __name__ == "__main__":
    organize_dicom()