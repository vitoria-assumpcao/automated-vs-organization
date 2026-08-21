import re

# Patient folders/files follow "VS-SEG-###" (e.g. VS-SEG-001).
PATIENT_ID_PREFIX = "VS-SEG-"
PATIENT_ID_REGEX = re.compile(r"^VS-SEG-\d{3}$", re.IGNORECASE)
PATIENT_ID_TEMPLATE = "VS-SEG-{:03d}"

# Source contour folders follow the TCIA/Gamma-Knife naming convention
# "vs_gk_<patient_number>_<t1|t2>". Group 1 is the patient number (used with
# PATIENT_ID_TEMPLATE), group 2 is the modality.
CONTOUR_FOLDER_REGEX = re.compile(r"^vs_gk_(\d+)_(t1|t2)$", re.IGNORECASE)

# MRI series used by the tumor-only pipeline; the DICOM subfolder and the
# contour JSON filename are both derived from this.
MODALITY = "t1"

# Max distance (mm) between a contour's plane and a slice's plane for the
# two to be considered a match. Depends on the dataset's slice spacing --
# increase proportionally for datasets with coarser spacing than VS-SEG's.
TOL_MM = 0.8


def contour_json_filename(modality: str = MODALITY) -> str:
    return f"{modality}_contours.json"


# Structure name aliases (lowercase) that identify the tumor contour in the
# source JSON.
TUMOR_STRUCTURE_NAMES = {
    "tv", "tumor", "tumour", "schwannoma", "vs", "vestibular schwannoma",
}

# Keys in each entry of the contour JSON (a per-structure, per-slice LPS
# polygon list -- this repo's JSON mirror of a DICOM RTSTRUCT's
# ROIContourSequence).
STRUCTURE_NAME_KEY = "structure_name"
CONTOUR_POINTS_KEY = "LPS_contour_points"
