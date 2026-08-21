from pathlib import Path
import os
import json
import numpy as np
import pydicom
import cv2
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import CONTOUR_POINTS_KEY, MODALITY, PATIENT_ID_REGEX, STRUCTURE_NAME_KEY, TOL_MM, TUMOR_STRUCTURE_NAMES, contour_json_filename

load_dotenv()


DICOM_SOURCE = Path(os.getenv("DICOM_SOURCE"))
CONTOURS_SOURCE = Path(os.getenv("CONTOURS_SOURCE"))
BASE_DIR = Path(os.getenv("BASE_DIR"))


def is_tumor_structure(name: str) -> bool:
    name = name.lower()
    return any(k in name for k in TUMOR_STRUCTURE_NAMES)


def lps_to_pixels(poly_lps, ds: pydicom.Dataset):
    """
    Convert contour points from LPS (patient coordinates)
    to image pixel coordinates.
    """
    img_pos = np.array(ds.ImagePositionPatient, dtype=float)
    orient = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = orient[:3], orient[3:]
    row_sp, col_sp = map(float, ds.PixelSpacing)

    pts = []
    for p in poly_lps:
        v = np.array(p, dtype=float) - img_pos
        r = int(round(np.dot(v, col_dir) / col_sp))
        c = int(round(np.dot(v, row_dir) / row_sp))
        if 0 <= r < ds.Rows and 0 <= c < ds.Columns:
            pts.append([c, r]) 

    return np.array(pts, dtype=np.int32) if len(pts) >= 3 else None


def slice_plane_position(ds: pydicom.Dataset):
    """
    Compute the scalar position of the image plane.
    """
    orient = np.array(ds.ImageOrientationPatient, dtype=float)
    normal = np.cross(orient[:3], orient[3:])
    return float(np.dot(normal, ds.ImagePositionPatient)), normal


def point_plane_position(point_lps, normal):
    return float(np.dot(normal, point_lps))


def pick_nearest_slice(ds_list, poly_lps, tol_mm=TOL_MM):
    """
    Select the DICOM slice whose image plane is closest
    to the contour polygon plane.
    """
    if not ds_list or not poly_lps:
        return None

    _, normal = slice_plane_position(ds_list[0])
    target = point_plane_position(poly_lps[0], normal)

    best, best_delta = None, None
    for ds in ds_list:
        s, _ = slice_plane_position(ds)
        delta = abs(s - target)
        if best is None or delta < best_delta:
            best, best_delta = ds, delta

    return best if best is not None and best_delta <= tol_mm else None


def load_tumor_polys(json_path: Path):
    """
    Load all tumor contour polygons from a contour JSON file.
    """
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text())
    polys = []

    for s in data:
        if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")):
            polys.extend(s.get(CONTOUR_POINTS_KEY, []))

    return polys


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "masks").mkdir(parents=True, exist_ok=True)

    patients = sorted(
        p for p in DICOM_SOURCE.iterdir()
        if p.is_dir() and PATIENT_ID_REGEX.match(p.name)
    )

    for pdir in tqdm(patients, desc="Generating full masks"):
        patient_id = pdir.name

        t1_dir = pdir / MODALITY
        if not t1_dir.exists():
            continue

        contour_json = CONTOURS_SOURCE / patient_id / contour_json_filename()
        tumor_polys = load_tumor_polys(contour_json)

        ds_list = []
        for dcm in sorted(t1_dir.iterdir()):
            if dcm.suffix.lower() != ".dcm":
                continue
            try:
                ds = pydicom.dcmread(dcm, force=True)
                if all(k in ds for k in ("ImagePositionPatient",
                                         "ImageOrientationPatient",
                                         "PixelSpacing")):
                    ds.filename = str(dcm)
                    ds_list.append(ds)
            except Exception:
                pass

        if not ds_list:
            continue

        poly_map = {}
        for poly in tumor_polys:
            ds = pick_nearest_slice(ds_list, poly)
            if ds:
                poly_map.setdefault(ds.filename, []).append(poly)

        out_dir = BASE_DIR / "masks" / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for ds in ds_list:
            mask = np.zeros((ds.Rows, ds.Columns), dtype=np.uint8)

            for poly in poly_map.get(ds.filename, []):
                pts = lps_to_pixels(poly, ds)
                if pts is not None:
                    cv2.fillPoly(mask, [pts], 255)

            stem = Path(ds.filename).stem
            cv2.imwrite(str(out_dir / f"{stem}.png"), mask)

    print("Full mask dataset generated.")


if __name__ == "__main__":
    main()
