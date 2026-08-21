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


def lps_to_pixels(lps_points, ds: pydicom.Dataset):
    if not all(k in ds for k in ("ImagePositionPatient",
                                 "ImageOrientationPatient",
                                 "PixelSpacing")):
        return None

    img_pos = np.array(ds.ImagePositionPatient, dtype=float)
    img_orient = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = img_orient[:3], img_orient[3:]
    row_sp, col_sp = map(float, ds.PixelSpacing)

    h, w = int(ds.Rows), int(ds.Columns)
    pts = []

    for pt in lps_points:
        p = np.array(pt, dtype=float)
        v = p - img_pos
        r = int(round(np.dot(v, col_dir) / col_sp))
        c = int(round(np.dot(v, row_dir) / row_sp))
        if 0 <= r < h and 0 <= c < w:
            pts.append([c, r])  

    if len(pts) < 3:
        return None

    return np.array(pts, dtype=np.int32)


def slice_plane_position(ds: pydicom.Dataset):
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    return float(np.dot(normal, ipp)), normal


def point_plane_position(point_lps, normal):
    p = np.array(point_lps, dtype=float)
    return float(np.dot(normal, p))


def pick_nearest_slice(ds_list, poly_lps, tol_mm=TOL_MM):
    if not ds_list or not poly_lps:
        return None

    _, normal = slice_plane_position(ds_list[0])
    target = point_plane_position(poly_lps[0], normal)

    best, best_delta = None, None
    for ds in ds_list:
        s, _ = slice_plane_position(ds)
        delta = abs(target - s)
        if best is None or delta < best_delta:
            best, best_delta = ds, delta

    return best if best is not None and best_delta <= tol_mm else None


def has_tumor(contour_json: Path) -> bool:
    if not contour_json.is_file():
        return False

    try:
        data = json.loads(contour_json.read_text())
        for s in data:
            if (is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")) and
                s.get(CONTOUR_POINTS_KEY)):
                return len(s[CONTOUR_POINTS_KEY]) > 0
    except Exception:
        pass

    return False


def load_tumor_slices(contour_json: Path):
    data = json.loads(contour_json.read_text())
    tumor_polys = []

    for s in data:
        if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")):
            tumor_polys.extend(s.get(CONTOUR_POINTS_KEY, []))

    return tumor_polys if tumor_polys else None


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "masks").mkdir(parents=True, exist_ok=True)

    patients = sorted([
        p for p in DICOM_SOURCE.iterdir()
        if p.is_dir() and PATIENT_ID_REGEX.match(p.name)
    ])

    print(f"Patients found: {len(patients)}")
    total_masks = 0

    for pdir in tqdm(patients, desc="Generating masks"):
        patient_id = pdir.name
        contour_json = CONTOURS_SOURCE / patient_id / contour_json_filename()

        if not has_tumor(contour_json):
            continue

        tumor_slices = load_tumor_slices(contour_json)
        if not tumor_slices:
            continue

        ds_list = []
        t1_dir = pdir / MODALITY
        if not t1_dir.exists():
            continue

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

        out_dir = BASE_DIR / "masks" / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_by_filename = {ds.filename: ds for ds in ds_list}
        poly_map = {}
        for poly_lps in tumor_slices:
            if not (isinstance(poly_lps, list) and len(poly_lps) >= 3):
                continue

            ds_best = pick_nearest_slice(ds_list, poly_lps, TOL_MM)
            if ds_best is None:
                continue

            poly_map.setdefault(ds_best.filename, []).append(poly_lps)

        for ds_filename, polys in poly_map.items():
            ds_best = ds_by_filename[ds_filename]
            mask = np.zeros((ds_best.Rows, ds_best.Columns), np.uint8)

            for poly_lps in polys:
                pts = lps_to_pixels(poly_lps, ds_best)
                if pts is not None:
                    cv2.fillPoly(mask, [pts], 255)

            stem = Path(ds_best.filename).stem
            cv2.imwrite(str(out_dir / f"{stem}.png"), mask)
            total_masks += 1

    print(f"\nMasks saved to {BASE_DIR / 'masks'}")
    print(f"Total masks generated: {total_masks}")


if __name__ == "__main__":
    main()
