from pathlib import Path
import os
import json
import random
import numpy as np
import pydicom
import cv2
import SimpleITK as sitk
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import CONTOUR_POINTS_KEY, MODALITY, PATIENT_ID_REGEX, STRUCTURE_NAME_KEY, TOL_MM, TUMOR_STRUCTURE_NAMES, contour_json_filename

load_dotenv()

DICOM_SOURCE = Path(os.getenv("DICOM_SOURCE"))
CONTOURS_SOURCE = Path(os.getenv("CONTOURS_SOURCE"))

MAX_PATIENTS = None
SAVE_VISUAL_SAMPLES = True
VISUAL_SAMPLE_COUNT = 6          # how many sample slices to save, one per patient
VISUAL_OUTPUT_DIR = Path("./validation_visuals")


def is_tumor_structure(name: str) -> bool:
    name = name.lower()
    return any(k in name for k in TUMOR_STRUCTURE_NAMES)


def has_tumor(contour_json: Path) -> bool:
    if not contour_json.is_file():
        return False
    try:
        data = json.loads(contour_json.read_text())
        for s in data:
            if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")) and s.get(CONTOUR_POINTS_KEY):
                return len(s[CONTOUR_POINTS_KEY]) > 0
    except Exception as e:
        print(f"WARNING: failed to parse {contour_json}: {e}")
    return False


def load_tumor_slices(contour_json: Path):
    data = json.loads(contour_json.read_text())
    tumor_polys = []
    for s in data:
        if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")):
            tumor_polys.extend(s.get(CONTOUR_POINTS_KEY, []))
    return tumor_polys if tumor_polys else None


def slice_plane_position(ds: pydicom.Dataset):
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    return float(np.dot(normal, ipp)), normal


def point_plane_position(point_lps, normal):
    return float(np.dot(normal, np.array(point_lps, dtype=float)))


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


def own_lps_to_pixel_continuous(lps_point, ds: pydicom.Dataset):
    """Same formula as the pipeline's lps_to_pixels(), without rounding,
    for a fair sub-pixel comparison against SimpleITK."""
    img_pos = np.array(ds.ImagePositionPatient, dtype=float)
    img_orient = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = img_orient[:3], img_orient[3:]
    row_sp, col_sp = map(float, ds.PixelSpacing)

    v = np.array(lps_point, dtype=float) - img_pos
    row_idx = np.dot(v, col_dir) / col_sp
    col_idx = np.dot(v, row_dir) / row_sp
    return col_idx, row_idx  # (x, y)


def sitk_lps_to_pixel(lps_point, dcm_path: str):
    """Independent reference: resolves the same physical-to-pixel mapping
    via ITK's own TransformPhysicalPointToContinuousIndex."""
    img = sitk.ReadImage(dcm_path)
    idx = img.TransformPhysicalPointToContinuousIndex(tuple(map(float, lps_point)))
    return idx[0], idx[1]  # (x, y); ignores idx[2] (out of plane)


def load_dicom_uint8(ds: pydicom.Dataset):
    """Percentile normalization for visualization only (no VOI LUT,
    keeps the script lightweight)."""
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / max(1e-6, hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def save_visual_comparison(ds_best, pts_own, pts_sitk, mean_err_mm, out_path: Path):
    """Overlays the pipeline's polygon (green) and SimpleITK's points
    (red 'x') on the DICOM slice and saves a zoomed crop as PNG."""
    base = load_dicom_uint8(ds_best)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    pts_own_i = np.round(pts_own).astype(np.int32)
    pts_sitk_i = np.round(pts_sitk).astype(np.int32)

    # pipeline polygon (green)
    cv2.polylines(img, [pts_own_i.reshape(-1, 1, 2)], isClosed=True,
                  color=(0, 255, 0), thickness=1)

    # SimpleITK points (red 'x')
    for x, y in pts_sitk_i:
        cv2.drawMarker(img, (int(x), int(y)), color=(0, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)

    # zoomed crop around the tumor so the markers are visible
    x_min, y_min = pts_own_i.min(axis=0) - 15
    x_max, y_max = pts_own_i.max(axis=0) + 15
    h, w = img.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    crop = img[y_min:y_max, x_min:x_max]
    crop = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)

    # dark banner behind the text so it stays legible over any tissue
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 34), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_TRIPLEX
    white = (255, 255, 255)
    cv2.putText(crop, f"mean error: {mean_err_mm:.4e} mm", (8, 16),
                font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(crop, "green=pipeline  red=SimpleITK", (8, 30),
                font, 0.45, white, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def main():
    patients = sorted([
        p for p in DICOM_SOURCE.iterdir()
        if p.is_dir() and PATIENT_ID_REGEX.match(p.name)
    ])
    random.shuffle(patients)  # no fixed seed: sample set varies on every run
    if MAX_PATIENTS:
        patients = patients[:MAX_PATIENTS]

    print(f"Checking {len(patients)} patient(s)...\n")

    all_errors_mm = []
    per_patient_summary = []
    visuals_saved = 0
    patients_with_visual = set()
    n_no_slice_match = 0

    for pdir in tqdm(patients, desc="Validating LPS-to-pixel transform"):
        patient_id = pdir.name
        contour_json = CONTOURS_SOURCE / patient_id / contour_json_filename()

        if not has_tumor(contour_json):
            continue
        tumor_slices = load_tumor_slices(contour_json)
        if not tumor_slices:
            continue

        t1_dir = pdir / MODALITY
        if not t1_dir.exists():
            continue

        ds_list = []
        for dcm in sorted(t1_dir.iterdir()):
            if dcm.suffix.lower() != ".dcm":
                continue
            try:
                ds = pydicom.dcmread(dcm, force=True)
                if all(k in ds for k in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing")):
                    ds.filename = str(dcm)
                    ds_list.append(ds)
            except Exception as e:
                print(f"WARNING: failed to read DICOM {dcm}: {e}")
        if not ds_list:
            continue

        patient_errors_mm = []

        for poly_lps in tumor_slices:
            if not (isinstance(poly_lps, list) and len(poly_lps) >= 3):
                continue

            ds_best = pick_nearest_slice(ds_list, poly_lps, TOL_MM)
            if ds_best is None:
                n_no_slice_match += 1
                continue

            row_sp, col_sp = map(float, ds_best.PixelSpacing)

            slice_pts_own, slice_pts_sitk = [], []

            for pt in poly_lps:
                x_own, y_own = own_lps_to_pixel_continuous(pt, ds_best)
                x_sitk, y_sitk = sitk_lps_to_pixel(pt, ds_best.filename)

                # x is a row_dir-axis index (1/row_sp units), y is a col_dir-axis
                # index (1/col_sp units); scale each axis before combining so
                # this is correct for anisotropic pixel spacing too.
                dx_mm = (x_own - x_sitk) * row_sp
                dy_mm = (y_own - y_sitk) * col_sp
                d_mm = np.hypot(dx_mm, dy_mm)

                patient_errors_mm.append(d_mm)
                all_errors_mm.append(d_mm)
                slice_pts_own.append((x_own, y_own))
                slice_pts_sitk.append((x_sitk, y_sitk))

            if (SAVE_VISUAL_SAMPLES and visuals_saved < VISUAL_SAMPLE_COUNT
                    and patient_id not in patients_with_visual and slice_pts_own):
                stem = Path(ds_best.filename).stem
                out_path = VISUAL_OUTPUT_DIR / f"{patient_id}_{stem}.png"
                slice_mean_err = float(np.mean(
                    [np.hypot((a[0] - b[0]) * row_sp, (a[1] - b[1]) * col_sp)
                     for a, b in zip(slice_pts_own, slice_pts_sitk)]
                ))
                save_visual_comparison(ds_best, slice_pts_own, slice_pts_sitk,
                                        slice_mean_err, out_path)
                visuals_saved += 1
                patients_with_visual.add(patient_id)

        if patient_errors_mm:
            per_patient_summary.append((
                patient_id,
                len(patient_errors_mm),
                float(np.mean(patient_errors_mm)),
                float(np.std(patient_errors_mm)),
                float(np.max(patient_errors_mm)),
            ))

    print("\n--- Per-patient results (mm) ---")
    for pid, n, mean_e, std_e, max_e in per_patient_summary:
        print(f"{pid}: n_points={n:5d}  mean_err={mean_e:.4e} mm  std_err={std_e:.4e} mm  max_err={max_e:.4e} mm")

    if all_errors_mm:
        arr = np.array(all_errors_mm)
        print("\n--- Overall summary ---")
        print(f"Total points compared    : {len(arr)}")
        print(f"Mean reprojection error  : {arr.mean():.4e} mm")
        print(f"Standard deviation       : {arr.std():.4e} mm")
        print(f"Max error                : {arr.max():.4e} mm")
        print(f"Median error             : {np.median(arr):.4e} mm")
    else:
        print("No points compared -- check DICOM_SOURCE/CONTOURS_SOURCE in .env.")

    if n_no_slice_match:
        print(f"WARNING: {n_no_slice_match} contour(s) skipped -- no matching DICOM slice within {TOL_MM}mm")


if __name__ == "__main__":
    main()
