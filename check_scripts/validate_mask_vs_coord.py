from pathlib import Path
import os
import json
import random
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
MASKS_DIR = BASE_DIR / "masks"

MAX_PATIENTS = None

SAVE_VISUAL_SAMPLES = True
VISUAL_SAMPLE_COUNT = 6          # how many sample slices to save, one per patient
VISUAL_OUTPUT_DIR = Path("./validation_visuals_mask")


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


def lps_to_pixels(lps_points, ds: pydicom.Dataset):
    """Same rounding logic as build_masks_tumor.py, to reproduce the
    exact points that generated the on-disk mask."""
    img_pos = np.array(ds.ImagePositionPatient, dtype=float)
    img_orient = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = img_orient[:3], img_orient[3:]
    row_sp, col_sp = map(float, ds.PixelSpacing)
    h, w = int(ds.Rows), int(ds.Columns)

    pts = []
    for pt in lps_points:
        v = np.array(pt, dtype=float) - img_pos
        r = int(round(np.dot(v, col_dir) / col_sp))
        c = int(round(np.dot(v, row_dir) / row_sp))
        if 0 <= r < h and 0 <= c < w:
            pts.append([c, r])
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32)


def extract_mask_contour(mask_path: Path):
    """Pools points from every external contour, since a slice can hold
    more than one disjoint tumor component (e.g. a satellite nodule)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)  # (x, y) pairs


def nearest_neighbor_distances_mm(pts_a, pts_b, row_sp, col_sp):
    """Distance (mm) from each point in pts_a to its nearest neighbor in
    pts_b. Points are (x, y) = (row_dir-axis, col_dir-axis) pixel indices,
    so x is scaled by row_sp and y by col_sp before pairing -- this keeps
    both the nearest-neighbor match and the reported distance correct for
    anisotropic pixel spacing, not just square pixels."""
    scale = np.array([row_sp, col_sp], dtype=float)
    pts_a_mm = np.asarray(pts_a, dtype=float) * scale
    pts_b_mm = np.asarray(pts_b, dtype=float) * scale
    dists = []
    for p in pts_a_mm:
        d = np.min(np.linalg.norm(pts_b_mm - p, axis=1))
        dists.append(d)
    return np.array(dists)


def load_dicom_uint8(ds: pydicom.Dataset):
    """Percentile normalization for visualization only (no VOI LUT,
    keeps the script lightweight)."""
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / max(1e-6, hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def save_mask_comparison(ds_best, pts_original, mask, mean_err_mm, out_path: Path):
    """Overlays the JSON-derived polygon (green) and the on-disk mask
    (red fill) on the DICOM slice and saves a zoomed crop as PNG."""
    base = load_dicom_uint8(ds_best)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # saved mask (red fill)
    red_layer = np.zeros_like(img)
    red_layer[mask > 0] = (0, 0, 255)
    img = cv2.addWeighted(img, 1.0, red_layer, 0.45, 0)

    # JSON-derived polygon (green)
    cv2.polylines(img, [pts_original.reshape(-1, 1, 2)], isClosed=True,
                  color=(0, 255, 0), thickness=1)

    # zoomed crop around the tumor so the overlay is visible
    x_min, y_min = pts_original.min(axis=0) - 15
    x_max, y_max = pts_original.max(axis=0) + 15
    h, w = img.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    crop = img[y_min:y_max, x_min:x_max]
    crop = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)

    # dark banner behind the text so it stays legible over any tissue
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 34), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_TRIPLEX
    white = (255, 255, 255)
    cv2.putText(crop, f"mean error: {mean_err_mm:.4f} mm", (8, 16),
                font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(crop, "green=JSON polygon  red=saved mask", (8, 30),
                font, 0.45, white, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def main():
    if not MASKS_DIR.exists():
        print(f"ERROR: masks folder not found at {MASKS_DIR}. "
              f"Run build_masks_tumor.py first.")
        return

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
    n_out_of_bounds = 0

    for pdir in tqdm(patients, desc="Comparing mask vs JSON coordinates"):
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

        mask_dir = MASKS_DIR / patient_id
        if not mask_dir.exists():
            continue

        patient_errors_mm = []

        for poly_lps in tumor_slices:
            if not (isinstance(poly_lps, list) and len(poly_lps) >= 3):
                continue

            ds_best = pick_nearest_slice(ds_list, poly_lps, TOL_MM)
            if ds_best is None:
                n_no_slice_match += 1
                continue

            pts_original = lps_to_pixels(poly_lps, ds_best)
            if pts_original is None:
                n_out_of_bounds += 1
                continue

            stem = Path(ds_best.filename).stem
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue

            pts_from_mask = extract_mask_contour(mask_path)
            if pts_from_mask is None or len(pts_from_mask) == 0:
                continue

            row_sp, col_sp = map(float, ds_best.PixelSpacing)

            # distance from each original point to the mask's extracted contour
            d_mm = nearest_neighbor_distances_mm(pts_original, pts_from_mask, row_sp, col_sp)

            patient_errors_mm.extend(d_mm.tolist())
            all_errors_mm.extend(d_mm.tolist())

            if (SAVE_VISUAL_SAMPLES and visuals_saved < VISUAL_SAMPLE_COUNT
                    and patient_id not in patients_with_visual):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                out_path = VISUAL_OUTPUT_DIR / f"{patient_id}_{stem}.png"
                save_mask_comparison(ds_best, pts_original, mask, float(d_mm.mean()), out_path)
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
        print(f"{pid}: n_points={n:5d}  mean_err={mean_e:.6f} mm  std_err={std_e:.6f} mm  max_err={max_e:.6f} mm")

    if all_errors_mm:
        arr = np.array(all_errors_mm)
        print("\n--- Overall summary ---")
        print(f"Total points compared    : {len(arr)}")
        print(f"Mean reprojection error  : {arr.mean():.6f} mm")
        print(f"Standard deviation       : {arr.std():.6f} mm")
        print(f"Max error                : {arr.max():.6f} mm")
        print(f"Median error             : {np.median(arr):.6f} mm")
    else:
        print("No points compared -- check BASE_DIR/masks and .env.")

    if n_no_slice_match:
        print(f"WARNING: {n_no_slice_match} contour(s) skipped -- no matching DICOM slice within {TOL_MM}mm")
    if n_out_of_bounds:
        print(f"WARNING: {n_out_of_bounds} contour(s) skipped -- fewer than 3 points fell inside the image bounds")


if __name__ == "__main__":
    main()