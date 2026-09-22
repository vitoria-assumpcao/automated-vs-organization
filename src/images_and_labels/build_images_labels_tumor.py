from pathlib import Path
import os
import json
import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut
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
    return any(keyword in name for keyword in TUMOR_STRUCTURE_NAMES)


def load_dicom_uint8(ds: pydicom.Dataset):
    """
    Loads a DICOM slice and converts it to an 8-bit grayscale image.
    VOI LUT is applied when available, followed by percentile-based
    intensity normalization.
    """
    if not hasattr(ds, "pixel_array"):
        return None

    arr = ds.pixel_array
    if arr.ndim != 2:
        return None

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception as e:
        print(f"WARNING: VOI LUT failed for {getattr(ds, 'filename', ds)}: {e}")

    arr = arr.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())

    arr = np.clip((arr - lo) / max(1e-6, hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def lps_to_pixels(lps_points, ds: pydicom.Dataset):
    """
    Converts contour points from LPS (patient coordinate system)
    to image pixel coordinates.

    Returns points as (x, y) pairs compatible with OpenCV.
    """
    if ('ImagePositionPatient' not in ds or
        'ImageOrientationPatient' not in ds or
        'PixelSpacing' not in ds):
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
        row_idx = np.dot(v, col_dir) / col_sp
        col_idx = np.dot(v, row_dir) / row_sp

        r = int(round(row_idx))
        c = int(round(col_idx))

        if 0 <= r < h and 0 <= c < w:
            pts.append([c, r])

    if len(pts) < 3:
        return None

    return np.array(pts, dtype=np.int32)


def has_tumor(contour_path: Path) -> bool:
    """
    Checks whether a contour file contains at least one tumor structure.
    """
    if not contour_path.is_file():
        return False

    try:
        data = json.loads(contour_path.read_text())
        for s in data:
            if (is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")) and
                s.get(CONTOUR_POINTS_KEY)):
                return len(s[CONTOUR_POINTS_KEY]) > 0
    except Exception as e:
        print(f"WARNING: failed to parse {contour_path}: {e}")

    return False


def load_tumor_slices(contour_path: Path):
    """
    Returns the list of LPS polygons corresponding to tumor contours,
    one polygon per slice.
    """
    data = json.loads(contour_path.read_text())
    tumor_polys = []
    for s in data:
        if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")):
            tumor_polys.extend(s.get(CONTOUR_POINTS_KEY, []))
    return tumor_polys if tumor_polys else None


def slice_plane_position(ds: pydicom.Dataset):
    """
    Computes a scalar position of the image plane using
    ImageOrientationPatient and ImagePositionPatient.
    """
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    s = float(np.dot(normal, ipp))
    return s, normal


def point_plane_position(point_lps, normal):
    p = np.array(point_lps, dtype=float)
    return float(np.dot(normal, p))


def pick_nearest_slice(ds_list, poly_lps, tol_mm=TOL_MM):
    """
    Selects the DICOM slice whose image plane is closest
    to the contour polygon plane.
    """
    if not ds_list or not poly_lps:
        return None

    s0, normal = slice_plane_position(ds_list[0])
    target = point_plane_position(poly_lps[0], normal)

    best, best_delta = None, None
    for ds in ds_list:
        s, _ = slice_plane_position(ds)
        delta = abs(target - s)
        if best is None or delta < best_delta:
            best, best_delta = ds, delta

    if best is not None and best_delta <= tol_mm:
        return best

    return None


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels_detect").mkdir(parents=True, exist_ok=True)

    patients = sorted([
        p for p in DICOM_SOURCE.iterdir()
        if p.is_dir() and PATIENT_ID_REGEX.match(p.name)
    ])

    print(f"Patients found in DICOM directory: {len(patients)}")

    total_png, total_label_files = 0, 0
    n_no_slice_match = 0
    n_out_of_bounds = 0

    for pdir in tqdm(patients, desc="Processing patients"):
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
            if not dcm.is_file() or dcm.suffix.lower() != ".dcm":
                continue
            try:
                ds = pydicom.dcmread(dcm, force=True)
                if ('ImagePositionPatient' in ds and
                    'ImageOrientationPatient' in ds and
                    'PixelSpacing' in ds):
                    ds.filename = str(dcm)
                    ds_list.append(ds)
            except Exception as e:
                print(f"WARNING: failed to read DICOM {dcm}: {e}")

        if not ds_list:
            continue

        out_img_dir = BASE_DIR / "images" / patient_id
        out_lbl_seg = BASE_DIR / "labels" / patient_id
        out_lbl_det = BASE_DIR / "labels_detect" / patient_id
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_seg.mkdir(parents=True, exist_ok=True)
        out_lbl_det.mkdir(parents=True, exist_ok=True)

        for poly_lps in tumor_slices:
            if not (isinstance(poly_lps, list) and len(poly_lps) >= 3):
                continue

            ds_best = pick_nearest_slice(ds_list, poly_lps, tol_mm=TOL_MM)
            if ds_best is None:
                n_no_slice_match += 1
                continue

            img = load_dicom_uint8(ds_best)
            if img is None:
                continue

            pts_px = lps_to_pixels(poly_lps, ds_best)
            if pts_px is None:
                n_out_of_bounds += 1
                continue

            dcm_stem = Path(ds_best.filename).stem
            img_out = out_img_dir / f"{dcm_stem}.png"
            seg_out = out_lbl_seg / f"{dcm_stem}.txt"
            det_out = out_lbl_det / f"{dcm_stem}.txt"

            if not img_out.exists():
                cv2.imwrite(str(img_out), img)
                total_png += 1

            h, w = img.shape[:2]
            x, y, ww, hh = cv2.boundingRect(pts_px)

            x_c = (x + ww / 2) / w
            y_c = (y + hh / 2) / h
            w_n = ww / w
            h_n = hh / h

            poly = (pts_px.astype(np.float32) / np.array([w, h])).flatten()

            # Segmentation format: class x1 y1 x2 y2 ...
            seg_line = "0 {}".format(" ".join(f"{v:.6f}" for v in poly))

            # Detection format: class xc yc w h
            det_line = "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(x_c, y_c, w_n, h_n)

            # Write segmentation label
            if seg_out.exists() and seg_out.read_text().strip():
                seg_out.write_text(seg_out.read_text().rstrip() + "\n" + seg_line + "\n")
            else:
                seg_out.write_text(seg_line + "\n")

            # Write detection label
            if det_out.exists() and det_out.read_text().strip():
                det_out.write_text(det_out.read_text().rstrip() + "\n" + det_line + "\n")
            else:
                det_out.write_text(det_line + "\n")
                total_label_files += 1

    print("\nPre-split completed.")
    print(f"PNG images generated ...............: {total_png}")
    print(f"Label files created ...............: {total_label_files}")
    print(f"Images directory         : {BASE_DIR / 'images'}")
    print(f"Segmentation labels      : {BASE_DIR / 'labels'}")
    print(f"Detection labels         : {BASE_DIR / 'labels_detect'}")
    if n_no_slice_match:
        print(f"WARNING: {n_no_slice_match} contour(s) skipped -- no matching DICOM slice within {TOL_MM}mm")
    if n_out_of_bounds:
        print(f"WARNING: {n_out_of_bounds} contour(s) skipped -- fewer than 3 points fell inside the image bounds")


if __name__ == "__main__":
    main()