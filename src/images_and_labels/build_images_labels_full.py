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
    return any(k in name for k in TUMOR_STRUCTURE_NAMES)


def load_dicom_uint8(ds):
    arr = ds.pixel_array
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


def lps_to_pixels(lps_points, ds):
    img_pos = np.array(ds.ImagePositionPatient, float)
    orient = np.array(ds.ImageOrientationPatient, float)
    row_dir, col_dir = orient[:3], orient[3:]
    row_sp, col_sp = map(float, ds.PixelSpacing)

    pts = []
    for p in lps_points:
        v = np.array(p) - img_pos
        r = int(round(np.dot(v, col_dir) / col_sp))
        c = int(round(np.dot(v, row_dir) / row_sp))
        if 0 <= r < ds.Rows and 0 <= c < ds.Columns:
            pts.append([c, r])

    return np.array(pts, np.int32) if len(pts) >= 3 else None


def slice_position(ds):
    orient = np.array(ds.ImageOrientationPatient, float)
    normal = np.cross(orient[:3], orient[3:])
    return float(np.dot(normal, ds.ImagePositionPatient)), normal


def point_position(p, normal):
    return float(np.dot(normal, p))


def pick_slice(ds_list, poly):
    _, normal = slice_position(ds_list[0])
    target = point_position(poly[0], normal)

    best, delta = None, None
    for ds in ds_list:
        s, _ = slice_position(ds)
        d = abs(s - target)
        if best is None or d < delta:
            best, delta = ds, d

    return best if best is not None and delta <= TOL_MM else None


def load_tumor_polys(json_path):
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text())
    polys = []
    for s in data:
        if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")):
            polys.extend(s.get(CONTOUR_POINTS_KEY, []))
    return polys


def main():
    (BASE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels_detect").mkdir(parents=True, exist_ok=True)

    patients = sorted(
        p for p in DICOM_SOURCE.iterdir()
        if p.is_dir() and PATIENT_ID_REGEX.match(p.name)
    )
    n_no_slice_match = 0
    n_out_of_bounds = 0

    for pdir in tqdm(patients, desc="Generating full dataset"):
        pid = pdir.name

        t1_dir = pdir / MODALITY
        if not t1_dir.exists():
            continue

        contour_json = CONTOURS_SOURCE / pid / contour_json_filename()
        tumor_polys = load_tumor_polys(contour_json)

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

        poly_to_slice = {}
        for poly in tumor_polys:
            ds = pick_slice(ds_list, poly)
            if ds:
                poly_to_slice.setdefault(ds.filename, []).append(poly)
            else:
                n_no_slice_match += 1

        out_img = BASE_DIR / "images" / pid
        out_lbl_seg = BASE_DIR / "labels" / pid
        out_lbl_det = BASE_DIR / "labels_detect" / pid
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl_seg.mkdir(parents=True, exist_ok=True)
        out_lbl_det.mkdir(parents=True, exist_ok=True)

        for ds in ds_list:
            img = load_dicom_uint8(ds)
            stem = Path(ds.filename).stem

            img_path = out_img / f"{stem}.png"
            seg_path = out_lbl_seg / f"{stem}.txt"
            det_path = out_lbl_det / f"{stem}.txt"

            cv2.imwrite(str(img_path), img)

            h, w = img.shape
            seg_lines = []
            det_lines = []

            for poly in poly_to_slice.get(ds.filename, []):
                pts = lps_to_pixels(poly, ds)
                if pts is None:
                    n_out_of_bounds += 1
                    continue

                x, y, ww, hh = cv2.boundingRect(pts)
                xc, yc = (x + ww / 2) / w, (y + hh / 2) / h
                wn, hn = ww / w, hh / h
                poly_n = (pts / np.array([w, h])).flatten()

                # Segmentation format: class x1 y1 x2 y2 ...
                seg_lines.append(
                    "0 {}".format(" ".join(f"{v:.6f}" for v in poly_n))
                )

                # Detection format: class xc yc w h
                det_lines.append(
                    "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(xc, yc, wn, hn)
                )

            seg_path.write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""))
            det_path.write_text("\n".join(det_lines) + ("\n" if det_lines else ""))

    print("Full dataset (images + labels) generated.")
    print(f"  Segmentation labels : {BASE_DIR / 'labels'}")
    print(f"  Detection labels    : {BASE_DIR / 'labels_detect'}")
    if n_no_slice_match:
        print(f"WARNING: {n_no_slice_match} contour(s) skipped -- no matching DICOM slice within {TOL_MM}mm")
    if n_out_of_bounds:
        print(f"WARNING: {n_out_of_bounds} contour(s) skipped -- fewer than 3 points fell inside the image bounds")


if __name__ == "__main__":
    main()