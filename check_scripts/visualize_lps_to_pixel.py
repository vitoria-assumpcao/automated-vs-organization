from pathlib import Path
import os
import json
import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.config import CONTOUR_POINTS_KEY, MODALITY, STRUCTURE_NAME_KEY, TOL_MM, TUMOR_STRUCTURE_NAMES, contour_json_filename

load_dotenv()

DICOM_SOURCE = Path(os.getenv("DICOM_SOURCE"))
CONTOURS_SOURCE = Path(os.getenv("CONTOURS_SOURCE"))
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def is_tumor_structure(name: str) -> bool:
    name = name.lower()
    return any(k in name for k in TUMOR_STRUCTURE_NAMES)


def load_dicom_uint8(ds: pydicom.Dataset):
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


def lps_to_pixels(lps_points, ds: pydicom.Dataset):
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
    return float(np.dot(normal, np.array(point_lps, dtype=float)))


def pick_nearest_slice(ds_list, poly_lps, tol_mm=TOL_MM):
    _, normal = slice_plane_position(ds_list[0])
    target = point_plane_position(poly_lps[0], normal)

    best, best_delta = None, None
    for ds in ds_list:
        s, _ = slice_plane_position(ds)
        delta = abs(target - s)
        if best is None or delta < best_delta:
            best, best_delta = ds, delta

    return best if best is not None and best_delta <= tol_mm else None


def load_tumor_slices(contour_json: Path):
    data = json.loads(contour_json.read_text())
    tumor_polys = []
    for s in data:
        if is_tumor_structure(s.get(STRUCTURE_NAME_KEY, "")):
            tumor_polys.extend(s.get(CONTOUR_POINTS_KEY, []))
    return tumor_polys


def load_ds_list(t1_dir: Path):
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
        except Exception as e:
            print(f"WARNING: failed to read DICOM {dcm}: {e}")
    return ds_list


def visualize(patient_id: str, slice_index: int = 0):
    """
    Builds a 4-panel figure for one tumor slice of a patient:
    raw DICOM, normalized PNG, projected LPS contour points, and
    the resulting binary mask from cv2.fillPoly.
    """
    contour_json = CONTOURS_SOURCE / patient_id / contour_json_filename()
    tumor_slices = load_tumor_slices(contour_json)
    if not tumor_slices:
        raise ValueError(f"No tumor contours found for {patient_id}")

    t1_dir = DICOM_SOURCE / patient_id / MODALITY
    ds_list = load_ds_list(t1_dir)
    if not ds_list:
        raise ValueError(f"No usable DICOM slices found for {patient_id}")

    poly_lps = tumor_slices[slice_index]
    ds_best = pick_nearest_slice(ds_list, poly_lps)
    if ds_best is None:
        raise ValueError("Could not match contour to a DICOM slice")

    raw = ds_best.pixel_array
    img = load_dicom_uint8(ds_best)
    pts = lps_to_pixels(poly_lps, ds_best)
    if pts is None:
        raise ValueError("Contour points fell outside the image bounds")

    mask = np.zeros((ds_best.Rows, ds_best.Columns), np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"LPS-to-Pixel Coordinate Conversion — {patient_id}", fontsize=13, fontweight="bold")

    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("1. Raw DICOM\n(pixel array, no normalization)")

    axes[1].imshow(img, cmap="gray")
    axes[1].set_title("2. Converted PNG\n(percentile normalization)")

    axes[2].imshow(img, cmap="gray")
    axes[2].plot(pts[:, 0], pts[:, 1], "o-", color="orangered", markersize=3, linewidth=1)
    axes[2].set_title("3. LPS contour points\nprojected to pixel space")

    axes[3].imshow(img, cmap="gray")
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask > 0] = (1, 0, 0, 0.45)
    axes[3].imshow(overlay)
    axes[3].set_title("4. Generated binary mask\n(cv2.fillPoly)")

    for ax in axes:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color="red", lw=6, label="Tumor region")]
    fig.legend(handles=handles, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"lps_to_pixel_{patient_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    visualize("VS-SEG-005")
