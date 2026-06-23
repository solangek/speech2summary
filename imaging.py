"""Pure radiograph imaging helpers — no Streamlit, no LLM deps.

Depends only on numpy + Pillow (+ pydicom, imported lazily in dicom_to_png).
"""
import io

import numpy as np
from PIL import Image

MAX_EDGE = 1600  # downscale the long edge before use


def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_EDGE:
        scale = MAX_EDGE / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def _safe_meta(ds) -> dict:
    """Allowlist of non-PHI DICOM tags only.

    Owner/patient names, IDs, dates and institution are never read, so no
    identifying header data leaves the app (pixels only go to the model; the
    rendered PNG is persisted — never the DICOM header).
    """
    def g(attr):
        v = getattr(ds, attr, None)
        return str(v) if v not in (None, "") else ""
    return {
        "species": g("PatientSpeciesDescription"),
        "breed": g("PatientBreedDescription"),
        "body_part": g("BodyPartExamined"),
        "view": g("ViewPosition"),
        "modality": g("Modality"),
    }


def dicom_to_png(data: bytes) -> tuple[bytes, dict]:
    """Decode DICOM pixels, apply modality + VOI/windowing, normalize to 8-bit PNG."""
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array
    meta = _safe_meta(ds)

    if arr.ndim == 3:  # already RGB (e.g. ultrasound) — display as-is
        return _to_png_bytes(_resize(Image.fromarray(arr.astype(np.uint8)[..., :3]))), meta

    arr = apply_modality_lut(arr, ds).astype(np.float32)
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr  # invert so dense structures (bone) render bright

    lo, hi = np.percentile(arr, [1, 99])  # robust window against outlier pixels
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0) * 255.0
    return _to_png_bytes(_resize(Image.fromarray(arr.astype(np.uint8)))), meta


def load_image(name: str, data: bytes) -> tuple[bytes, dict]:
    """Route DICOM vs raster image; return (png_bytes, meta)."""
    if name.lower().endswith(".dcm") or _is_dicom(data):
        return dicom_to_png(data)
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return _to_png_bytes(_resize(img)), {}
