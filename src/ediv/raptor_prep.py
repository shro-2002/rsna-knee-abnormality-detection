"""Own implementation of the knee-MRI study-stack builder used by the Raptor checkpoints.

Geometry and preprocessing semantics follow the public documentation of dreaddevelopment's
"Knee MRI: twelve findings from a single model" (Apache-2.0 notebook; CC0 weights
dreaddevelopment/raptor-knee-widedense and siblings): fixed slot layout per plane, slices taken evenly across a
span of each series, per-series 2-98 percentile intensity window, centre crop of 140 mm from PixelSpacing,
resize to the stored size, uint8. Code here is written independently; equivalence with the original is tested
in tests/test_ediv_parity.py (v4 geometry) rather than assumed.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import cv2
import numpy as np
import pydicom
try:
    from pydicom.pixels import apply_modality_lut          # pydicom >= 3
except ImportError:                                         # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_modality_lut


@dataclass(frozen=True)
class Geometry:
    name: str
    slots: tuple          # (plane, fluid_pref, n_slices); fluid_pref 1 / 0 prefer that Fluid_Sensitive value, -1 any
    span_lo: float
    span_hi: float
    stored: int           # stack image size in pixels (crop is resized to this)
    k_eval: int           # windows scored per study
    crop_mm: float = 140.0

    @property
    def n_slices(self) -> int:
        return sum(s[2] for s in self.slots)


SLOTS_64 = (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12))
SLOTS_80 = (("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15))
# E-OAI-1 (research/36-37): nartaa 0.949 checkpoint geometry, constants from its public notebook rsna-knee-0949-anatomical-mirror
SLOTS_96 = (("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18))

GEOMETRIES = {
    "v4_64_6to94_336_42": Geometry("v4", SLOTS_64, 0.06, 0.94, 336, 42),   # parity-test geometry (sub #4)
    "v5_64_2to98_336_62": Geometry("v5", SLOTS_64, 0.02, 0.98, 336, 62),
    "v10_64_2to98_384_62": Geometry("v10", SLOTS_64, 0.02, 0.98, 384, 62),
    "v9_80_2to98_336_78": Geometry("v9", SLOTS_80, 0.02, 0.98, 336, 78),
    "nartaa_96_2to98_384_94": Geometry("nartaa", SLOTS_96, 0.02, 0.98, 384, 94),
}


def _modality(ds):
    try:
        return apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        return ds.pixel_array.astype(np.float32)


def read_slice(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    a = _modality(ds)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def order_series(series_dir: str):
    """Return ([(path, pixel_spacing_mm)] sorted along the slice normal, median spacing)."""
    recs, spacings = [], []
    for path in glob.glob(series_dir + "/*.dcm"):
        pos, ps = 0.0, 0.5
        try:
            h = pydicom.dcmread(path, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), normal))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            sp = getattr(h, "PixelSpacing", None)
            ps = float(sp[0]) if sp is not None else 0.5
            spacings.append(ps)
        except Exception:
            pass
        recs.append((pos, path, ps))
    recs.sort(key=lambda r: r[0])
    return [(p, s) for _, p, s in recs], (float(np.median(spacings)) if spacings else 0.5)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def crop_resize(a: np.ndarray, spacing: float, crop_mm: float, size: int) -> np.ndarray:
    h, w = a.shape
    side = min(int(round(crop_mm / max(spacing, 1e-3))), h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(a[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def build_stacks(study_id: str, series_rows: list, series_root: str, geo: Geometry, sizes=None):
    """Decode once, emit one stack per stored size. -> {size: (uint8 [n_slices,S,S], mask uint8 [n_slices])}."""
    sizes = tuple(sizes or (geo.stored,))
    stacks = {sz: np.zeros((geo.n_slices, sz, sz), np.uint8) for sz in sizes}
    cursor, used = 0, set()
    for plane, fluid, count in geo.slots:
        row = pick_series(series_rows, plane, fluid, used)
        if row is None:
            cursor += count
            continue
        used.add(row["SeriesInstanceUID"])
        files, med_ps = order_series(f"{series_root}/{study_id}/{row['SeriesInstanceUID']}")
        if not files:
            cursor += count
            continue
        n = len(files)
        lo, hi = int(n * geo.span_lo), max(int(n * geo.span_hi) - 1, int(n * geo.span_lo))
        picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else [0] * count
        slices, spacings = [], []
        for p in picks:
            path, ps = files[min(int(p), n - 1)]
            try:
                slices.append(read_slice(path)); spacings.append(ps)
            except Exception:
                slices.append(None); spacings.append(med_ps)
        good = [s for s in slices if s is not None]
        lo_v, hi_v = np.percentile(np.concatenate([s.ravel() for s in good]), [2.0, 98.0]) if good else (0.0, 1.0)
        for s, ps in zip(slices, spacings):
            if cursor >= geo.n_slices:
                break
            if s is not None:
                win = np.clip((s - lo_v) / (hi_v - lo_v + 1e-6), 0, 1)
                for sz in sizes:
                    stacks[sz][cursor] = (crop_resize(win, ps if ps > 0 else med_ps, geo.crop_mm, sz) * 255).astype(np.uint8)
            cursor += 1
        if cursor >= geo.n_slices:
            break
    out = {}
    for sz, st in stacks.items():
        out[sz] = (st, (st.reshape(geo.n_slices, -1).sum(1) > 0).astype(np.uint8))
    return out


def build_stack(study_id: str, series_rows: list, series_root: str, geo: Geometry):
    """-> (uint8 stack [n_slices, S, S], per-slice validity mask uint8 [n_slices])."""
    return build_stacks(study_id, series_rows, series_root, geo)[geo.stored]


def window_centres(mask: np.ndarray, k: int):
    """Centres of 3-slice windows lying strictly inside the span of valid slices; k evenly spaced of them."""
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi)]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, len(mask) - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]
