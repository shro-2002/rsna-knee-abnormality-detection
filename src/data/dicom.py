"""Deterministic DICOM decode + MRI preprocessing.

Discipline (from data audit + DICOM standard):
  - apply RescaleSlope/Intercept BEFORE normalization; handle non-unity slopes;
  - invert MONOCHROME1; MR has no fixed intensity scale -> per-series percentile normalization;
  - order slices by true geometry (ImagePositionPatient projected on the slice normal from
    ImageOrientationPatient), never by SOP UID or InstanceNumber alone;
  - bounded, evenly-spaced slice sampling for 2.5D/MIL;
  - a decode failure raises DecodeError (caller logs + emits a deterministic fallback row;
    studies are never silently dropped).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pydicom
from PIL import Image


class DecodeError(Exception):
    pass


@dataclass
class Slice:
    pixels: np.ndarray  # float32, rescaled, HxW
    position: float     # projection on slice normal (for ordering)
    instance: int


def _slice_normal(iop) -> np.ndarray:
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:6], dtype=float)
    return np.cross(row, col)


def load_dicom(path: str) -> Slice:
    try:
        d = pydicom.dcmread(path)
        px = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1) or 1)
        intercept = float(getattr(d, "RescaleIntercept", 0) or 0)
        px = px * slope + intercept
        if str(getattr(d, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            px = px.max() - px
        iop = getattr(d, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
        ipp = getattr(d, "ImagePositionPatient", [0, 0, 0])
        pos = float(np.dot(np.array(ipp, dtype=float), _slice_normal(iop)))
        inst = int(getattr(d, "InstanceNumber", 0) or 0)
        return Slice(pixels=px, position=pos, instance=inst)
    except Exception as e:  # noqa: BLE001
        raise DecodeError(f"{path}: {e}") from e


def load_series(series_dir: str) -> list[Slice]:
    paths = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not paths:
        raise DecodeError(f"no DICOMs in {series_dir}")
    slices = [load_dicom(p) for p in paths]
    # geometric order; fall back to InstanceNumber if positions are degenerate
    if len({round(s.position, 3) for s in slices}) > 1:
        slices.sort(key=lambda s: s.position)
    else:
        slices.sort(key=lambda s: s.instance)
    return slices


def sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return list(np.linspace(0, n - 1, k).round().astype(int))


def normalize_series(stack: np.ndarray, lo_p: float = 0.5, hi_p: float = 99.5) -> np.ndarray:
    """Per-series percentile normalization to [0,1]. stack: (S,H,W)."""
    lo, hi = np.percentile(stack, [lo_p, hi_p])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((stack - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a single HxW float[0,1] image to size x size (bilinear)."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def series_to_array(series_dir: str, k_slices: int = 16, size: int = 224) -> np.ndarray:
    """Decode a series -> (k, size, size) float32 in [0,1], geometry-ordered + sampled."""
    slices = load_series(series_dir)
    idx = sample_indices(len(slices), k_slices)
    raw = np.stack([slices[i].pixels for i in idx])  # (k,H,W) possibly ragged? same series ~ same HxW
    norm = normalize_series(raw)
    return np.stack([resize(norm[j], size) for j in range(norm.shape[0])])


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_3ch_imagenet(gray: np.ndarray) -> np.ndarray:
    """(H,W) [0,1] -> (3,H,W) ImageNet-normalized by REPLICATING grayscale (fake 2.5D).

    Kept for exact EXP-004/EXP-005 reproduction. New experiments should use true 2.5D
    (adjacent_windows / series_to_windows), which encode cross-slice context in the channels.
    """
    rgb = np.repeat(gray[None, :, :], 3, axis=0)
    return (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def imagenet_norm_chw(chw: np.ndarray) -> np.ndarray:
    """(3,H,W) [0,1] -> ImageNet-normalized (no channel replication)."""
    return (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def adjacent_windows(norm: np.ndarray, centers: list[int]) -> np.ndarray:
    """Build true-2.5D windows: for each center c, stack slices (c-1, c, c+1) as channels.

    norm: (n,H,W) in [0,1] (whole geometry-ordered series). Edges clamp (first/last repeat the
    center's existing neighbour). Returns (k,3,H,W) in [0,1].
    """
    n = norm.shape[0]
    out = []
    for c in centers:
        lo, hi = max(0, c - 1), min(n - 1, c + 1)
        out.append(np.stack([norm[lo], norm[c], norm[hi]]))  # (3,H,W)
    return np.stack(out).astype(np.float32)


def series_to_windows(series_dir: str, k_slices: int = 8, size: int = 224) -> np.ndarray:
    """Decode a series -> (k,3,size,size) true-2.5D windows in [0,1], geometry-ordered + sampled.

    Normalizes over the WHOLE series (so neighbours are on the same intensity scale), then samples
    k window centers and resizes each of the 3 channels.
    """
    slices = load_series(series_dir)
    full = np.stack([s.pixels for s in slices])          # (n,H,W)
    norm = normalize_series(full)
    centers = sample_indices(len(slices), k_slices)
    win = adjacent_windows(norm, centers)                # (k,3,H,W) native res
    k = win.shape[0]
    out = np.empty((k, 3, size, size), dtype=np.float32)
    for i in range(k):
        for ch in range(3):
            out[i, ch] = resize(win[i, ch], size)
    return out
