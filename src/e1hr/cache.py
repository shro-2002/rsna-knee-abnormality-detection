"""E1-HR crop cache: 100 mm knee-centred crops, 24 slices per plane, 224 px (research/21).

Reuses E1's series choice and geometric slice ordering (src/e1/cache.py) unchanged. New here:
an anatomy-model-free crop: a CROP_MM physical square (PixelSpacing) at the IMAGE CENTRE
(knee MRI is acquired centred on the joint). An intensity-weighted Otsu-foreground centroid was
tried first and rejected on real studies before any training: bright femoral marrow pulled the
box off the joint line (research/22). The centroid is kept only as a label-free QA statistic:
`offset` = centroid distance from the image centre as a fraction of the crop side.
Fallbacks (flagged per plane in the index):
  "spacing"  no PixelSpacing   -> side = 2/3 of the short image side
  "fov"      crop >= image     -> whole image
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom

from src.constants import ID_COL
from src.data.dicom import load_dicom, resize
from src.e1.cache import PLANES, _ordered_paths, central_indices, image_dir, select_series

K_SLICES = 24
SIZE = 224
CROP_MM = 100.0


def otsu_threshold(img: np.ndarray, bins: int = 128) -> float:
    hist, edges = np.histogram(img, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def foreground_offset(px: np.ndarray, side: int) -> float:
    """QA only: Otsu-foreground centroid distance from the image centre, in units of crop side."""
    step = max(1, min(px.shape) // 128)
    small = px[::step, ::step].astype(np.float32)
    ys, xs = np.nonzero(small > otsu_threshold(small))
    if len(ys) == 0:
        return float("nan")
    dy, dx = ys.mean() * step - px.shape[0] / 2, xs.mean() * step - px.shape[1] / 2
    return float(np.hypot(dy, dx) / max(side, 1))


def crop_box(px: np.ndarray, spacing_mm: float | None, crop_mm: float = CROP_MM) -> tuple[tuple, str]:
    """Image-centred square (r0, r1, c0, c1) of crop_mm, and the fallback reason ('' if none)."""
    h, w = px.shape
    reason = ""
    if spacing_mm and spacing_mm > 0:
        side = int(round(crop_mm / spacing_mm))
    else:
        side, reason = int(min(h, w) * 2 / 3), "spacing"
    if side >= min(h, w):
        return (0, h, 0, w), "fov"
    r0, c0 = (h - side) // 2, (w - side) // 2
    return (r0, r0 + side, c0, c0 + side), reason


def _spacing(path: str) -> float | None:
    ps = getattr(pydicom.dcmread(path, stop_before_pixels=True), "PixelSpacing", None)
    return float(ps[0]) if ps else None


def series_crop_slices(series_dir: str, k: int = K_SLICES, size: int = SIZE,
                       crop_mm: float = CROP_MM) -> tuple[np.ndarray, dict]:
    """(k, size, size) uint8 crops for one series + info (fallback, crop fraction, n_slices)."""
    paths = _ordered_paths(series_dir)
    idx = central_indices(len(paths), k)
    picked = [load_dicom(paths[i]).pixels for i in idx]
    mid = picked[len(picked) // 2]
    (r0, r1, c0, c1), reason = crop_box(mid, _spacing(paths[idx[len(idx) // 2]]), crop_mm)
    crops = [p[r0:r1, c0:c1] if p.shape == mid.shape else p for p in picked]  # odd-shaped slice: keep whole
    lo, hi = np.percentile(np.concatenate([c.ravel() for c in crops]), [0.5, 99.5])
    hi = hi if hi > lo else lo + 1.0
    out = np.empty((k, size, size), dtype=np.uint8)
    for j, c in enumerate(crops):
        out[j] = np.round(resize(np.clip((c - lo) / (hi - lo), 0, 1).astype(np.float32), size) * 255)
    info = {"fallback": reason, "crop_frac": round((r1 - r0) / min(mid.shape), 3), "n_slices": len(paths),
            "offset": round(foreground_offset(mid, r1 - r0), 3)}
    return out, info


def study_volume_hr(study_dir: str, series_rows: pd.DataFrame, k: int = K_SLICES,
                    size: int = SIZE) -> tuple[np.ndarray, dict]:
    vol = np.zeros((len(PLANES), k, size, size), dtype=np.uint8)
    rec, errors = {}, []
    chosen = select_series(series_rows)
    for p, plane in enumerate(PLANES):
        rec[f"has_{plane}"] = 0
        uid = chosen.get(plane)
        if uid is None:
            continue
        try:
            vol[p], info = series_crop_slices(os.path.join(study_dir, uid), k, size)
            rec.update({f"has_{plane}": 1, **{f"{key}_{plane}": v for key, v in info.items()}})
        except Exception as e:  # noqa: BLE001 - recorded per plane, study never dropped silently
            errors.append(f"{plane}: {e}")
    rec["error"] = "; ".join(errors)
    return vol, rec


def _work(args):
    i, study_dir, rows, k, size = args
    vol, rec = study_volume_hr(study_dir, rows, k, size)
    return i, vol, rec


def build_cache_hr(sids: list[str], series_df: pd.DataFrame, split_dir: str, out_dir: str,
                   k: int = K_SLICES, size: int = SIZE, workers: int = 4, log=print) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    arr = np.lib.format.open_memmap(os.path.join(out_dir, "e1hr_cache.npy"), mode="w+",
                                    dtype=np.uint8, shape=(len(sids), len(PLANES), k, size, size))
    groups = dict(tuple(series_df.groupby(ID_COL)))
    jobs = [(i, os.path.join(split_dir, s), groups.get(s, series_df.iloc[:0]), k, size) for i, s in enumerate(sids)]
    index, t0 = [None] * len(sids), time.time()
    with Pool(workers) as pool:
        for n, (i, vol, rec) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            arr[i] = vol
            index[i] = {ID_COL: sids[i], "row": i, **rec}
            if (n + 1) % 200 == 0:
                log(f"cached {n+1}/{len(sids)} {(time.time()-t0)/(n+1):.2f}s/study")
    arr.flush()
    idx = pd.DataFrame(index)
    idx.to_csv(os.path.join(out_dir, "e1hr_index.csv"), index=False)
    fb = {p: int(idx.get(f"fallback_{p}", pd.Series(dtype=str)).fillna("").ne("").sum()) for p in PLANES}
    meta = {"n": len(sids), "planes": PLANES, "k": k, "size": size, "crop_mm": CROP_MM,
            "wall_s": round(time.time() - t0, 1), "n_error": int(idx["error"].ne("").sum()), "fallbacks": fb}
    with open(os.path.join(out_dir, "e1hr_cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(json.dumps(meta))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ids = pd.read_csv(os.path.join(a.data_root, f"{a.split}.csv"))[ID_COL].tolist()
    ids = ids[: a.limit] if a.limit else ids
    series = pd.read_csv(os.path.join(a.data_root, f"{a.split}_series.csv"))
    build_cache_hr(ids, series, image_dir(a.data_root, a.split), a.out, workers=a.workers)


if __name__ == "__main__":
    main()
