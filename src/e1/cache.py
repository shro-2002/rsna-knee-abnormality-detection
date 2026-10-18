"""E1 slice cache: one uint8 volume (3 planes x K slices x S x S) per study.

Same function serves training cache and test-time inference, so preprocessing cannot drift.
Per plane (Sagittal, Coronal, Axial) one series is chosen: Fluid_Sensitive=1 first, then CSV
order. Slices are geometry-ordered from headers only, K are sampled evenly over the central 80%
(duplicates when the series is shorter than K), and only those K are pixel-decoded. Intensity:
0.5-99.5 percentile window over the K sampled slices, then resize to S x S.

CLI (CPU Kaggle kernel): python -m src.e1.cache --data-root ... --split train --out /kaggle/working
writes e1_cache.npy (N,3,K,S,S uint8, memmap), e1_index.csv (StudyInstanceUID, row, plane mask,
error) and e1_cache_meta.json.
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
from src.data.dicom import DecodeError, _slice_normal, load_dicom, normalize_series, resize

PLANES = ("Sagittal", "Coronal", "Axial")
K_SLICES = 16
SIZE = 224
CENTRAL = (0.1, 0.9)


def select_series(series_rows: pd.DataFrame) -> dict[str, str]:
    """plane -> SeriesInstanceUID, preferring Fluid_Sensitive=1, then CSV order."""
    chosen = {}
    for plane in PLANES:
        rows = series_rows[series_rows["Anatomical_Plane"] == plane]
        if rows.empty:
            continue
        rows = rows.sort_values("Fluid_Sensitive", ascending=False, kind="stable")
        chosen[plane] = rows.iloc[0]["SeriesInstanceUID"]
    return chosen


def central_indices(n: int, k: int = K_SLICES) -> list[int]:
    lo, hi = round(CENTRAL[0] * (n - 1)), round(CENTRAL[1] * (n - 1))
    return list(np.linspace(lo, hi, k).round().astype(int))


def _ordered_paths(series_dir: str) -> list[str]:
    paths = sorted(p for p in os.listdir(series_dir) if p.endswith(".dcm"))
    if not paths:
        raise DecodeError(f"no DICOMs in {series_dir}")
    keyed = []
    for p in paths:
        full = os.path.join(series_dir, p)
        d = pydicom.dcmread(full, stop_before_pixels=True)
        iop = getattr(d, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
        ipp = getattr(d, "ImagePositionPatient", [0, 0, 0])
        pos = float(np.dot(np.array(ipp, dtype=float), _slice_normal(iop)))
        keyed.append((pos, int(getattr(d, "InstanceNumber", 0) or 0), full))
    if len({round(k[0], 3) for k in keyed}) > 1:
        keyed.sort(key=lambda k: k[0])
    else:
        keyed.sort(key=lambda k: k[1])
    return [k[2] for k in keyed]


def series_slices(series_dir: str, k: int = K_SLICES, size: int = SIZE) -> np.ndarray:
    """(k, size, size) uint8 for one series."""
    paths = _ordered_paths(series_dir)
    picked = [load_dicom(paths[i]).pixels for i in central_indices(len(paths), k)]
    lo, hi = np.percentile(np.concatenate([p.ravel() for p in picked]), [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = np.empty((k, size, size), dtype=np.uint8)
    for j, px in enumerate(picked):
        norm = np.clip((px - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        out[j] = np.round(resize(norm, size) * 255.0).astype(np.uint8)
    return out


def study_volume(study_dir: str, series_rows: pd.DataFrame, k: int = K_SLICES,
                 size: int = SIZE) -> tuple[np.ndarray, np.ndarray, str]:
    """(3,k,size,size) uint8, plane mask (3,), error text ('' when every chosen plane decoded)."""
    vol = np.zeros((len(PLANES), k, size, size), dtype=np.uint8)
    pmask = np.zeros(len(PLANES), dtype=np.uint8)
    errors = []
    chosen = select_series(series_rows)
    for p, plane in enumerate(PLANES):
        uid = chosen.get(plane)
        if uid is None:
            continue
        try:
            vol[p] = series_slices(os.path.join(study_dir, uid), k, size)
            pmask[p] = 1
        except (DecodeError, OSError, ValueError) as e:
            errors.append(f"{plane}: {e}")
    return vol, pmask, "; ".join(errors)


def _work(args):
    i, study_dir, rows, k, size = args
    try:
        vol, pmask, err = study_volume(study_dir, rows, k, size)
    except Exception as e:  # noqa: BLE001 - logged per study, never silently dropped
        vol = np.zeros((len(PLANES), k, size, size), dtype=np.uint8)
        pmask, err = np.zeros(len(PLANES), dtype=np.uint8), f"fatal: {e}"
    return i, vol, pmask, err


def build_cache(sids: list[str], series_df: pd.DataFrame, split_dir: str, out_dir: str,
                k: int = K_SLICES, size: int = SIZE, workers: int = 4, log=print) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    arr = np.lib.format.open_memmap(os.path.join(out_dir, "e1_cache.npy"), mode="w+",
                                    dtype=np.uint8, shape=(len(sids), len(PLANES), k, size, size))
    groups = dict(tuple(series_df.groupby(ID_COL)))
    jobs = [(i, os.path.join(split_dir, sid), groups.get(sid, series_df.iloc[:0]), k, size)
            for i, sid in enumerate(sids)]
    index = [None] * len(sids)
    t0 = time.time()
    with Pool(workers) as pool:
        for n, (i, vol, pmask, err) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            arr[i] = vol
            index[i] = {ID_COL: sids[i], "row": i, **{f"has_{p}": int(m) for p, m in zip(PLANES, pmask)},
                        "error": err}
            if (n + 1) % 200 == 0:
                log(f"cached {n+1}/{len(sids)} {(time.time()-t0)/(n+1):.2f}s/study")
    arr.flush()
    idx = pd.DataFrame(index)
    idx.to_csv(os.path.join(out_dir, "e1_index.csv"), index=False)
    meta = {"n": len(sids), "planes": PLANES, "k": k, "size": size, "central": CENTRAL,
            "wall_s": round(time.time() - t0, 1), "n_error": int((idx["error"] != "").sum())}
    with open(os.path.join(out_dir, "e1_cache_meta.json"), "w") as f:
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
    if a.limit:
        ids = ids[: a.limit]
    series = pd.read_csv(os.path.join(a.data_root, f"{a.split}_series.csv"))
    build_cache(ids, series, image_dir(a.data_root, a.split), a.out, workers=a.workers)


def image_dir(root: str, split: str) -> str:
    for name in (f"{split}_series", f"{split}_images"):
        if os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    raise FileNotFoundError(f"no {split} image directory under {root}")


if __name__ == "__main__":
    main()
