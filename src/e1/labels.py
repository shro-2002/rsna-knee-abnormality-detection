"""Training targets and the frozen OOF reference from public CC0 report-label sets.

Sources (all CC0, derived only from competition train.csv reports; see research/license_log.md):
  steven   stevenleehans/rsna-knee-llm-report-labels  llm_labels_full.csv
  pilkwang pilkwang/rsna-knee-llm-labels              report_labels_v2.csv
  dread    dreaddevelopment/rsna-knee-labels          labels_llm_soft.csv
  nartaa   nartaa/rsna-knee-hpo-assets                labels_v1_blend.parquet
  jev      riadmohamed42/jev-knee-labels              FINAL2_labels_scores.csv  (text ensemble only;
           the HYBRID file re-uses steven/pilkwang columns, so it is not an independent source)

Training target = per-cell mean over available sources; a cell no source covers gets the column
prior with weight 0. The OOF reference is FROZEN across all E1/E2 variants: a cell is kept only
where >= MIN_AGREE non-abstaining sources vote the same way, so changing the training labels
cannot move the evaluation target.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS

SOURCE_FILES = {
    "steven": "llm_labels_full.csv",
    "pilkwang": "report_labels_v2.csv",
    "dread": "labels_llm_soft.csv",
    "nartaa": "labels_v1_blend.parquet",
    "jev": "FINAL2_labels_scores.csv",
}
ABSTAIN_LO, ABSTAIN_HI = 0.4, 0.6   # a source value in this band is "unsure" for the reference
MIN_AGREE = 4                       # same-direction votes needed to keep a reference cell


IMAGE_DIRS = {"train_series", "test_series", "train_images", "test_images"}


def find_source_files(roots: list[str]) -> dict[str, str]:
    """Locate each source file under any root. Image trees (~820k DICOMs) are pruned from the walk.
    First hit in sorted walk order wins; missing sources are skipped."""
    wanted = {fname: name for name, fname in SOURCE_FILES.items()}
    found: dict[str, str] = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames[:] = sorted(d for d in dirnames if d not in IMAGE_DIRS)
            for fname in sorted(set(filenames) & set(wanted)):
                found.setdefault(wanted[fname], os.path.join(dirpath, fname))
    return found


def _read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if ID_COL not in df.columns:
        df = df.reset_index()
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing target columns {missing}")
    df = df.drop_duplicates(ID_COL).set_index(ID_COL)[TARGETS].astype(float)
    return df.clip(0.0, 1.0)


def load_sources(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {name: _read(p) for name, p in paths.items()}


def _stack(sources: dict[str, pd.DataFrame], sids: list[str]) -> np.ndarray:
    """(S, N, 12) float with NaN where a source lacks a study or a cell."""
    return np.stack([df.reindex(sids)[TARGETS].to_numpy(float) for df in sources.values()])


def train_targets(sources: dict[str, pd.DataFrame], sids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Soft targets (N,12) and cell weights (N,12): mean of sources; uncovered cells -> prior, w=0."""
    if not sources:
        raise ValueError("no label sources")
    x = _stack(sources, sids)
    covered = ~np.isnan(x).all(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells are handled below
        y = np.nanmean(x, axis=0)
    prior = np.nanmean(np.where(covered, y, np.nan), axis=0)
    y = np.where(covered, y, prior[None, :])
    return y.astype(np.float32), covered.astype(np.float32)


def reference_labels(sources: dict[str, pd.DataFrame], sids: list[str],
                     min_agree: int = MIN_AGREE) -> tuple[np.ndarray, np.ndarray]:
    """Binary reference (N,12) and mask (N,12) of cells with >= min_agree agreeing votes."""
    x = _stack(sources, sids)
    unsure = np.isnan(x) | ((x > ABSTAIN_LO) & (x < ABSTAIN_HI))
    pos = ((x >= ABSTAIN_HI) & ~unsure).sum(axis=0)
    neg = ((x <= ABSTAIN_LO) & ~unsure).sum(axis=0)
    ref = (pos > neg).astype(np.float32)
    mask = (np.maximum(pos, neg) >= min_agree) & (pos != neg)
    return ref, mask.astype(np.float32)
