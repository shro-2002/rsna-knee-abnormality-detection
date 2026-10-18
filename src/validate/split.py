"""Leakage-safe data splits.

Construction fact (VERIFIED from data audit): the 58 expert-labeled studies and the 4,349
report-only studies are disjoint by construction. We therefore:
  - hold the 58 gold studies as a LOCKED final-check set (used sparingly, with bootstrap CI),
  - train/select on the 4,349 weak (report-derived) studies via grouped K-fold.

Patient/site identifiers are NOT in the CSVs, so folds are grouped by StudyInstanceUID (the
finest guaranteed unit). Patient/site-level disjointness is NOT VERIFIED and must be labeled as
such until DICOM tags are inspected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS


@dataclass
class Labels:
    gold: pd.DataFrame  # 58 expert-labeled (0/1 targets, has Report)
    weak: pd.DataFrame  # report-only (targets NaN, has Report)


def load_labels(train_csv: str) -> Labels:
    df = pd.read_csv(train_csv)
    labeled_mask = df[TARGETS].notna().all(axis=1)
    return Labels(gold=df[labeled_mask].reset_index(drop=True),
                 weak=df[~labeled_mask].reset_index(drop=True))


def group_kfold_indices(df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                        group_col: str = ID_COL):
    """Deterministic grouped K-fold. Groups never split across folds.

    Yields (train_idx, val_idx) into df's positional index.
    """
    groups = df[group_col].to_numpy()
    uniq = np.array(sorted(pd.unique(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_group = {g: i % n_splits for i, g in enumerate(uniq)}
    fold = np.array([fold_of_group[g] for g in groups])
    for k in range(n_splits):
        val_idx = np.where(fold == k)[0]
        train_idx = np.where(fold != k)[0]
        yield train_idx, val_idx


if __name__ == "__main__":
    import os

    raw = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "train.csv")
    lab = load_labels(raw)
    assert len(lab.gold) == 58, len(lab.gold)
    assert len(lab.weak) == 4349, len(lab.weak)
    folds = list(group_kfold_indices(lab.weak, n_splits=5))
    val_sizes = [len(v) for _, v in folds]
    assert sum(val_sizes) == len(lab.weak)
    # no study appears in two val folds
    seen = set()
    for _, v in folds:
        ids = set(lab.weak.iloc[v][ID_COL])
        assert not (ids & seen)
        seen |= ids
    print("split OK: gold=58 weak=4349 fold_val_sizes=", val_sizes)
