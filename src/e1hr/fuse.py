"""FUSE-NX: fold exclusion, strict teacher-OOF merge and fused soft targets (research/30 §7).

Fused target (fixed before any student result): y = 0.5 * report_mean + 0.5 * teacher_oof_probability, per cell,
all 12 targets. report_mean is src.e1.labels.train_targets (prior-filled, in [0,1]); teacher probabilities are
sigmoid outputs in [0,1]; so y is in [0,1] and valid for BCE. A training study without a teacher row is an error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS

FUSE_WEIGHT = 0.5


def parse_folds(text: str) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip() != ""}


def merge_teacher_oof(paths: list[str], expected: dict[int, set[str]]) -> pd.DataFrame:
    """Merge one OOF file per held-out fold. expected: fold -> study IDs that fold must contain exactly."""
    if len(paths) != len(expected):
        raise ValueError(f"{len(paths)} teacher files for {len(expected)} folds")
    parts = []
    for path, (fold, ids) in zip(paths, sorted(expected.items())):
        df = pd.read_csv(path)
        missing_cols = [t for t in TARGETS if t not in df.columns]
        if missing_cols:
            raise ValueError(f"{path}: missing targets {missing_cols}")
        got = set(df[ID_COL])
        if df[ID_COL].duplicated().any() or got != ids:
            raise ValueError(f"{path}: IDs do not equal fold {fold} ({len(got)} vs {len(ids)}, dup={df[ID_COL].duplicated().sum()})")
        vals = df[TARGETS].to_numpy(float)
        if not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
            raise ValueError(f"{path}: predictions not finite in [0,1]")
        parts.append(df[[ID_COL] + TARGETS].assign(teacher_fold=fold))
    merged = pd.concat(parts, ignore_index=True)
    if merged[ID_COL].duplicated().any():
        raise ValueError("a study appears in two teacher folds")
    return merged.set_index(ID_COL)


def fuse_targets(report: np.ndarray, sids: list[str], teacher: pd.DataFrame) -> np.ndarray:
    absent = [s for s in sids if s not in teacher.index]
    if absent:
        raise ValueError(f"{len(absent)} training studies have no teacher OOF prediction")
    p = teacher.loc[sids, TARGETS].to_numpy(np.float32)
    fused = (1 - FUSE_WEIGHT) * report + FUSE_WEIGHT * p
    assert np.isfinite(fused).all() and fused.min() >= 0 and fused.max() <= 1
    return fused.astype(np.float32)


def teacher_sanity(oof: pd.DataFrame, min_std: float = 0.01, prev_range=(0.01, 0.99)) -> list[str]:
    """Operational checks that do not penalise disagreement with report labels. [] = pass."""
    vals = oof[TARGETS].to_numpy(float)
    problems = []
    if not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        problems.append("non-finite or out-of-range predictions")
    for t, sd, mu in zip(TARGETS, vals.std(0), vals.mean(0)):
        if sd < min_std:
            problems.append(f"{t}: collapsed (std {sd:.4f})")
        if not prev_range[0] <= mu <= prev_range[1]:
            problems.append(f"{t}: mean {mu:.3f} outside {prev_range}")
    return problems


def main() -> None:
    import argparse
    import json

    from src.validate.split import group_kfold_indices, load_labels

    ap = argparse.ArgumentParser(description="merge 4 FUSE-NX teacher OOF files (folds 1-4, in order)")
    ap.add_argument("--teacher-oof", nargs=4, required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weak = load_labels(a.train_csv).weak.reset_index(drop=True)
    folds = {k: set(weak[ID_COL].iloc[v]) for k, (_, v) in enumerate(group_kfold_indices(weak, 5, seed=0))}
    merged = merge_teacher_oof(a.teacher_oof, {k: folds[k] for k in (1, 2, 3, 4)})
    report = {f"fold{k}": teacher_sanity(merged[merged.teacher_fold == k]) for k in (1, 2, 3, 4)}
    if any(report.values()):
        raise SystemExit(f"teacher sanity failed: {json.dumps(report)}")
    merged.drop(columns="teacher_fold").reset_index().to_csv(a.out, index=False)
    print("FUSE MERGE OK", len(merged), json.dumps(report))


if __name__ == "__main__":
    main()
