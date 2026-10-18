"""Competition metric: macro mean of per-target ROC-AUC, with NaN-masking and bootstrap CI.

ROC-AUC is computed via the Mann-Whitney U identity (numpy only, no sklearn), with average
ranks for ties. A target with fewer than one positive AND one negative among non-NaN rows has
an undefined AUC and is reported as NaN; the macro mean is taken over defined targets only.

The exact behavior of the hidden Kaggle scorer on degenerate columns is UNKNOWN; on the 58 gold
studies every target has both classes, so the local metric is well defined there.
"""
from __future__ import annotations

import numpy as np

_MIN_PER_CLASS = 1


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC for one binary target. NaN if undefined. Ties handled by average rank."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(y_true)
    y_true, y_score = y_true[mask], y_score[mask]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos < _MIN_PER_CLASS or n_neg < _MIN_PER_CLASS:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranked = y_score[order]
    ranks = np.empty(len(ranked), dtype=float)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1] == ranked[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0  # average rank, 1-based
        i = j + 1
    rank_of_true = np.empty(len(ranked), dtype=float)
    rank_of_true[order] = ranks
    sum_pos_ranks = rank_of_true[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def per_target_auc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target AUCs, shape (n_targets,). NaN where undefined."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.array([roc_auc(y_true[:, k], y_pred[:, k]) for k in range(y_true.shape[1])])


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-target AUC over defined targets."""
    aucs = per_target_auc(y_true, y_pred)
    defined = aucs[~np.isnan(aucs)]
    return float(defined.mean()) if len(defined) else float("nan")


def bootstrap_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Study-resampled bootstrap of macro AUC. Returns point, mean, and (alpha/2, 1-alpha/2) CI."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = macro_auc(y_true, y_pred)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = macro_auc(y_true[idx], y_pred[idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "boot_mean": float(boots.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(len(boots)),
    }
