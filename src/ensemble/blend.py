"""Fractional-rank blending for decorrelated model arms.

The competition metric is ROC-AUC (ranking), so we average per-target fractional RANKS, not raw
probabilities — this is invariant to each model's calibration and preserves monotonic order
(Gemini audit Claim C: blending only helps when arms are decorrelated, which EXP-004 EffB0 and
EXP-005 DINOv2 are — EffB0 wins diffuse targets, DINOv2 wins focal).
"""
from __future__ import annotations

import numpy as np


def rank_blend(preds_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """preds_list: K arrays of shape (N,T) aligned on the same N studies. -> (N,T) blended ranks.

    Each array is converted to per-target fractional ranks in [0,1], then (weighted) averaged.
    """
    if not preds_list:
        raise ValueError("empty preds_list")
    arrays = [np.asarray(p, dtype=float) for p in preds_list]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all prediction arrays must share shape")
    n = shape[0]
    w = [1.0] * len(arrays) if weights is None else list(weights)
    if len(w) != len(arrays):
        raise ValueError("weights length mismatch")
    ranks = []
    for a in arrays:
        r = np.argsort(np.argsort(a, axis=0), axis=0).astype(float)
        if n > 1:
            r /= (n - 1)
        ranks.append(r)
    return np.average(ranks, axis=0, weights=w)
