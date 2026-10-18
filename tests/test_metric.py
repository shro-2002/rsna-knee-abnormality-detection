"""CPU unit tests for the competition metric."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.validate.metric import bootstrap_macro_auc, macro_auc, roc_auc  # noqa: E402


def test_perfect_separation_auc_is_one():
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0


def test_reversed_is_zero():
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(y, s) == 0.0


def test_random_tie_is_half():
    y = np.array([0, 1, 0, 1], dtype=float)
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(roc_auc(y, s) - 0.5) < 1e-9


def test_known_value_matches_mannwhitney():
    # 1 neg below, 1 neg above a single pos -> AUC = 0.5
    y = np.array([0, 1, 0], dtype=float)
    s = np.array([0.1, 0.5, 0.9])
    assert abs(roc_auc(y, s) - 0.5) < 1e-9


def test_undefined_target_is_nan():
    y = np.array([1, 1, 1], dtype=float)  # no negatives
    s = np.array([0.2, 0.5, 0.9])
    assert np.isnan(roc_auc(y, s))


def test_macro_skips_nan_and_masks_missing():
    # 3 studies, 2 targets; target 1 has a missing (NaN) label row
    y = np.array([[0, 0.0], [1, np.nan], [1, 1.0]], dtype=float)
    p = np.array([[0.1, 0.2], [0.9, 0.9], [0.8, 0.7]])
    m = macro_auc(y, p)
    assert 0.0 <= m <= 1.0 and not np.isnan(m)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 3)) > 0.5).astype(float)
    p = y * 0.7 + rng.random((200, 3)) * 0.3  # correlated -> AUC > 0.5
    res = bootstrap_macro_auc(y, p, n_boot=300, seed=1)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]
    assert res["point"] > 0.5


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(["python", "-m", "pytest", __file__, "-q"]))
