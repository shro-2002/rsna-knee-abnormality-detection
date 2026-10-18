"""CPU correctness test for the weak-CV OOF harness (synthetic separable signal)."""
import numpy as np

from src.validate.weak_cv import score_oof, weak_cv_oof


def test_weak_cv_recovers_learnable_signal():
    rng = np.random.default_rng(0)
    m, nslice, dim, t = 120, 4, 8, 12
    wF = rng.standard_normal((m, nslice, dim)).astype(np.float32)
    wS = np.ones((m, nslice), dtype=np.float32)
    # target 0 is a learnable function of feature mean; others are pure noise labels
    signal = wF[:, :, 0].mean(axis=1)  # linear in one feature dim -> head can capture it
    wY = np.zeros((m, t), dtype=np.float32)
    wY[:, 0] = (signal > np.median(signal)).astype(np.float32)
    wY[:, 1:] = (rng.random((m, t - 1)) > 0.5).astype(np.float32)
    wT = np.ones((m, t), dtype=np.float32)
    oof = weak_cv_oof(wF, wS, wY, wT, dim=dim, n_folds=3, epochs=60)
    assert oof.shape == (m, t)
    res = score_oof(oof, wY, wT, n_boot=200)
    # the learnable target should be recovered above chance OOF (noise targets ~0.5)
    assert res["per_target"][0] > 0.70
    assert 0.0 <= res["point"] <= 1.0
