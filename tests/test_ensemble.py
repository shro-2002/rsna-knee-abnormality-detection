"""Tests for rank-blend + multi-backbone checkpoint loading (uses real EXP-004/005 ckpts if present)."""
import os

import numpy as np
import pytest
import torch

from src.ensemble.blend import rank_blend
from src.infer.predict import load_model

SC = os.environ.get("RSNA_CKPT_DIR", "checkpoints")   # own EXP-004/005 checkpoints are not distributed
CK4 = f"{SC}/exp004_out/model_exp004.pt"
CK5 = f"{SC}/exp005_out/model_exp005.pt"


def test_rank_blend_shape_and_monotone():
    a = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    b = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    out = rank_blend([a, b])
    assert out.shape == (3, 2)
    assert (out >= 0).all() and (out <= 1).all()
    # column 0: a ranks ascending, b ranks descending -> blended middle row stays mid
    assert abs(out[1, 0] - 0.5) < 1e-9


def test_rank_blend_identity_single():
    a = np.array([[0.2], [0.5], [0.9]])
    out = rank_blend([a])
    assert np.allclose(out[:, 0], [0.0, 0.5, 1.0])  # fractional ranks


def test_rank_blend_rejects_mismatch():
    with pytest.raises(ValueError):
        rank_blend([np.zeros((3, 2)), np.zeros((4, 2))])


@pytest.mark.skipif(not os.path.exists(CK4), reason="EXP-004 ckpt not downloaded")
def test_load_effb0_checkpoint():
    m = load_model(CK4, "cpu", backbone="efficientnet_b0")
    x = torch.randn(1, 3, 3, 224, 224)
    out = m(x, torch.ones(1, 3))
    assert out.shape == (1, 12)


@pytest.mark.skipif(not os.path.exists(CK5), reason="EXP-005 ckpt not downloaded")
def test_load_dinov2_checkpoint():
    m = load_model(CK5, "cpu", backbone="vit_small_patch14_dinov2", img_size=224)
    x = torch.randn(1, 3, 3, 224, 224)
    out = m(x, torch.ones(1, 3))
    assert out.shape == (1, 12)
