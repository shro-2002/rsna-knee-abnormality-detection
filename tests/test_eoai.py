"""E-OAI-1 (research/37): frozen 50/50 blend and nartaa-arm window semantics (crop 320, anatomical mirror)."""
import numpy as np
import torch

from src.ediv import raptor_model as RM
from src.ediv import raptor_prep as P
from src.ediv import run_ensemble as R
from src.ensemble.blend import rank_blend


class _Capture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.clone())
        return torch.zeros(12)


def test_nartaa_geometry_matches_published_constants():
    g = P.GEOMETRIES["nartaa_96_2to98_384_94"]
    assert g.n_slices == 96 and g.stored == 384 and g.k_eval == 94 and (g.span_lo, g.span_hi) == (0.02, 0.98)
    assert [s[2] for s in g.slots] == [26, 22, 18, 12, 18]


def test_mirror_crop_windows():
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 256, (96, 384, 384), dtype=np.uint8)
    centres = [5, 60]                                  # slot 0 (sagittal) and slot 3 (coronal)
    net = _Capture()
    RM.predict_study_mirror_crop(net, stack, centres, 320, [26, 22, 18, 12, 18], (0, 1), "cpu")
    plain, mirrored = net.seen
    assert plain.shape == (2, 3, 320, 320)
    mean = torch.tensor(RM._MEAN).view(1, 3, 1, 1)
    std = torch.tensor(RM._STD).view(1, 3, 1, 1)
    raw = lambda t: torch.round((t * std + mean) * 255).to(torch.uint8)
    tri = torch.from_numpy(np.stack([stack[[c - 1, c, c + 1], 32:352, 32:352] for c in centres]))
    assert torch.equal(raw(plain), tri)
    assert torch.equal(raw(mirrored)[0], tri[0].flip(0))      # sagittal window: slice order reversed
    assert torch.equal(raw(mirrored)[1], tri[1].flip(-1))     # coronal window: width flipped


def test_eoai_blend_is_frozen_half_half():
    rng = np.random.default_rng(1)
    probs = {k: rng.random((40, 12)) for k in R.ALL_ARMS + ["nartaa_oai"]}
    ediv = rank_blend([probs[k] for k in R.ALL_ARMS])
    assert np.allclose(R.final_blend(probs, "ediv"), ediv)
    assert R.EOAI_WEIGHTS == (0.5, 0.5)
    assert np.allclose(R.final_blend(probs, "eoai"), rank_blend([ediv, probs["nartaa_oai"]], [0.5, 0.5]))
