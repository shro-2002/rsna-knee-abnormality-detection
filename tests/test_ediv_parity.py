"""Parity of our own Raptor re-implementation vs the original public notebook (used ONLY as test oracle).
Run: pytest tests/test_ediv_parity.py  (needs local data/raw test study + scratch copies of sub #4 notebook/ckpt)."""
import importlib.util, os, sys
import numpy as np, pandas as pd, pytest, torch

ORIG = os.environ.get("RAPTOR_ORIG_NB")
CKPT = os.environ.get("RAPTOR_V4_CKPT")
pytestmark = pytest.mark.skipif(not (ORIG and CKPT), reason="oracle files not provided")
sys.path.insert(0, ".")
from src.ediv import raptor_prep as P
from src.ediv.raptor_model import load_raptor, predict_study

C = "data/raw"


def _oracle():
    spec = importlib.util.spec_from_file_location("orig", ORIG); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _study():
    ts = pd.read_csv(f"{C}/test_series.csv")
    sid = next(s for s in ts.StudyInstanceUID.unique() if os.path.isdir(f"{C}/test_series/{s}") and os.listdir(f"{C}/test_series/{s}"))
    return sid, ts.to_dict("records"), ts


def test_stack_matches_original_v4_geometry():
    o = _oracle(); sid, _, ts = _study()
    SER = {k: v.to_dict("records") for k, v in ts.groupby("StudyInstanceUID")}
    vol_o, mask_o = o.build_study(sid, SER, f"{C}/test_series", o._make_reader())
    vol_m, mask_m = P.build_stack(sid, SER[sid], f"{C}/test_series", P.GEOMETRIES["v4_64_6to94_336_42"])
    assert vol_o.shape == vol_m.shape and mask_o.sum() > 0
    assert np.array_equal(mask_o, mask_m)
    assert np.array_equal(vol_o, vol_m), f"max abs diff {np.abs(vol_o.astype(int) - vol_m.astype(int)).max()}"


def test_probabilities_match_original_v4():
    o = _oracle(); sid, _, ts = _study()
    SER = {k: v.to_dict("records") for k, v in ts.groupby("StudyInstanceUID")}
    vol, mask = o.build_study(sid, SER, f"{C}/test_series", o._make_reader())
    k = 4
    xo = o.eval_windows(vol, mask, k=k, res=384)
    model, res = o.load_model(CKPT, o.ARMS[0]["arch"], 384, "cpu")
    po = o.infer_probs(model, xo, "cpu")
    net, res2 = load_raptor(CKPT, "cpu")
    pm = predict_study(net, vol, mask, P.window_centres(mask, k), res2, "cpu")
    assert np.abs(po - pm).max() < 1e-4, (po, pm)
