"""End-to-end CPU smoke test of the submission path, with no competition data and no third-party weights.

Builds a synthetic competition folder (2 studies of generated DICOMs) and three random-weight checkpoints in the
Raptor checkpoint format (tiny timm backbone), then runs the real CLI exactly as the Kaggle wrapper does and checks
submission.csv. Covers: file discovery, series selection, stack building, windowing, sharded worker, rank blend,
submission validation. Does not cover: the ConvNeXt arm (needs goodpjw2008's knee.py) or model quality.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from src.ediv import raptor_model as RM
from src.ediv import run_ensemble as R
from tests.test_e1 import _write_study

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["raptor_v5", "raptor_v10", "raptor_v9"]
TINY_ARCH = "resnet10t"
TINY_RES = 64


def _fake_competition(root):
    series_root = root / "comp" / "test_series"
    ids, rows = ["1.2.826.0.1.1", "1.2.826.0.1.2"], []
    for sid in ids:
        rows += _write_study(str(series_root), sid, ["Sagittal", "Coronal", "Axial"], n_slices=8)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(root / "comp" / "test.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "comp" / "test_series.csv", index=False)
    sample = pd.DataFrame(0.5, index=range(len(ids)), columns=RM.LABELS)
    sample.insert(0, "StudyInstanceUID", ids)
    sample.to_csv(root / "comp" / "sample_submission.csv", index=False)
    return ids


def _fake_checkpoints(root):
    (root / "weights").mkdir()
    torch.manual_seed(0)
    for arm in ARMS:
        net = RM.RaptorNet(TINY_ARCH)
        ck = {"arch": TINY_ARCH, "lab": RM.LABELS, "res": TINY_RES, "model": net.state_dict()}
        torch.save(ck, root / "weights" / R.RAPTOR_ARMS[arm][0])


def test_cli_writes_valid_submission_from_synthetic_inputs(tmp_path):
    ids = _fake_competition(tmp_path)
    _fake_checkpoints(tmp_path)
    out = tmp_path / "out"
    env = dict(os.environ, EDIV_SEARCH=str(tmp_path), EDIV_SMOKE="2", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "-m", "src.ediv.run_ensemble", "--mode", "test", "--arms", ",".join(ARMS),
           "--out", str(out), "--gpus", "1", "--workers", "0"]
    subprocess.run(cmd, cwd=REPO, env=env, check=True, timeout=600)

    sub = pd.read_csv(out / "submission.csv")
    assert list(sub.columns) == ["StudyInstanceUID"] + RM.LABELS
    assert sub.StudyInstanceUID.astype(str).tolist() == ids
    values = sub[RM.LABELS].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    report = json.load(open(out / "report_test.json"))
    assert all(f == {"no_data": 0, "error": 0} for f in report["flags"].values())
