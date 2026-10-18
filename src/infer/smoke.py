"""CPU end-to-end inference smoke test.

Proves the plumbing: read required test study IDs -> for each, gather its series (from
*_series.csv) -> decode+preprocess whatever DICOMs exist under data_root -> run a prediction
head -> assemble + validate a schema-correct submission.csv. Studies with no local DICOMs get a
deterministic fallback row (never dropped). The head here is a STUB (constant 0.5): this file
proves schema + failure handling only; it is NOT a model and must never be uploaded.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS
from src.data.dicom import DecodeError, series_to_array
from src.infer.submission import FALLBACK_VALUE, build_submission, validate_submission


def stub_head(study_features) -> np.ndarray:
    """Placeholder prediction head. Replace with the trained model. Constant 0.5."""
    return np.full(len(TARGETS), 0.5, dtype=float)


def predict_study(study_id: str, series_rows: pd.DataFrame, data_root: str, test_split: str):
    """Returns (vector, status). status in {ok, no_local_data, decode_error}."""
    feats = []
    for _, r in series_rows.iterrows():
        sdir = os.path.join(data_root, test_split, study_id, r["SeriesInstanceUID"])
        if not os.path.isdir(sdir):
            continue
        try:
            feats.append(series_to_array(sdir, k_slices=8, size=224))
        except DecodeError:
            return None, "decode_error"
    if not feats:
        return None, "no_local_data"
    return stub_head(feats), "ok"


def run(data_root: str, out_path: str, test_csv: str, series_csv: str,
        test_split: str = "test_series") -> pd.DataFrame:
    test = pd.read_csv(test_csv)
    series = pd.read_csv(series_csv)
    required = test[ID_COL].tolist()
    preds, statuses = {}, {}
    for sid in required:
        vec, status = predict_study(sid, series[series[ID_COL] == sid], data_root, test_split)
        statuses[sid] = status
        if vec is not None:
            preds[sid] = vec
    df = build_submission(required, preds)
    validate_submission(df, required)
    df.to_csv(out_path, index=False)
    return df, statuses


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    root = os.path.join(here, "..", "..", "data", "raw")
    df, st = run(
        data_root=root,
        out_path=os.path.join(here, "..", "..", "data", "smoke_submission.csv"),
        test_csv=os.path.join(root, "test.csv"),
        series_csv=os.path.join(root, "test_series.csv"),
    )
    print("statuses:", st)
    print("submission.csv:\n", df.to_string())
    print("\nSCHEMA VALID: 3 rows x 13 cols, finite [0,1], no dupes/missing — plumbing OK.")
