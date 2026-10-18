"""Submission assembly + strict validation against the official schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import ID_COL, SUBMISSION_COLUMNS, TARGETS

FALLBACK_VALUE = 0.5  # deterministic row for a failed/empty study (never silently dropped)


def build_submission(study_ids: list[str], preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """study_ids in required order; preds maps study_id -> length-12 vector (missing -> fallback)."""
    rows = []
    for sid in study_ids:
        vec = preds.get(sid)
        if vec is None:
            vec = np.full(len(TARGETS), FALLBACK_VALUE, dtype=float)
        rows.append([sid, *[float(x) for x in vec]])
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def validate_submission(df: pd.DataFrame, required_ids: list[str]) -> None:
    """Raise AssertionError on any schema violation."""
    assert list(df.columns) == SUBMISSION_COLUMNS, f"bad columns: {list(df.columns)}"
    ids = df[ID_COL].tolist()
    assert len(ids) == len(set(ids)), "duplicate StudyInstanceUID rows"
    assert set(ids) == set(required_ids), "row set != required test studies"
    assert len(ids) == len(required_ids), f"row count {len(ids)} != {len(required_ids)}"
    vals = df[TARGETS].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "non-finite predictions"
    assert (vals >= 0).all() and (vals <= 1).all(), "predictions outside [0,1]"
