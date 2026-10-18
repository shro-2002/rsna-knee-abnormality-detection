"""Schema-validation tests for the submission builder."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.constants import TARGETS  # noqa: E402
from src.infer.submission import build_submission, validate_submission  # noqa: E402

IDS = ["a", "b", "c"]


def test_missing_study_gets_fallback_and_validates():
    preds = {"a": np.full(12, 0.3), "c": np.full(12, 0.7)}  # b missing
    df = build_submission(IDS, preds)
    validate_submission(df, IDS)  # must not raise
    assert (df.loc[df.StudyInstanceUID == "b", TARGETS].to_numpy() == 0.5).all()


def test_rejects_out_of_range():
    df = build_submission(IDS, {"a": np.full(12, 1.5)})
    with pytest.raises(AssertionError):
        validate_submission(df, IDS)


def test_rejects_wrong_row_set():
    df = build_submission(["a", "b"], {})
    with pytest.raises(AssertionError):
        validate_submission(df, IDS)


def test_rejects_non_finite():
    preds = {"a": np.full(12, np.nan)}
    df = build_submission(IDS, preds)
    with pytest.raises(AssertionError):
        validate_submission(df, IDS)


def test_column_order_exact():
    df = build_submission(IDS, {})
    assert list(df.columns)[0] == "StudyInstanceUID"
    assert list(df.columns)[1:] == TARGETS
