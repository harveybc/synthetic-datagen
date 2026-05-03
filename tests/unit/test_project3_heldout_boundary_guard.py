"""Project 3 heldout-boundary guard tests."""
import numpy as np
import pandas as pd
import pytest

from app.heldout_guard import assert_no_heldout_violation


def _frame(start="2018-01-01", n=200, freq="1h"):
    ts = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"DATE_TIME": ts, "CLOSE": np.linspace(1.0, 2.0, n)})


def test_no_violation_passes():
    df = _frame("2018-01-01", 100)
    assert_no_heldout_violation(df, {"heldout_boundary": "2025-01-01"})


def test_violation_raises():
    df = _frame("2024-12-31 00:00:00", 100)
    with pytest.raises(ValueError):
        assert_no_heldout_violation(df, {"heldout_boundary": "2025-01-01"})


def test_project3_requires_boundary():
    df = _frame()
    with pytest.raises(ValueError):
        assert_no_heldout_violation(df, {"project3_mode": True})


def test_project3_rejects_escape_hatch():
    df = _frame("2024-12-31", 100)
    with pytest.raises(ValueError):
        assert_no_heldout_violation(df, {
            "project3_mode": True,
            "heldout_boundary": "2025-01-01",
            "allow_non_research_mode": True,  # NEVER for Project 3
        })


def test_non_research_escape_hatch_allows_when_not_project3():
    df = _frame("2024-12-31", 100)
    assert_no_heldout_violation(df, {
        "project3_mode": False,
        "heldout_boundary": "2025-01-01",
        "allow_non_research_mode": True,
    })
