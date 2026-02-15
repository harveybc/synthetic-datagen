"""Tests for app.data_processor."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from app.data_processor import (
    prices_to_returns,
    returns_to_prices,
    create_windows,
    load_csv,
    save_csv,
    prepare_training_data,
)


def test_returns_roundtrip():
    prices = np.array([1.0, 1.01, 1.005, 1.02, 0.99])
    rets = prices_to_returns(prices)
    recovered = returns_to_prices(rets, prices[0])
    np.testing.assert_allclose(recovered, prices, rtol=1e-12)


def test_create_windows():
    data = np.arange(10, dtype=float)
    w = create_windows(data, 3)
    assert w.shape == (8, 3)
    np.testing.assert_array_equal(w[0], [0, 1, 2])
    np.testing.assert_array_equal(w[-1], [7, 8, 9])


def test_create_windows_too_short():
    with pytest.raises(ValueError):
        create_windows(np.array([1.0, 2.0]), 5)


def test_csv_roundtrip():
    df = pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=5, freq="4h"),
        "typical_price": [1.3, 1.31, 1.29, 1.305, 1.32],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        save_csv(df, path)
        df2 = load_csv(path)
        assert len(df2) == 5
        assert "typical_price" in df2.columns
    finally:
        os.unlink(path)


def test_prepare_training_data():
    """Test full pipeline with a temp CSV."""
    df = pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=200, freq="4h"),
        "typical_price": 1.3 + np.cumsum(np.random.randn(200) * 0.001),
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        df.to_csv(path, index=False)
        windows, ip = prepare_training_data([path], window_size=10, use_returns=True)
        assert windows.shape[1] == 10
        assert windows.shape[0] == 200 - 1 - 10 + 1  # returns lose 1
        assert ip > 0
    finally:
        os.unlink(path)
