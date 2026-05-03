"""Tests for the deterministic OHLCV multi-timeframe aggregator."""
import numpy as np
import pandas as pd

from sdg_plugins.aggregator.ohlcv_timeframe_aggregator import OhlcvTimeframeAggregator


def _hourly(n=24):
    ts = pd.date_range("2020-01-01", periods=n, freq="1h")
    o = np.arange(1, n + 1, dtype=float)
    h = o + 0.5
    l = o - 0.5
    c = o + 0.1
    v = np.full(n, 10.0)
    return pd.DataFrame({"DATE_TIME": ts, "OPEN": o, "HIGH": h, "LOW": l,
                         "CLOSE": c, "VOLUME": v})


def test_aggregate_to_4h():
    df = _hourly(24)
    agg = OhlcvTimeframeAggregator().aggregate(df, "4h")
    # 24 / 4 = 6 windows
    assert len(agg) == 6
    # First window 00:00-04:00 should have OPEN=1, HIGH=4.5, LOW=0.5, CLOSE=4.1
    first = agg.iloc[0]
    assert first["OPEN"] == 1.0
    assert first["HIGH"] == 4.5
    assert first["LOW"] == 0.5
    assert first["CLOSE"] == 4.1
    assert first["VOLUME"] == 40.0


def test_aggregate_by_count_equivalent():
    df = _hourly(24)
    agg = OhlcvTimeframeAggregator().aggregate(df, 4)
    assert len(agg) == 6
    assert agg.iloc[-1]["CLOSE"] == 24.1
