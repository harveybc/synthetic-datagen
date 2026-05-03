"""Deterministic OHLCV multi-timeframe aggregator.

Aggregation rules:

* ``OPEN``   = first open in the window
* ``HIGH``   = max high
* ``LOW``    = min low
* ``CLOSE``  = last close
* ``VOLUME`` = sum volume
* ``DATE_TIME`` = window-start timestamp

The aggregator is timestamp-aware and accepts either a pandas Timedelta
string (``"15min"``, ``"1h"``, ``"4h"``) or an integer bar-count.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd


class OhlcvTimeframeAggregator:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params: Dict[str, Any] = dict(self.plugin_params)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    def aggregate(
        self,
        df: pd.DataFrame,
        rule: Union[str, int],
        origin: str = "start_day",
    ) -> pd.DataFrame:
        """Aggregate a base-resolution OHLCV frame to a coarser timeframe.

        ``rule`` may be a pandas resample alias (``"15min"``, ``"1h"``,
        ``"4h"``) or an integer count of base bars to merge.
        """
        p = self.params
        if isinstance(rule, int):
            return self._aggregate_by_count(df, rule)
        df = df.copy()
        df[p["datetime_column"]] = pd.to_datetime(df[p["datetime_column"]])
        df = df.set_index(p["datetime_column"]).sort_index()
        agg = df.resample(rule, origin=origin, label="left", closed="left").agg({
            p["open_col"]: "first",
            p["high_col"]: "max",
            p["low_col"]: "min",
            p["close_col"]: "last",
            p["volume_col"]: "sum",
        }).dropna(subset=[p["open_col"], p["close_col"]])
        agg = agg.reset_index().rename(columns={"index": p["datetime_column"]})
        return agg

    def _aggregate_by_count(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 1:
            return df.copy()
        p = self.params
        n = len(df) // k
        rows = []
        for i in range(n):
            chunk = df.iloc[i * k : (i + 1) * k]
            rows.append({
                p["datetime_column"]: chunk[p["datetime_column"]].iloc[0],
                p["open_col"]: chunk[p["open_col"]].iloc[0],
                p["high_col"]: chunk[p["high_col"]].max(),
                p["low_col"]: chunk[p["low_col"]].min(),
                p["close_col"]: chunk[p["close_col"]].iloc[-1],
                p["volume_col"]: chunk[p["volume_col"]].sum(),
            })
        return pd.DataFrame(rows)


__all__ = ["OhlcvTimeframeAggregator"]
