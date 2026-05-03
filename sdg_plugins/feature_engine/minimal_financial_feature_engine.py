"""Minimal causal financial feature engine.

Recomputes a small, well-defined set of deterministic features from raw
synthetic OHLCV.  This is intentionally tiny: the purpose is to provide
a *test-able* recompute path so that the spec rule "deterministic
indicators must be recomputed, never independently generated" can be
enforced and verified end-to-end.

For production use, configure an external Project 3 feature engine
through the ``feature_engine_plugin`` config key instead.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MinimalFinancialFeatureEngine:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "rolling_window": 14,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params: Dict[str, Any] = dict(self.plugin_params)
        if config:
            for k, v in config.items():
                if k in self.plugin_params:
                    self.params[k] = v

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        w = int(p["rolling_window"])
        o = ohlcv[p["open_col"]].astype(np.float64)
        h = ohlcv[p["high_col"]].astype(np.float64)
        l = ohlcv[p["low_col"]].astype(np.float64)
        c = ohlcv[p["close_col"]].astype(np.float64)
        v = ohlcv[p["volume_col"]].astype(np.float64)

        typical_price = (h + l + c) / 3.0
        ret = c.pct_change()
        log_ret = np.log(c / c.shift(1))
        roll_mean = c.rolling(w, min_periods=1).mean()
        roll_std = c.rolling(w, min_periods=1).std().fillna(0.0)
        realized_var = log_ret.rolling(w, min_periods=1).var().fillna(0.0)
        # Wilder ATR-style true range.
        prev_close = c.shift(1).fillna(c)
        tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(w, min_periods=1).mean()
        vol_mean = v.rolling(w, min_periods=1).mean()
        vol_ratio = (v / vol_mean.replace(0.0, np.nan)).fillna(1.0)

        out = ohlcv.copy()
        out["TYPICAL_PRICE"] = typical_price
        out["RETURN"] = ret.fillna(0.0)
        out["LOG_RETURN"] = log_ret.fillna(0.0)
        out[f"ROLL_MEAN_{w}"] = roll_mean
        out[f"ROLL_STD_{w}"] = roll_std
        out[f"REALIZED_VAR_{w}"] = realized_var
        out[f"ATR_{w}"] = atr
        out[f"VOL_RATIO_{w}"] = vol_ratio
        return out


__all__ = ["MinimalFinancialFeatureEngine"]
