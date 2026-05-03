"""Algebraic OHLCV validity evaluator.

Checks that a synthetic dataframe satisfies all hard structural rules:

* OHLC values are strictly positive
* VOLUME is non-negative
* HIGH >= max(OPEN, CLOSE)
* LOW  <= min(OPEN, CLOSE)
* recomputed TYPICAL_PRICE matches (HIGH+LOW+CLOSE)/3 within tolerance
* no NaN/Inf in any primitive column

Returns a dict with both per-rule violation counts and a single boolean
``valid`` flag.  Callers are expected to fail the run when ``valid`` is
``False`` (the spec disallows silent post-hoc clipping).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class OhlcvAlgebraicEvaluator:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "synthetic_data": None,
        "metrics_file": None,
        "typical_price_tol": 1e-6,
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

    def _read(self, path: str) -> pd.DataFrame:
        if str(path).lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def evaluate(self) -> Dict[str, Any]:
        p = self.params
        if not p["synthetic_data"]:
            raise ValueError("synthetic_data is required")
        df = self._read(p["synthetic_data"])
        return self.evaluate_df(df)

    def evaluate_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        p = self.params
        o = df[p["open_col"]].to_numpy(dtype=np.float64)
        h = df[p["high_col"]].to_numpy(dtype=np.float64)
        l = df[p["low_col"]].to_numpy(dtype=np.float64)
        c = df[p["close_col"]].to_numpy(dtype=np.float64)
        v = df[p["volume_col"]].to_numpy(dtype=np.float64)

        nan_violations = int(
            np.isnan(o).sum() + np.isnan(h).sum() + np.isnan(l).sum()
            + np.isnan(c).sum() + np.isnan(v).sum()
        )
        inf_violations = int(
            np.isinf(o).sum() + np.isinf(h).sum() + np.isinf(l).sum()
            + np.isinf(c).sum() + np.isinf(v).sum()
        )
        positivity = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
        volume_neg = int((v < 0).sum())
        high_violations = int((h < np.maximum(o, c)).sum())
        low_violations = int((l > np.minimum(o, c)).sum())

        tp_violations = 0
        if "TYPICAL_PRICE" in df.columns:
            recomputed = (h + l + c) / 3.0
            tp_violations = int(
                (np.abs(df["TYPICAL_PRICE"].to_numpy(dtype=np.float64) - recomputed)
                 > p["typical_price_tol"]).sum()
            )

        total = (nan_violations + inf_violations + positivity + volume_neg
                 + high_violations + low_violations + tp_violations)

        report = {
            "valid": total == 0,
            "n_rows": int(len(df)),
            "violations": {
                "nan": nan_violations,
                "inf": inf_violations,
                "non_positive_ohlc": positivity,
                "negative_volume": volume_neg,
                "high_below_max_oc": high_violations,
                "low_above_min_oc": low_violations,
                "typical_price_mismatch": tp_violations,
            },
        }
        if p["metrics_file"]:
            os.makedirs(os.path.dirname(os.path.abspath(p["metrics_file"])) or ".", exist_ok=True)
            import json as _json
            with open(p["metrics_file"], "w") as f:
                _json.dump(report, f, indent=2)
        return report


__all__ = ["OhlcvAlgebraicEvaluator"]
