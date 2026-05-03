"""Train-only OHLCV transformer.

Maps raw OHLCV bars onto five transformed primitive variables:

* ``r_close = log(CLOSE_t / CLOSE_{t-1})``
* ``r_open  = log(OPEN_t  / CLOSE_{t-1})``
* ``d_high  = log(HIGH_t  / max(OPEN_t, CLOSE_t))``        (>= 0)
* ``d_low   = log(min(OPEN_t, CLOSE_t) / LOW_t)``           (>= 0)
* ``v       = log1p(VOLUME_t)`` or volume-residual

Robust per-column scaling (median / MAD) is fit on the training split
ONLY.  Validation/heldout windows are never used to fit any statistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


_PRIMITIVES = ("r_close", "r_open", "d_high", "d_low", "v")


@dataclass
class FittedScaler:
    median: Dict[str, float] = field(default_factory=dict)
    mad: Dict[str, float] = field(default_factory=dict)

    def transform(self, x: np.ndarray, key: str) -> np.ndarray:
        m = self.median.get(key, 0.0)
        s = self.mad.get(key, 1.0) or 1.0
        return (x - m) / s

    def inverse_transform(self, z: np.ndarray, key: str) -> np.ndarray:
        m = self.median.get(key, 0.0)
        s = self.mad.get(key, 1.0) or 1.0
        return z * s + m


class OhlcvTransformer:
    """Train-only OHLCV→transformed-primitives transformer.

    Backward compatible with the predictor plugin contract: declares
    ``plugin_params`` and exposes ``set_params(**kw)``.
    """

    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "volume_transform": "log1p",  # log1p | log1p_residual
        "scale": True,
        "min_range_eps": 1e-9,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params: Dict[str, Any] = dict(self.plugin_params)
        self.scaler: Optional[FittedScaler] = None
        self.fitted: bool = False
        self._fit_meta: Dict[str, Any] = {}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    # ------------------------------------------------------------------
    # Forward (real -> transformed primitives)
    # ------------------------------------------------------------------
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the five transformed primitives for every row of ``df``.

        The first row is dropped because returns require a previous close.
        """
        p = self.params
        eps = p["min_range_eps"]
        o = df[p["open_col"]].to_numpy(dtype=np.float64)
        h = df[p["high_col"]].to_numpy(dtype=np.float64)
        l = df[p["low_col"]].to_numpy(dtype=np.float64)
        c = df[p["close_col"]].to_numpy(dtype=np.float64)
        v = df[p["volume_col"]].to_numpy(dtype=np.float64)

        if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
            raise ValueError("OhlcvTransformer requires strictly positive OHLC values")
        if (v < 0).any():
            raise ValueError("OhlcvTransformer requires non-negative VOLUME")

        prev_close = np.roll(c, 1)
        # Sanitize the (invalid) first prev_close.
        prev_close[0] = c[0]

        max_oc = np.maximum(o, c)
        min_oc = np.minimum(o, c)
        # Numerical floor protects log against tiny negative noise.
        d_high = np.log(np.maximum(h, max_oc + eps) / max_oc)
        d_low = np.log(min_oc / np.minimum(l, min_oc - eps))

        out = pd.DataFrame({
            "r_close": np.log(c / prev_close),
            "r_open": np.log(o / prev_close),
            "d_high": d_high,
            "d_low": d_low,
            "v": np.log1p(v),
        })
        # Drop the first row whose prev_close is fabricated.
        out = out.iloc[1:].reset_index(drop=True)
        # Carry the timestamps for traceability.
        if p["datetime_column"] in df.columns:
            out[p["datetime_column"]] = (
                df[p["datetime_column"]].iloc[1:].reset_index(drop=True)
            )
        return out

    def fit(self, df_train: pd.DataFrame) -> "OhlcvTransformer":
        """Fit median/MAD on TRAIN-ONLY transformed primitives."""
        z = self.transform_dataframe(df_train)
        scaler = FittedScaler()
        for k in _PRIMITIVES:
            arr = z[k].to_numpy()
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826  # ~ std
            scaler.median[k] = med
            scaler.mad[k] = mad if mad > 0 else 1.0
        self.scaler = scaler
        self.fitted = True
        self._fit_meta = {
            "n_train_rows": int(len(df_train)),
            "n_train_transformed": int(len(z)),
            "first_close": float(df_train[self.params["close_col"]].iloc[0]),
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward transform (optionally scaled with the fitted scaler)."""
        z = self.transform_dataframe(df)
        if self.params["scale"]:
            if not self.fitted:
                raise RuntimeError("OhlcvTransformer.transform() called before fit()")
            for k in _PRIMITIVES:
                z[k] = self.scaler.transform(z[k].to_numpy(), k)
        return z

    # ------------------------------------------------------------------
    # State (de)serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "fitted": self.fitted,
            "median": dict(self.scaler.median) if self.scaler else {},
            "mad": dict(self.scaler.mad) if self.scaler else {},
            "fit_meta": dict(self._fit_meta),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "OhlcvTransformer":
        self.params.update(state.get("params", {}))
        self.fitted = bool(state.get("fitted", False))
        if self.fitted:
            sc = FittedScaler()
            sc.median = dict(state.get("median", {}))
            sc.mad = dict(state.get("mad", {}))
            self.scaler = sc
        self._fit_meta = dict(state.get("fit_meta", {}))
        return self


__all__ = ["OhlcvTransformer", "FittedScaler"]
