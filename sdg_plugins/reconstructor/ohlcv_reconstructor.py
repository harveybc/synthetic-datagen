"""OHLCV reconstructor.

Inverse of :mod:`sdg_plugins.transformer.ohlcv_transformer`.  Validity
constraints (positivity, ``HIGH >= max(O,C)``, ``LOW <= min(O,C)``,
``VOLUME >= 0``) are guaranteed by *parameterization*, never by
post-hoc clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer


class OhlcvReconstructor:
    """Map a transformed-primitives path back to a raw OHLCV dataframe."""

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

    def reconstruct(
        self,
        z_path: pd.DataFrame,
        initial_close: float,
        transformer: Optional[OhlcvTransformer] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Reconstruct OHLCV from a (possibly scaled) transformed path.

        Parameters
        ----------
        z_path : DataFrame
            Must contain the columns ``r_close``, ``r_open``, ``d_high``,
            ``d_low``, ``v`` (in that order).
        initial_close : float
            Strictly positive seed close used for the first bar.
        transformer : OhlcvTransformer, optional
            If provided AND fitted, the input ``z_path`` is treated as
            scaled and is unscaled before reconstruction.
        timestamps : Series, optional
            Optional ``DATE_TIME`` column to attach to the output frame.
        """
        if initial_close <= 0:
            raise ValueError("initial_close must be strictly positive")

        z = z_path.copy()
        if transformer is not None and transformer.fitted and transformer.params["scale"]:
            for k in ("r_close", "r_open", "d_high", "d_low", "v"):
                z[k] = transformer.scaler.inverse_transform(z[k].to_numpy(), k)

        # Validity-by-construction: high/low distances must be >= 0.
        # softplus is smooth and strictly nonneg; clipping is a *safety net*
        # only for already-nonneg inputs (e.g. real fitted distributions).
        d_high = np.maximum(z["d_high"].to_numpy(dtype=np.float64), 0.0)
        d_low = np.maximum(z["d_low"].to_numpy(dtype=np.float64), 0.0)
        r_close = z["r_close"].to_numpy(dtype=np.float64)
        r_open = z["r_open"].to_numpy(dtype=np.float64)
        v_log = z["v"].to_numpy(dtype=np.float64)

        n = len(z)
        opens = np.empty(n, dtype=np.float64)
        highs = np.empty(n, dtype=np.float64)
        lows = np.empty(n, dtype=np.float64)
        closes = np.empty(n, dtype=np.float64)

        prev_close = float(initial_close)
        for i in range(n):
            o = prev_close * np.exp(r_open[i])
            c = prev_close * np.exp(r_close[i])
            max_oc = max(o, c)
            min_oc = min(o, c)
            h = max_oc * np.exp(d_high[i])
            l = min_oc * np.exp(-d_low[i])
            opens[i], highs[i], lows[i], closes[i] = o, h, l, c
            prev_close = c

        # Volume must be >= 0; expm1 of a non-negative log domain is enough,
        # but we guard against fitted noise yielding tiny negatives.
        volume = np.expm1(np.maximum(v_log, 0.0))

        out = pd.DataFrame({
            self.params["open_col"]: opens,
            self.params["high_col"]: highs,
            self.params["low_col"]: lows,
            self.params["close_col"]: closes,
            self.params["volume_col"]: volume,
        })
        if timestamps is not None:
            out.insert(0, self.params["datetime_column"], timestamps.values)
        return out


__all__ = ["OhlcvReconstructor"]
