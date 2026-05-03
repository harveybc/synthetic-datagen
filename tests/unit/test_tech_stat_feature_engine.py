"""Smoke test for TechStatFeatureEngine — every column expected by the
agent-multi feature_window_preprocessor MUST be present in the output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdg_plugins.feature_engine.tech_stat_feature_engine import TechStatFeatureEngine


EXPECTED = [
    "typical_price", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
    "return_1", "log_return_1", "return_5", "log_return_5",
    "return_10", "log_return_10", "return_20", "log_return_20",
    "return_60", "log_return_60",
    "sma_10", "ema_10", "close_sma_ratio_10",
    "sma_20", "ema_20", "close_sma_ratio_20",
    "sma_50", "ema_50", "close_sma_ratio_50",
    "sma_100", "ema_100", "close_sma_ratio_100",
    "sma_200", "ema_200", "close_sma_ratio_200",
    "macd", "macd_signal", "macd_hist",
    "rsi_7", "rsi_14", "rsi_21",
    "stoch_k", "stoch_d", "williams_r_14", "cci_14",
    "roc_10", "roc_20", "roc_60", "mom_10", "mom_20",
    "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_width",
    "atr_14", "natr_14",
    "hist_vol_10", "hist_vol_20", "hist_vol_60",
    "ema_cross_10_50", "ema_cross_20_100",
    "trend_slope_50", "trend_strength_50",
    "obv", "obv_delta_20",
    "volume_sma_10", "volume_sma_20", "volume_ratio_20", "vwap_60", "mfi_14",
    "statistical__log_return_1",
    "roll_mean_ret_20", "roll_std_ret_20", "roll_skew_ret_20", "roll_kurt_ret_20",
    "roll_mean_ret_60", "roll_std_ret_60", "roll_skew_ret_60", "roll_kurt_ret_60",
    "roll_mean_ret_252", "roll_std_ret_252", "roll_skew_ret_252", "roll_kurt_ret_252",
    "realized_var_12", "realized_var_48",
    "autocorr_lag1_100", "autocorr_lag5_100", "sqret_autocorr_lag1_100",
    "vol_regime_high", "vol_regime_low", "hurst_proxy_200", "zscore_close_100",
]


def _gbm_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.005, size=n)
    c = 100.0 * np.exp(np.cumsum(r))
    o = np.concatenate([[c[0]], c[:-1]])
    spread = np.abs(rng.normal(0.0, 0.5, size=n))
    h = np.maximum(o, c) + spread
    l = np.minimum(o, c) - spread
    v = rng.uniform(100, 1000, size=n)
    return pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=n, freq="4h"),
        "OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "VOLUME": v,
    })


def test_tech_stat_engine_emits_all_required_columns():
    fe = TechStatFeatureEngine()
    out = fe.compute(_gbm_ohlcv())
    for col in EXPECTED:
        assert col in out.columns, f"missing column: {col}"


def test_tech_stat_engine_has_no_nan_inf():
    fe = TechStatFeatureEngine()
    out = fe.compute(_gbm_ohlcv())
    arr = out.drop(columns=["DATE_TIME"]).to_numpy(dtype=np.float64)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_tech_stat_engine_is_deterministic():
    fe = TechStatFeatureEngine()
    df = _gbm_ohlcv(seed=7)
    a = fe.compute(df)
    b = fe.compute(df)
    pd.testing.assert_frame_equal(a, b)
