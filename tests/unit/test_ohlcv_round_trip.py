"""Round-trip test: transformer -> reconstructor must reproduce real OHLCV."""
import numpy as np
import pandas as pd

from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer
from sdg_plugins.reconstructor.ohlcv_reconstructor import OhlcvReconstructor


def _toy_ohlcv(n=200, seed=1):
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.002, n))
    h_sp = np.abs(rng.normal(0, 0.003, n))
    l_sp = np.abs(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(h_sp)
    low = np.minimum(open_, close) * np.exp(-l_sp)
    vol = np.abs(rng.normal(1000, 100, n))
    return pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=n, freq="1h"),
        "OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close, "VOLUME": vol,
    })


def test_transformer_fit_is_train_only():
    df = _toy_ohlcv(300, seed=7)
    train = df.iloc[:200]
    val = df.iloc[200:]
    t = OhlcvTransformer().set_params(scale=True)  # scale=True
    t = OhlcvTransformer()
    t.set_params(scale=True)
    t.fit(train)
    fit_meta = t._fit_meta.copy()
    # Calling .transform() on validation must NOT change the fitted scaler.
    _ = t.transform(val)
    assert t._fit_meta == fit_meta


def test_round_trip_reconstructs_real_ohlcv():
    df = _toy_ohlcv(150, seed=42)
    t = OhlcvTransformer()
    t.set_params(scale=False)  # disable scaling for exact round-trip
    t.fit(df)
    z = t.transform(df)
    recon = OhlcvReconstructor().reconstruct(
        z, initial_close=float(df["CLOSE"].iloc[0]),
        transformer=t,
    )
    # The transformer drops the first row; align with df.iloc[1:].
    real = df.iloc[1:].reset_index(drop=True)
    for col in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"):
        np.testing.assert_allclose(
            recon[col].to_numpy(), real[col].to_numpy(),
            rtol=1e-9, atol=1e-7,
            err_msg=f"round-trip mismatch in {col}",
        )


def test_reconstructor_enforces_validity():
    df = _toy_ohlcv(100, seed=2)
    t = OhlcvTransformer(); t.set_params(scale=False); t.fit(df)
    z = t.transform(df).copy()
    # Even if d_high/d_low were corrupted to negatives, recon must clamp.
    z["d_high"] = -0.5
    z["d_low"] = -0.7
    out = OhlcvReconstructor().reconstruct(z, initial_close=100.0, transformer=t)
    assert (out["HIGH"] >= np.maximum(out["OPEN"], out["CLOSE"]) - 1e-12).all()
    assert (out["LOW"] <= np.minimum(out["OPEN"], out["CLOSE"]) + 1e-12).all()
    assert (out[["OPEN", "HIGH", "LOW", "CLOSE"]] > 0).all().all()
    assert (out["VOLUME"] >= 0).all()
