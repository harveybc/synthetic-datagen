"""Generate a small deterministic OHLCV fixture under examples/data/."""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "financial_ohlcv_sample.csv")
OUT = os.path.abspath(OUT)

def main(n: int = 600, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.01, size=n)
    log_ret[100] = 0.05
    log_ret[300] = -0.04
    close = 100.0 * np.exp(np.cumsum(log_ret))
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.002, size=n))
    spread_h = np.abs(rng.normal(0, 0.003, size=n))
    spread_l = np.abs(rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * np.exp(spread_h)
    low = np.minimum(open_, close) * np.exp(-spread_l)
    volume = np.abs(rng.normal(1000, 200, size=n))
    ts = pd.date_range("2020-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "DATE_TIME": ts, "OPEN": open_, "HIGH": high, "LOW": low,
        "CLOSE": close, "VOLUME": volume,
    })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    return OUT

if __name__ == "__main__":
    print(main())
