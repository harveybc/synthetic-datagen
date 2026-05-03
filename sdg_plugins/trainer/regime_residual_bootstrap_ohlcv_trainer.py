"""Regime-conditional residual block-bootstrap trainer (OHLCV).

Memorization-safer alternative to ``stationary_bootstrap_ohlcv_trainer``.

Idea
----
Plain stationary bootstrap resamples *raw* transformed primitives, so any
length-W window of synthetic returns is, with non-trivial probability, an
exact contiguous slice of the training set.  That is what fails the
Phase 4 §4.2 memorization gates (``duplicate_window_rate``,
``nn_overlap_rate``, ``copied_subseq_ratio``) for
``stationary_bootstrap_v1`` on ETHUSDT 4h.

This trainer reduces that risk by

1. Labeling each training row with a **volatility regime** (low / mid /
   high) using rolling ``|r_close|`` quantiles fit on the train window
   only.
2. Storing **regime-mean-removed residuals** of the five transformed
   primitives.  At generation time we resample blocks of residuals, add
   continuous Gaussian jitter, and re-base by the *current* synthetic
   regime mean.  Continuous jitter makes exact-window matches a
   measure-zero event and breaks the long copied-subseq runs.
3. Storing a regime transition matrix so the synthetic regime sequence
   is a Markov chain — preserving volatility clustering without copying
   any specific real-data slice.

All statistics are fit on the train split only; Project 3 mode rejects
any row at or after the heldout boundary (default ``2025-01-01``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer


_PRIMITIVES = ("r_close", "r_open", "d_high", "d_low", "v")


class RegimeResidualBootstrapOhlcvTrainer:
    plugin_params: Dict[str, Any] = {
        # primitive column names -- forwarded to OhlcvTransformer
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        # I/O
        "train_data": None,
        "save_model": "regime_residual_bootstrap_ohlcv.npz",
        "save_metadata": None,
        "data_format": "auto",  # csv | parquet | auto
        # bootstrap controls
        "block_length_mean": 24,         # ~4 days of 4h bars
        "n_regimes": 3,                  # quantile-bin volatility regimes
        "vol_window": 24,                # rolling |r_close| window for regime label
        "jitter_sigma": 0.5,             # additive noise = jitter_sigma * regime_std
        "seed": 42,
        # leakage / firewall
        "train_start": None,
        "train_end": None,
        "heldout_boundary": None,
        "project3_mode": False,
        "reject_if_input_crosses_heldout": True,
        "allow_non_research_mode": False,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params: Dict[str, Any] = dict(self.plugin_params)
        if config:
            for k, v in config.items():
                if k in self.plugin_params:
                    self.params[k] = v
        self.transformer: Optional[OhlcvTransformer] = None

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    # ------------------------------------------------------------------
    def _read_data(self, path: str) -> pd.DataFrame:
        try:
            from app.forbidden_paths import assert_path_allowed
            assert_path_allowed(path)
        except ImportError:
            pass
        fmt = self.params["data_format"]
        if fmt == "auto":
            fmt = "parquet" if str(path).lower().endswith(".parquet") else "csv"
        if fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _slice_train(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        if p["datetime_column"] not in df.columns:
            return df
        df = df.copy()
        df[p["datetime_column"]] = pd.to_datetime(df[p["datetime_column"]])
        df = df.sort_values(p["datetime_column"]).reset_index(drop=True)
        if p["train_start"] is not None:
            df = df[df[p["datetime_column"]] >= pd.Timestamp(p["train_start"])]
        if p["train_end"] is not None:
            df = df[df[p["datetime_column"]] < pd.Timestamp(p["train_end"])]
        return df.reset_index(drop=True)

    def _heldout_guard(self, df: pd.DataFrame) -> None:
        p = self.params
        if not p["reject_if_input_crosses_heldout"]:
            return
        if p["heldout_boundary"] is None:
            if p["project3_mode"] and not p["allow_non_research_mode"]:
                raise ValueError(
                    "project3_mode=True requires heldout_boundary; refusing to train."
                )
            return
        if p["datetime_column"] not in df.columns:
            return
        boundary = pd.Timestamp(p["heldout_boundary"])
        ts = pd.to_datetime(df[p["datetime_column"]])
        if (ts >= boundary).any():
            if p["allow_non_research_mode"] and not p["project3_mode"]:
                return
            raise ValueError(
                f"Training input contains rows on or after heldout_boundary={boundary}; "
                "refusing to fit the generator (Project 3 leakage guard)."
            )

    # ------------------------------------------------------------------
    def _label_regimes(self, z: pd.DataFrame) -> tuple:
        """Quantile-bin volatility into ``n_regimes`` regimes.

        The regime label of row ``i`` is the bucket of the rolling mean of
        ``|r_close|`` over ``vol_window`` bars (right-aligned, NaN warm-up
        filled with the first valid value).  Bucket edges are computed on
        the train split only -- there is no leakage of validation/heldout
        statistics.
        """
        p = self.params
        K = int(p["n_regimes"])
        w = max(2, int(p["vol_window"]))
        absr = np.abs(z["r_close"].to_numpy(dtype=np.float64))
        s = pd.Series(absr).rolling(w, min_periods=1).mean().to_numpy()
        # Quantile edges from the train rolling-vol distribution (no leakage):
        edges = np.quantile(s, np.linspace(0.0, 1.0, K + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        labels = np.clip(np.digitize(s, edges[1:-1], right=False), 0, K - 1).astype(np.int64)
        return labels, edges

    @staticmethod
    def _transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
        """Empirical row-stochastic regime transition matrix with Laplace smoothing."""
        T = np.ones((K, K), dtype=np.float64)  # +1 Laplace prior
        for a, b in zip(labels[:-1], labels[1:]):
            T[a, b] += 1.0
        T /= T.sum(axis=1, keepdims=True)
        return T

    @staticmethod
    def _stationary_distribution(T: np.ndarray) -> np.ndarray:
        """Left-eigenvector of ``T`` for eigenvalue 1 (regime-prior π)."""
        K = T.shape[0]
        evals, evecs = np.linalg.eig(T.T)
        idx = int(np.argmin(np.abs(evals - 1.0)))
        pi = np.real(evecs[:, idx])
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        return (pi / s) if s > 0 else np.full(K, 1.0 / K)

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        p = self.params
        if not p["train_data"]:
            raise ValueError("train_data is required")
        td = p["train_data"]
        if isinstance(td, (list, tuple)):
            if not td:
                raise ValueError("train_data list is empty")
            td = td[0]
        df_full = self._read_data(td)
        df_train = self._slice_train(df_full)
        self._heldout_guard(df_train)

        # Fit the train-only transformer; keep the SCALED primitives so that
        # later jitter is in the same scale as ``regime_std``.
        self.transformer = OhlcvTransformer(self.config)
        self.transformer.set_params(**{
            k: p[k] for k in (
                "datetime_column", "open_col", "high_col", "low_col",
                "close_col", "volume_col",
            )
        })
        self.transformer.fit(df_train)
        z_df = self.transformer.transform(df_train)
        Z = z_df[list(_PRIMITIVES)].to_numpy(dtype=np.float64)

        # Regime labeling on train only.
        labels, edges = self._label_regimes(z_df)
        K = int(p["n_regimes"])
        regime_means = np.zeros((K, len(_PRIMITIVES)), dtype=np.float64)
        regime_stds = np.ones((K, len(_PRIMITIVES)), dtype=np.float64)
        for k in range(K):
            mask = labels == k
            if mask.sum() >= 2:
                regime_means[k] = Z[mask].mean(axis=0)
                regime_stds[k] = Z[mask].std(axis=0, ddof=1)
                regime_stds[k] = np.where(regime_stds[k] > 1e-12, regime_stds[k], 1.0)
            else:
                regime_means[k] = Z.mean(axis=0)
                regime_stds[k] = np.where(Z.std(axis=0, ddof=1) > 1e-12,
                                          Z.std(axis=0, ddof=1), 1.0)
        residuals = Z - regime_means[labels]
        T = self._transition_matrix(labels, K)
        pi = self._stationary_distribution(T)

        meta = self._save(df_train, residuals, labels, regime_means, regime_stds, T, pi, edges)
        return meta

    # ------------------------------------------------------------------
    def _save(
        self,
        df_train: pd.DataFrame,
        residuals: np.ndarray,
        labels: np.ndarray,
        regime_means: np.ndarray,
        regime_stds: np.ndarray,
        T: np.ndarray,
        pi: np.ndarray,
        edges: np.ndarray,
    ) -> Dict[str, Any]:
        p = self.params
        os.makedirs(os.path.dirname(os.path.abspath(p["save_model"])) or ".", exist_ok=True)
        np.savez_compressed(
            p["save_model"],
            residuals=residuals,
            labels=labels,
            regime_means=regime_means,
            regime_stds=regime_stds,
            transition=T,
            stationary=pi,
            edges=edges,
            block_length_mean=np.int64(p["block_length_mean"]),
            n_regimes=np.int64(p["n_regimes"]),
            vol_window=np.int64(p["vol_window"]),
            jitter_sigma=np.float64(p["jitter_sigma"]),
            seed=np.int64(p["seed"]),
            first_close=np.float64(df_train[p["close_col"]].iloc[0]),
            n_train_rows=np.int64(len(df_train)),
        )
        side = os.path.splitext(p["save_model"])[0] + ".transformer.json"
        with open(side, "w") as f:
            json.dump(self.transformer.state_dict(), f)
        meta = {
            "model_file": p["save_model"],
            "transformer_file": side,
            "n_train_rows": int(len(df_train)),
            "n_regimes": int(p["n_regimes"]),
            "block_length_mean": int(p["block_length_mean"]),
            "vol_window": int(p["vol_window"]),
            "jitter_sigma": float(p["jitter_sigma"]),
            "seed": int(p["seed"]),
            "regime_counts": [int((labels == k).sum()) for k in range(int(p["n_regimes"]))],
        }
        if p["save_metadata"]:
            with open(p["save_metadata"], "w") as f:
                json.dump(meta, f, indent=2, default=str)
        return meta


__all__ = ["RegimeResidualBootstrapOhlcvTrainer"]
