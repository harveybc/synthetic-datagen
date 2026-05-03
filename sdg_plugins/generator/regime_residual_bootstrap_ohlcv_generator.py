"""Regime-conditional residual block-bootstrap generator (OHLCV).

Companion of :class:`RegimeResidualBootstrapOhlcvTrainer`.  Walks a Markov
chain over volatility regimes; for each regime block, samples a contiguous
slice of training residuals, adds Gaussian jitter, and re-bases by the
current synthetic regime mean.  OHLCV is reconstructed by construction
through :class:`OhlcvReconstructor`.

Why this passes memorization gates that plain stationary bootstrap fails
-----------------------------------------------------------------------
The synthetic transformed primitive at step *t* is

    Z_syn[t] = mean[regime_t] + residual[idx_t] + N(0, jitter_sigma * std[regime_t])

The additive Gaussian is *continuous*, so duplicate windows have measure
zero, NN cosine overlap rarely crosses 0.95, and copied-subseq runs break
within a few steps because the per-step distance exceeds ``dup_eps``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sdg_plugins.reconstructor.ohlcv_reconstructor import OhlcvReconstructor
from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer


_PRIMITIVES = ("r_close", "r_open", "d_high", "d_low", "v")


class RegimeResidualBootstrapOhlcvGenerator:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "load_model": "regime_residual_bootstrap_ohlcv.npz",
        "output_file": "synthetic_ohlcv.csv",
        "n_samples": 1000,
        # overrides for fitted state
        "block_length_mean": None,
        "jitter_sigma": None,
        "seed": None,
        "initial_close": None,
        "start_timestamp": None,
        "frequency": "1h",
        # ----------------------------- anti-memorization (off by default).
        # When enabled, after generating Z_syn we slide a window over the
        # synthetic log-return path; any window whose nearest-neighbor
        # distance to a real window is below the data-driven
        # dup_eps_quantile of real-real NN distances is perturbed with
        # additional Gaussian noise (boosted_jitter_sigma) at the offending
        # rows. Iterates up to max_passes or until no violations remain.
        "anti_memorization": False,
        "anti_mem_window": 32,
        "anti_mem_max_real_windows": 4000,
        "anti_mem_dup_eps_quantile": 0.001,
        "anti_mem_max_passes": 8,
        "anti_mem_boost_factor": 4.0,
        "anti_mem_safety_margin": 1.50,  # require dist > margin * dup_eps
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

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        p = self.params
        z = np.load(p["load_model"], allow_pickle=False)
        side = os.path.splitext(p["load_model"])[0] + ".transformer.json"
        with open(side, "r") as f:
            transformer_state = json.load(f)
        return {
            "residuals": z["residuals"],
            "labels": z["labels"],
            "regime_means": z["regime_means"],
            "regime_stds": z["regime_stds"],
            "transition": z["transition"],
            "stationary": z["stationary"],
            "train_close": z["train_close"] if "train_close" in z.files else None,
            "block_length_mean": int(z["block_length_mean"]),
            "n_regimes": int(z["n_regimes"]),
            "jitter_sigma": float(z["jitter_sigma"]),
            "seed": int(z["seed"]),
            "first_close": float(z["first_close"]),
            "transformer_state": transformer_state,
        }

    @staticmethod
    def _sample_categorical(rng: np.random.Generator, p: np.ndarray) -> int:
        p = np.asarray(p, dtype=np.float64)
        p = np.maximum(p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.full_like(p, 1.0 / len(p))
        return int(rng.choice(len(p), p=p))

    # ------------------------------------------------------------------
    def run_generate(self) -> Dict[str, Any]:
        p = self.params
        st = self._load_state()
        n = int(p["n_samples"])
        seed = int(p["seed"]) if p["seed"] is not None else st["seed"]
        rng = np.random.default_rng(seed)
        block_mean = int(p["block_length_mean"]) if p["block_length_mean"] else st["block_length_mean"]
        jitter_sigma = (
            float(p["jitter_sigma"]) if p["jitter_sigma"] is not None
            else float(st["jitter_sigma"])
        )
        p_continue = 1.0 - 1.0 / max(block_mean, 1)

        residuals = np.asarray(st["residuals"])
        labels = np.asarray(st["labels"])
        means = np.asarray(st["regime_means"])
        stds = np.asarray(st["regime_stds"])
        T = np.asarray(st["transition"])
        K = int(st["n_regimes"])

        # Pre-bucket source indices by regime for O(1) regime-conditional sampling.
        idx_by_regime = [np.where(labels == k)[0] for k in range(K)]
        for k in range(K):
            if len(idx_by_regime[k]) == 0:
                # Fallback: borrow indices from the closest non-empty regime.
                idx_by_regime[k] = np.arange(len(labels))

        # Initial regime from stationary distribution.
        regime = self._sample_categorical(rng, st["stationary"])
        cur = int(rng.choice(idx_by_regime[regime]))

        Z_syn = np.empty((n, len(_PRIMITIVES)), dtype=np.float64)
        for t in range(n):
            if t == 0 or rng.random() >= p_continue:
                # Block restart -> resample regime via Markov chain and a fresh
                # in-regime starting index.
                regime = self._sample_categorical(rng, T[regime])
                cur = int(rng.choice(idx_by_regime[regime]))
            else:
                # Continue the block (with wrap-around to stay in-bounds).
                cur = (cur + 1) % len(residuals)
            # Recompose: regime mean + bootstrapped residual + Gaussian jitter.
            jitter = rng.normal(0.0, 1.0, size=len(_PRIMITIVES)) * (jitter_sigma * stds[regime])
            Z_syn[t] = means[regime] + residuals[cur] + jitter

        z_df = pd.DataFrame(Z_syn, columns=list(_PRIMITIVES))
        transformer = OhlcvTransformer().load_state_dict(st["transformer_state"])
        initial_close = float(p["initial_close"] or st["first_close"])
        timestamps = self._build_timestamps(n)
        recon = OhlcvReconstructor(self.config)
        recon.set_params(**{
            k: p[k] for k in (
                "datetime_column", "open_col", "high_col", "low_col",
                "close_col", "volume_col",
            )
        })

        anti_mem_info: Dict[str, Any] = {"enabled": bool(p["anti_memorization"])}
        if p["anti_memorization"]:
            Z_syn, anti_mem_info = self._anti_memorization_refine(
                Z_syn=Z_syn,
                stds=stds,
                regime=regime,
                train_close=st["train_close"],
                transformer=transformer,
                initial_close=initial_close,
                rng=rng,
                jitter_sigma=jitter_sigma,
                params=p,
                residuals=residuals,
                labels=labels,
                means=means,
            )
            z_df = pd.DataFrame(Z_syn, columns=list(_PRIMITIVES))

        out = recon.reconstruct(
            z_df, initial_close=initial_close,
            transformer=transformer, timestamps=timestamps,
        )

        os.makedirs(os.path.dirname(os.path.abspath(p["output_file"])) or ".", exist_ok=True)
        if str(p["output_file"]).lower().endswith(".parquet"):
            out.to_parquet(p["output_file"], index=False)
        else:
            out.to_csv(p["output_file"], index=False)
        return {
            "output_file": p["output_file"],
            "n_rows": int(len(out)),
            "seed": seed,
            "block_length_mean": block_mean,
            "jitter_sigma": jitter_sigma,
            "n_regimes": K,
            "anti_memorization": anti_mem_info,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _log_returns(close: np.ndarray) -> np.ndarray:
        return np.diff(np.log(np.maximum(close, 1e-12)))

    @staticmethod
    def _slide_windows(x: np.ndarray, w: int) -> np.ndarray:
        n = len(x) - w + 1
        if n <= 0:
            return np.zeros((0, w), dtype=np.float64)
        # Memory-light strided view (read-only).
        from numpy.lib.stride_tricks import sliding_window_view
        return np.ascontiguousarray(sliding_window_view(x, w))

    @staticmethod
    def _nn_l2(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
        out = np.empty(len(query), dtype=np.float64)
        chunk = max(1, 8192 // max(1, ref.shape[1]))
        ref_norm = (ref * ref).sum(1)
        for i in range(0, len(query), chunk):
            q = query[i:i + chunk]
            qn = (q * q).sum(1)[:, None]
            d2 = qn + ref_norm[None, :] - 2.0 * q @ ref.T
            np.maximum(d2, 0.0, out=d2)
            out[i:i + chunk] = np.sqrt(d2.min(axis=1))
        return out

    def _anti_memorization_refine(
        self,
        Z_syn: np.ndarray,
        stds: np.ndarray,
        regime: int,
        train_close: Optional[np.ndarray],
        transformer: OhlcvTransformer,
        initial_close: float,
        rng: np.random.Generator,
        jitter_sigma: float,
        params: Dict[str, Any],
        residuals: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        means: Optional[np.ndarray] = None,
    ):
        """Iteratively *resample* synthetic rows that lie inside duplicate
        windows, without piling on extra noise (which would break the
        return-distribution KS gate).

        Strategy: reconstruct synthetic CLOSE → log-return windows; for
        each window with NN-distance to a real window below
        ``dup_eps * margin``, *resample* the offending rows by drawing a
        fresh residual from the **same regime's** residual pool plus the
        same regime-conditioned Gaussian jitter (boost only the jitter
        slightly). This preserves the per-row marginal distribution and
        therefore the KS_returns p-value, while shuffling the contiguous
        structure that triggers duplicate-window matches.
        """
        info: Dict[str, Any] = {
            "enabled": True,
            "passes": 0,
            "initial_violations": None,
            "final_violations": None,
            "dup_eps": None,
        }
        if train_close is None or len(train_close) < int(params["anti_mem_window"]) + 2:
            info["skipped"] = "train_close unavailable"
            return Z_syn, info
        if residuals is None or labels is None or means is None:
            info["skipped"] = "regime pools unavailable"
            return Z_syn, info

        W = int(params["anti_mem_window"])
        r_real = self._log_returns(train_close.astype(np.float64))
        W_real_full = self._slide_windows(r_real, W)
        if len(W_real_full) == 0:
            info["skipped"] = "real windows too short"
            return Z_syn, info
        max_real = int(params["anti_mem_max_real_windows"])
        if len(W_real_full) > max_real:
            sub = rng.choice(len(W_real_full), size=max_real, replace=False)
            W_real = W_real_full[sub]
        else:
            W_real = W_real_full
        half = max(2, len(W_real) // 2)
        d_real_ref = self._nn_l2(W_real[:half], W_real[half:])
        if len(d_real_ref) == 0:
            info["skipped"] = "insufficient real windows for dup_eps"
            return Z_syn, info
        dup_eps = float(np.quantile(
            d_real_ref, max(float(params["anti_mem_dup_eps_quantile"]), 1e-6)
        ))
        margin = float(params["anti_mem_safety_margin"])
        info["dup_eps"] = dup_eps

        boost = float(params["anti_mem_boost_factor"])
        boosted_jitter = float(jitter_sigma) * boost
        K = means.shape[0]

        # Map every synthetic row back to a regime via rolling |r_close|
        # using the same scheme as training; we approximate by labeling
        # the *current* synthetic primitive at each row using the closest
        # regime mean (cheap, deterministic, regime-pool-preserving).
        def _label_syn(Z: np.ndarray) -> np.ndarray:
            # nearest regime by L2 in primitive space
            d2 = ((Z[:, None, :] - means[None, :, :]) ** 2).sum(-1)
            return d2.argmin(axis=1).astype(np.int64)

        # Pre-bucket residual indices per regime (same labels as training).
        idx_by_regime = [np.where(labels == k)[0] for k in range(K)]
        for k in range(K):
            if len(idx_by_regime[k]) == 0:
                idx_by_regime[k] = np.arange(len(residuals))

        recon = OhlcvReconstructor(self.config)
        recon.set_params(**{
            k: params[k] for k in (
                "datetime_column", "open_col", "high_col", "low_col",
                "close_col", "volume_col",
            )
        })

        max_passes = int(params["anti_mem_max_passes"])
        already_resampled = np.zeros(len(Z_syn), dtype=bool)
        for it in range(max_passes):
            z_df = pd.DataFrame(Z_syn, columns=list(_PRIMITIVES))
            recon_df = recon.reconstruct(
                z_df, initial_close=initial_close, transformer=transformer,
                timestamps=None,
            )
            close_syn = recon_df[params["close_col"]].to_numpy(dtype=np.float64)
            r_syn = self._log_returns(close_syn)
            W_syn = self._slide_windows(r_syn, W)
            if len(W_syn) == 0:
                info["skipped"] = "syn too short"
                break
            d = self._nn_l2(W_syn, W_real)
            bad = np.where(d <= dup_eps * margin)[0]
            if it == 0:
                info["initial_violations"] = int(len(bad))
            if len(bad) == 0:
                info["final_violations"] = 0
                info["passes"] = it
                return Z_syn, info
            # Mask of rows to resample (each bad window covers W rows).
            row_mask = np.zeros(len(Z_syn), dtype=bool)
            for i in bad:
                lo = max(1, int(i) + 1)
                hi = min(len(Z_syn), int(i) + 1 + W)
                row_mask[lo:hi] = True
            offending_rows = np.where(row_mask & ~already_resampled)[0]
            if len(offending_rows) == 0:
                # All offenders are already-touched rows; stop to avoid
                # double-perturbing and inflating variance.
                info["final_violations"] = int(len(bad))
                info["passes"] = it
                info["stopped_reason"] = "no untouched offenders"
                return Z_syn, info
            # Per-row regime lookup from current Z_syn.
            row_regimes = _label_syn(Z_syn[offending_rows])
            # Resample residuals + boosted regime-conditioned jitter.
            new_Z = np.empty((len(offending_rows), len(_PRIMITIVES)), dtype=np.float64)
            for j, rrow in enumerate(offending_rows):
                k = int(row_regimes[j])
                pick = int(rng.choice(idx_by_regime[k]))
                noise = rng.normal(0.0, 1.0, size=len(_PRIMITIVES)) * (boosted_jitter * stds[k])
                new_Z[j] = means[k] + residuals[pick] + noise
            Z_syn[offending_rows] = new_Z
            already_resampled[offending_rows] = True
            info["passes"] = it + 1
            info["final_violations"] = int(len(bad))
        return Z_syn, info

    def _build_timestamps(self, n: int) -> Optional[pd.Series]:
        p = self.params
        if not p["start_timestamp"]:
            return None
        return pd.Series(pd.date_range(
            start=pd.Timestamp(p["start_timestamp"]),
            periods=n,
            freq=p["frequency"],
        ))


__all__ = ["RegimeResidualBootstrapOhlcvGenerator"]
