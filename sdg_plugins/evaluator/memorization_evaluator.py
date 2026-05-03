"""Memorization evaluator for synthetic OHLCV.

Detects copy-from-train pathologies in synthetic generators:

* nearest-neighbor distance (mean / 5th-percentile) over fixed-length return windows
* duplicate-window count (windows with NN-distance below ``dup_eps``)
* longest copied subsequence: longest run where every consecutive
  synthetic return is the closest to the same real return at offset +1
* real-vs-synthetic classifier AUC: a tiny logistic-regression on
  per-window summary stats (no sklearn dep — closed-form via
  Mann-Whitney U for the AUC of a single-feature ranking)

Pass/fail gates (Phase 4 §4.2):

* ``classifier_auc_max``       = 0.70
* ``duplicate_window_rate_max``= 1e-3
* ``max_nn_overlap_rate_max``  = 1e-3   (windows with NN cosine > 0.95)
* ``copied_subseq_ratio_max``  = 0.50   (longest run / window size)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _log_returns(c: np.ndarray) -> np.ndarray:
    return np.diff(np.log(c))


def _windows(x: np.ndarray, w: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x) - w + 1
    if n <= 0:
        return np.zeros((0, w), dtype=np.float64)
    if n <= max_n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=max_n, replace=False)
    return np.stack([x[i:i + w] for i in idx])


def _pairwise_min_l2(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """For each row in ``query``, return the L2 distance to its NN in ``ref``."""
    # Broadcast in chunks to avoid O(N*M*W) memory blow-up.
    out = np.empty(len(query), dtype=np.float64)
    chunk = max(1, 4096 // max(1, ref.shape[1]))
    ref_norm = (ref * ref).sum(1)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        q_norm = (q * q).sum(1)[:, None]
        d2 = q_norm + ref_norm[None, :] - 2.0 * q @ ref.T
        np.maximum(d2, 0.0, out=d2)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def _pairwise_max_cos(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(query, axis=1, keepdims=True) + 1e-12
    rn = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
    q = query / qn
    r = ref / rn
    out = np.empty(len(query), dtype=np.float64)
    chunk = max(1, 4096 // max(1, r.shape[1]))
    for i in range(0, len(query), chunk):
        out[i:i + chunk] = (q[i:i + chunk] @ r.T).max(axis=1)
    return out


def _longest_consecutive_match(syn: np.ndarray, real: np.ndarray, tol: float) -> int:
    """Greedy longest-run-of-consecutive-NN.

    For each synthetic point, find its NN index in ``real``.  Then count
    the longest streak where ``nn_index[t+1] == nn_index[t] + 1`` AND
    the per-step distance is below ``tol``.
    """
    if len(syn) == 0 or len(real) == 0:
        return 0
    # 1-D NN: argmin |syn[t] - real[j]| for each t.  O(n*m) but n,m bounded.
    diffs = np.abs(syn[:, None] - real[None, :])
    nn_idx = diffs.argmin(axis=1)
    nn_dist = diffs[np.arange(len(syn)), nn_idx]
    best = cur = 0
    for t in range(1, len(syn)):
        if nn_idx[t] == nn_idx[t - 1] + 1 and nn_dist[t] < tol:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best + 1) if best > 0 else 0


def _auc_from_score(score: np.ndarray, label: np.ndarray) -> float:
    """AUC via Mann-Whitney U identity (no sklearn)."""
    # rank the scores; ties get average rank.
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))   # symmetric AUC


class MemorizationEvaluator:
    plugin_params: Dict[str, Any] = {
        "synthetic_data": None,
        "real_data": None,
        "close_col": "CLOSE",
        "metrics_file": None,
        "window": 32,
        "max_windows": 500,
        "dup_eps_quantile": 0.001,        # the 0.1%-quantile of real-real NN distances
        "max_nn_cos_overlap": 0.95,
        "seed": 0,
        # Pass/fail thresholds (Phase 4 §4.2):
        "classifier_auc_max": 0.70,
        "duplicate_window_rate_max": 1e-3,
        "max_nn_overlap_rate_max": 1e-3,
        "copied_subseq_ratio_max": 0.50,
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
        if not p["synthetic_data"] or not p["real_data"]:
            raise ValueError("synthetic_data and real_data are both required")
        syn = self._read(p["synthetic_data"])[p["close_col"]].to_numpy(dtype=np.float64)
        real = self._read(p["real_data"])[p["close_col"]].to_numpy(dtype=np.float64)
        return self.evaluate_arrays(real, syn)

    def evaluate_arrays(self, real_close: np.ndarray, syn_close: np.ndarray) -> Dict[str, Any]:
        p = self.params
        rng = np.random.default_rng(int(p["seed"]))
        r_real = _log_returns(real_close)
        r_syn = _log_returns(syn_close)
        w = int(p["window"])
        if len(r_real) < w + 1 or len(r_syn) < w + 1:
            return {"valid": False, "reason": "samples too short for window"}

        W_real = _windows(r_real, w, int(p["max_windows"]), rng)
        W_syn = _windows(r_syn, w, int(p["max_windows"]), rng)

        # NN distance distributions
        d_syn_to_real = _pairwise_min_l2(W_syn, W_real)
        # Reference: real-to-real NN distance (excluding self via small subsample swap).
        half = max(2, len(W_real) // 2)
        d_real_ref = _pairwise_min_l2(W_real[:half], W_real[half:])
        if len(d_real_ref) == 0:
            d_real_ref = d_syn_to_real
        dup_eps = float(np.quantile(d_real_ref, max(p["dup_eps_quantile"], 1e-6)))
        dup_count = int((d_syn_to_real <= dup_eps).sum())
        dup_rate = float(dup_count / max(1, len(d_syn_to_real)))

        cos_max = _pairwise_max_cos(W_syn, W_real)
        nn_overlap_count = int((cos_max >= p["max_nn_cos_overlap"]).sum())
        nn_overlap_rate = float(nn_overlap_count / max(1, len(cos_max)))

        # Longest copied subsequence (1-D, on raw returns, with tol = dup_eps/sqrt(w)).
        tol = float(dup_eps / max(1, np.sqrt(w)))
        lcs = _longest_consecutive_match(r_syn, r_real, tol)
        lcs_ratio = float(lcs / max(1, w))

        # Single-feature classifier AUC: feature = window std (a known
        # discriminative summary).  Replace with a real classifier if
        # sklearn becomes available.
        feat_real = W_real.std(axis=1)
        feat_syn = W_syn.std(axis=1)
        score = np.concatenate([feat_real, feat_syn])
        label = np.concatenate([np.zeros(len(feat_real)), np.ones(len(feat_syn))])
        auc = _auc_from_score(score, label)

        gates = {
            "classifier_auc_pass": auc <= p["classifier_auc_max"],
            "duplicate_rate_pass": dup_rate <= p["duplicate_window_rate_max"],
            "nn_overlap_pass": nn_overlap_rate <= p["max_nn_overlap_rate_max"],
            "copied_subseq_pass": lcs_ratio <= p["copied_subseq_ratio_max"],
        }
        gates["all_pass"] = all(gates.values())

        report = {
            "valid": True,
            "n_real_windows": int(len(W_real)),
            "n_syn_windows": int(len(W_syn)),
            "window": w,
            "nn_l2_mean_syn_to_real": float(d_syn_to_real.mean()),
            "nn_l2_q05_syn_to_real": float(np.quantile(d_syn_to_real, 0.05)),
            "real_to_real_nn_eps_quantile": dup_eps,
            "duplicate_window_count": dup_count,
            "duplicate_window_rate": dup_rate,
            "max_cos_overlap_mean": float(cos_max.mean()),
            "nn_overlap_count": nn_overlap_count,
            "nn_overlap_rate": nn_overlap_rate,
            "longest_copied_subseq": lcs,
            "copied_subseq_ratio": lcs_ratio,
            "classifier_auc_window_std": auc,
            "thresholds": {
                "classifier_auc_max": p["classifier_auc_max"],
                "duplicate_window_rate_max": p["duplicate_window_rate_max"],
                "max_nn_overlap_rate_max": p["max_nn_overlap_rate_max"],
                "copied_subseq_ratio_max": p["copied_subseq_ratio_max"],
            },
            "gates": gates,
        }
        if p["metrics_file"]:
            os.makedirs(os.path.dirname(os.path.abspath(p["metrics_file"])) or ".", exist_ok=True)
            with open(p["metrics_file"], "w") as f:
                json.dump(report, f, indent=2)
        return report


__all__ = ["MemorizationEvaluator"]
