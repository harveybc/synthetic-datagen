"""Stationary block-bootstrap trainer for OHLCV synthetic generation.

Fits the :class:`OhlcvTransformer` on the train split only and stores the
matrix of transformed primitives for later block sampling by the matching
generator.  No neural-net dependency; CPU-safe; deterministic given the
seed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer


class StationaryBootstrapOhlcvTrainer:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "train_data": None,
        "save_model": "stationary_bootstrap_ohlcv.npz",
        "save_metadata": None,
        "block_length_mean": 32,
        "seed": 42,
        "data_format": "auto",  # csv | parquet | auto
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
        self.fitted_matrix: Optional[np.ndarray] = None

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    # ------------------------------------------------------------------
    def _read_data(self, path: str) -> pd.DataFrame:
        # Phase 4 §7: refuse to open Stage C / 2025-heldout paths.
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
        """Project 3 protection: refuse to train on rows >= heldout_boundary."""
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
                return  # explicit non-research opt-in (NEVER for Project 3).
            raise ValueError(
                f"Training input contains rows on or after heldout_boundary={boundary}; "
                "refusing to fit the generator (Project 3 leakage guard)."
            )

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        p = self.params
        if not p["train_data"]:
            raise ValueError("train_data is required")
        # train_data may be either a string path or a list of paths (the CLI
        # uses ``nargs="+"`` for backward-compat); we use the first.
        td = p["train_data"]
        if isinstance(td, (list, tuple)):
            if not td:
                raise ValueError("train_data list is empty")
            td = td[0]
        df_full = self._read_data(td)
        df_train = self._slice_train(df_full)
        self._heldout_guard(df_train)

        self.transformer = OhlcvTransformer(self.config)
        # Forward primitive-column overrides.
        self.transformer.set_params(**{
            k: p[k] for k in (
                "datetime_column", "open_col", "high_col", "low_col",
                "close_col", "volume_col",
            )
        })
        self.transformer.fit(df_train)
        z = self.transformer.transform(df_train)
        self.fitted_matrix = z[["r_close", "r_open", "d_high", "d_low", "v"]].to_numpy()

        meta = self._save(df_train)
        return meta

    # ------------------------------------------------------------------
    def _save(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        p = self.params
        os.makedirs(os.path.dirname(os.path.abspath(p["save_model"])) or ".", exist_ok=True)
        state = {
            "matrix": self.fitted_matrix,
            "transformer_state": json.dumps(self.transformer.state_dict()),
            "block_length_mean": int(p["block_length_mean"]),
            "seed": int(p["seed"]),
            "first_close": float(df_train[p["close_col"]].iloc[0]),
            "last_close": float(df_train[p["close_col"]].iloc[-1]),
            "n_train_rows": int(len(df_train)),
        }
        np.savez_compressed(p["save_model"], **{k: v for k, v in state.items() if k != "transformer_state"})
        # Save the transformer state as a sibling JSON for portability.
        side = os.path.splitext(p["save_model"])[0] + ".transformer.json"
        with open(side, "w") as f:
            f.write(state["transformer_state"])
        meta = {
            "model_file": p["save_model"],
            "transformer_file": side,
            "n_train_rows": state["n_train_rows"],
            "block_length_mean": state["block_length_mean"],
            "seed": state["seed"],
        }
        if p["save_metadata"]:
            with open(p["save_metadata"], "w") as f:
                json.dump(meta, f, indent=2, default=str)
        return meta


__all__ = ["StationaryBootstrapOhlcvTrainer"]
