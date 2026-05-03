"""Stage 4.3 augmentation-orchestrator (manifest emitter).

Implements the *configuration* side of the Phase 4 §5 ablation matrix:

    real_train_only
    synthetic_train_only
    real_plus_synthetic_0_25x
    real_plus_synthetic_0_50x
    real_plus_synthetic_1_00x
    synthetic_pretrain_then_real_finetune

For each cell, the orchestrator:

1. Builds a downstream-training input file by mixing real and synthetic
   bars at the requested ratio (concatenation with a ``synthetic_origin``
   boolean column so the downstream replay buffer can up/down-weight).
2. Writes a per-cell manifest JSON containing the matched-experiment
   metadata REQUIRED by Phase 4 §5 (candidate run id, generator family,
   generator config hash, augmentation ratio, downstream algorithm,
   seed set, real validation file, cost scenario).
3. Appends the manifest to a top-level ``ablation_index.json``.

This plugin does NOT execute the downstream training itself.  Stage
4.3 §6 promotion requires the matched real-validation experiment to be
run by the predictor or agent-multi repos; the manifests produced here
are the inputs to those runs.  Keeping the actual training out of this
repo prevents synthetic-datagen from depending on either downstream
codebase and keeps the Phase 4 firewall clean.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


_ABLATION_CELLS_DEFAULT: List[Dict[str, Any]] = [
    {"id": "real_train_only",                  "real_ratio": 1.0, "synthetic_ratio": 0.0,  "mode": "train"},
    {"id": "synthetic_train_only",             "real_ratio": 0.0, "synthetic_ratio": 1.0,  "mode": "train"},
    {"id": "real_plus_synthetic_0_25x",        "real_ratio": 1.0, "synthetic_ratio": 0.25, "mode": "augment"},
    {"id": "real_plus_synthetic_0_50x",        "real_ratio": 1.0, "synthetic_ratio": 0.50, "mode": "augment"},
    {"id": "real_plus_synthetic_1_00x",        "real_ratio": 1.0, "synthetic_ratio": 1.00, "mode": "augment"},
    {"id": "synthetic_pretrain_then_real_finetune",
     "real_ratio": 1.0, "synthetic_ratio": 1.0, "mode": "pretrain_then_finetune"},
]


class AugmentationManifestPipeline:
    """Emits the six matched-experiment manifests for Phase 4 §5."""

    plugin_params: Dict[str, Any] = {
        "real_train": None,
        "synthetic_data": None,
        "real_validation": None,
        "output_dir": "experiments/synthetic_data/augmentation",
        "candidate_run_id": None,
        "generator_family_id": None,
        "generator_config_hash": None,
        "downstream_algorithm": "sac",
        "downstream_repo": "agent-multi",
        "seeds": [0, 1, 2, 3, 4],     # Phase 4 §3.9 minimum 5 seeds
        "cost_scenarios": ["base", "plus_50pct", "plus_100pct"],  # §3.5
        "real_only_baseline_id": None,
        "datetime_column": "DATE_TIME",
        "asset_id": None,
        "timeframe": None,
        "augmentation_ratios": [0.25, 0.5, 1.0],
        "ablation_cells": None,        # override if non-default
        "matched_split_seed": 0,
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
    def _read(self, path: str) -> pd.DataFrame:
        if str(path).lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _hash_file(self, path: str) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build_mixed(
        self, real: pd.DataFrame, syn: pd.DataFrame,
        real_ratio: float, syn_ratio: float,
    ) -> pd.DataFrame:
        # Take all real (real_ratio is the multiplier on the SLICE used).
        n_real = int(round(len(real) * real_ratio))
        n_syn = int(round(len(real) * syn_ratio)) if real_ratio > 0 else len(syn)
        n_syn = min(n_syn, len(syn))
        real_used = real.iloc[:n_real].copy()
        syn_used = syn.iloc[:n_syn].copy()
        real_used["synthetic_origin"] = 0
        syn_used["synthetic_origin"] = 1
        # Append synthetic AFTER real so causal time-ordering is unambiguous.
        mixed = pd.concat([real_used, syn_used], ignore_index=True)
        return mixed

    # ------------------------------------------------------------------
    def run_pipeline(self) -> Dict[str, Any]:
        return self.evaluate()    # alias used by main.py evaluator dispatch

    def evaluate(self) -> Dict[str, Any]:   # invoked by --mode evaluate
        return self.generate_manifests()

    def generate_manifests(self) -> Dict[str, Any]:
        p = self.params
        if not p["real_train"] or not p["synthetic_data"]:
            raise ValueError("real_train and synthetic_data are both required")
        os.makedirs(p["output_dir"], exist_ok=True)

        real = self._read(p["real_train"])
        syn = self._read(p["synthetic_data"])

        cells: List[Dict[str, Any]] = list(p["ablation_cells"] or _ABLATION_CELLS_DEFAULT)
        index_rows: List[Dict[str, Any]] = []
        real_hash = self._hash_file(p["real_train"])
        syn_hash = self._hash_file(p["synthetic_data"])
        val_hash = self._hash_file(p["real_validation"]) if p["real_validation"] else None

        for cell in cells:
            cell_id = cell["id"]
            cell_dir = os.path.join(p["output_dir"], cell_id)
            os.makedirs(cell_dir, exist_ok=True)

            mixed = self._build_mixed(
                real, syn, float(cell["real_ratio"]), float(cell["synthetic_ratio"]),
            )
            mixed_path = os.path.join(cell_dir, "training_data.csv")
            mixed.to_csv(mixed_path, index=False)
            mixed_hash = self._hash_file(mixed_path)

            for cost in p["cost_scenarios"]:
                manifest = {
                    "candidate_run_id": p["candidate_run_id"],
                    "real_only_baseline_id": p["real_only_baseline_id"],
                    "ablation_cell_id": cell_id,
                    "ablation_mode": cell["mode"],   # train | augment | pretrain_then_finetune
                    "real_ratio": cell["real_ratio"],
                    "synthetic_ratio": cell["synthetic_ratio"],
                    "asset_id": p["asset_id"],
                    "timeframe": p["timeframe"],
                    "generator_family_id": p["generator_family_id"],
                    "generator_config_hash": p["generator_config_hash"],
                    "downstream_repo": p["downstream_repo"],
                    "downstream_algorithm": p["downstream_algorithm"],
                    "seeds": list(p["seeds"]),
                    "cost_scenario": cost,
                    "training_data_file": os.path.relpath(mixed_path),
                    "training_data_sha256": mixed_hash,
                    "real_train_sha256": real_hash,
                    "synthetic_data_sha256": syn_hash,
                    "real_validation_file": p["real_validation"],
                    "real_validation_sha256": val_hash,
                    "n_real_rows_used": int((mixed["synthetic_origin"] == 0).sum()),
                    "n_synthetic_rows_used": int((mixed["synthetic_origin"] == 1).sum()),
                    # Phase 4 §6 promotion fields (filled in by downstream runner):
                    "downstream_metrics": None,
                    "beats_real_only": None,
                    "survives_pessimistic_costs": None,
                    "improvement_concentrated_in_one_seed": None,
                }
                manifest_path = os.path.join(cell_dir, f"manifest_{cost}.json")
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2, default=str)
                index_rows.append({
                    "cell_id": cell_id,
                    "cost_scenario": cost,
                    "manifest": os.path.relpath(manifest_path),
                })

        index_path = os.path.join(p["output_dir"], "ablation_index.json")
        with open(index_path, "w") as f:
            json.dump({
                "candidate_run_id": p["candidate_run_id"],
                "asset_id": p["asset_id"],
                "timeframe": p["timeframe"],
                "generator_family_id": p["generator_family_id"],
                "generator_config_hash": p["generator_config_hash"],
                "n_cells": len(cells),
                "n_cost_scenarios": len(p["cost_scenarios"]),
                "n_seeds": len(p["seeds"]),
                "manifests": index_rows,
            }, f, indent=2, default=str)
        return {
            "ablation_index": index_path,
            "n_cells": len(cells),
            "n_manifests": len(index_rows),
            "output_dir": p["output_dir"],
        }


__all__ = ["AugmentationManifestPipeline"]
