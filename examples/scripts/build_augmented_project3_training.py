"""Build a synthetic-augmented training CSV for Project 3 SAC training.

Pipeline:

    1. Load the 8-year ETH 4h ``model_ready`` CSV.
    2. Slice the 4-year training window (matches the agent-multi config).
    3. Train ``stationary_bootstrap_ohlcv_trainer`` on the OHLCV slice.
    4. Generate 1 year (≈2,190 bars) of synthetic OHLCV ending at the
       real-data start timestamp.
    5. Concatenate synthetic + real OHLCV (synthetic first, in time order).
    6. Recompute the full ``tech_stat`` feature matrix on the
       concatenation via ``TechStatFeatureEngine``.
    7. Write the augmented ``model_ready`` CSV plus an audit-sidecar JSON.

Usage:
    python -m examples.scripts.build_augmented_project3_training \\
        --method stationary_bootstrap \\
        --synthetic_years 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd

from app.audit import build_audit_record
from app.family_registry import register_family, write_protocol_stub
from app.synthetic_ledger import append_ledger
from sdg_plugins.evaluator.financial_distribution_evaluator import (
    FinancialDistributionEvaluator,
)
from sdg_plugins.evaluator.memorization_evaluator import MemorizationEvaluator
from sdg_plugins.evaluator.ohlcv_algebraic_evaluator import OhlcvAlgebraicEvaluator
from sdg_plugins.feature_engine.tech_stat_feature_engine import TechStatFeatureEngine
from sdg_plugins.generator.stationary_bootstrap_ohlcv_generator import (
    StationaryBootstrapOhlcvGenerator,
)
from sdg_plugins.trainer.stationary_bootstrap_ohlcv_trainer import (
    StationaryBootstrapOhlcvTrainer,
)


REAL_INPUT = "examples/data/ethusdt_4h_full_8yr.csv"
TRAIN_START = "2017-09-28 04:00:00"
TRAIN_END = "2021-09-28 00:00:00"
HELDOUT_BOUNDARY = "2021-09-28 00:00:00"

OHLCV_COLS = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
DATETIME_COL = "DATE_TIME"


def _slice_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    mask = (df[DATETIME_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATETIME_COL] < pd.Timestamp(TRAIN_END))
    return df.loc[mask].reset_index(drop=True)


def _build_pre_real_synthetic_timestamps(start_real: pd.Timestamp, n: int, freq: str = "4h") -> pd.DatetimeIndex:
    """N timestamps ending one bar before ``start_real``."""
    end = start_real - pd.Timedelta(freq)
    return pd.date_range(end=end, periods=n, freq=freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="stationary_bootstrap",
                    choices=["stationary_bootstrap"],
                    help="Synthetic generation method (currently only the Phase-4-ready bootstrap)")
    ap.add_argument("--synthetic_years", type=float, default=1.0,
                    help="How many years of synthetic data to prepend (≈2190 rows/year at 4h)")
    ap.add_argument("--output_dir", default="experiments/synthetic_data/project3_eth_4h")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    method_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_dir, exist_ok=True)

    # 1. Load + slice
    print(f"[1/7] Loading {REAL_INPUT}...")
    full = pd.read_csv(REAL_INPUT)
    full[DATETIME_COL] = pd.to_datetime(full[DATETIME_COL])
    train_real = _slice_train(full)
    print(f"      Real train slice: {len(train_real)} rows, "
          f"{train_real[DATETIME_COL].iloc[0]} -> {train_real[DATETIME_COL].iloc[-1]}")

    # 2. OHLCV-only file for the trainer
    train_ohlcv_path = os.path.join(method_dir, "real_train_ohlcv.csv")
    train_real[[DATETIME_COL] + OHLCV_COLS].to_csv(train_ohlcv_path, index=False)

    # 3. Train
    print(f"[2/7] Training {args.method}...")
    family_id = "stationary_bootstrap_v1"
    base_cfg = {
        "financial_mode": True,
        "asset_id": "ethusdt",
        "timeframe": "4h",
        "datetime_column": DATETIME_COL,
        "open_col": "OPEN", "high_col": "HIGH", "low_col": "LOW",
        "close_col": "CLOSE", "volume_col": "VOLUME",
        "seed": args.seed,
        "generator_family_id": family_id,
        "synthetic_ablation_id": f"ratio_{args.synthetic_years:.2f}yr",
        "synthetic_use_case": "rl_training_augmentation",
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "heldout_boundary": HELDOUT_BOUNDARY,
        "synthetic_ledger_path": os.path.join(args.output_dir, "SYNTHETIC_LEDGER.csv"),
    }
    model_path = os.path.join(method_dir, "bootstrap.npz")
    trainer = StationaryBootstrapOhlcvTrainer({
        **base_cfg,
        "train_data": train_ohlcv_path,
        "save_model": model_path,
        "block_length_mean": 24,
    })
    trainer.train()
    audit_fit = build_audit_record(
        {**base_cfg, "trainer": "stationary_bootstrap_ohlcv_trainer",
         "train_data": train_ohlcv_path, "save_model": model_path},
        input_files={"train_data": train_ohlcv_path},
    )
    append_ledger(base_cfg, kind="fit", audit=audit_fit,
                  extra={"model_file": model_path})

    # 4. Generate
    print(f"[3/7] Generating ~{int(args.synthetic_years * 2190)} synthetic 4h bars...")
    n_syn = int(round(args.synthetic_years * 2190))   # 365.25*24/4 ≈ 2191.5
    syn_start = train_real[DATETIME_COL].iloc[0]
    syn_timestamps = _build_pre_real_synthetic_timestamps(syn_start, n_syn, "4h")
    syn_path = os.path.join(method_dir, "synthetic_ohlcv.csv")
    gen = StationaryBootstrapOhlcvGenerator({
        **base_cfg,
        "load_model": model_path,
        "n_samples": n_syn,
        "output_file": syn_path,
        "start_timestamp": str(syn_timestamps[0]),
        "frequency": "4h",
    })
    gen.run_generate()
    audit_gen = build_audit_record(
        {**base_cfg, "generator": "stationary_bootstrap_ohlcv_generator",
         "load_model": model_path, "output_file": syn_path,
         "n_samples": n_syn},
        input_files={"model": model_path},
    )
    append_ledger(base_cfg, kind="generate", audit=audit_gen,
                  extra={"output_file": syn_path})
    syn_df = pd.read_csv(syn_path)
    syn_df[DATETIME_COL] = pd.to_datetime(syn_df[DATETIME_COL])
    print(f"      Synthetic span: {syn_df[DATETIME_COL].iloc[0]} -> {syn_df[DATETIME_COL].iloc[-1]}")

    # 5. Validate algebraic
    print("[4/7] Algebraic validation...")
    alg = OhlcvAlgebraicEvaluator({
        **base_cfg, "synthetic_data": syn_path,
        "metrics_file": os.path.join(method_dir, "algebraic_metrics.json"),
    }).evaluate()
    n_violations = sum(int(v) for v in alg.get("violations", {}).values())
    if n_violations:
        print(f"      ABORT: {n_violations} algebraic violations")
        sys.exit(2)
    print(f"      OK: 0 violations on {alg.get('n_rows')} rows")

    # 5b. Distribution + memorization
    print("[5/7] Distribution + memorization evaluators...")
    dist = FinancialDistributionEvaluator({
        **base_cfg, "synthetic_data": syn_path, "real_data": train_ohlcv_path,
        "metrics_file": os.path.join(method_dir, "distribution_metrics.json"),
    }).evaluate()
    mem = MemorizationEvaluator({
        **base_cfg, "synthetic_data": syn_path, "real_data": train_ohlcv_path,
        "metrics_file": os.path.join(method_dir, "memorization_metrics.json"),
        "window": 32, "max_windows": 200,
    }).evaluate()
    print(f"      distribution gates: {dist.get('gates')}")
    print(f"      memorization gates: {mem.get('gates')}")

    # 6. Concatenate synthetic + full real (synthetic first), recompute features
    print("[6/7] Concatenating + recomputing tech_stat features...")
    syn_ohlcv = syn_df[[DATETIME_COL] + OHLCV_COLS].copy()
    real_ohlcv = full[[DATETIME_COL] + OHLCV_COLS].copy()
    combined = pd.concat([syn_ohlcv, real_ohlcv], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)
    fe = TechStatFeatureEngine()
    augmented = fe.compute(combined)
    augmented["synthetic_origin"] = (
        (augmented[DATETIME_COL] < pd.Timestamp(TRAIN_START)).astype(int)
    )
    aug_path = os.path.join(method_dir, f"ethusdt_4h_tech_stat_augmented_{args.method}.csv")
    augmented.to_csv(aug_path, index=False)
    print(f"      Augmented CSV: {aug_path} ({len(augmented)} rows, "
          f"{int(augmented['synthetic_origin'].sum())} synthetic)")

    # 7. Audit + family registry
    print("[7/7] Updating ledger + registry + protocol stub...")
    gate_summary = {
        "algebraic_violations": n_violations,
        "ks_returns_pvalue": dist.get("ks_returns", {}).get("pvalue"),
        "wasserstein_returns_ratio": dist.get("wasserstein_returns_ratio"),
        "duplicate_window_rate": mem.get("duplicate_window_rate"),
        "copied_subseq_ratio": mem.get("copied_subseq_ratio"),
        "classifier_auc_window_std": mem.get("classifier_auc_window_std"),
    }
    register_family(
        family_id,
        description="Politis-Romano stationary bootstrap on OHLCV primitives.",
        assumptions=[
            "log-returns are weakly stationary on the fit window",
            "block geometric distribution with mean=block_length_mean=24",
            "validity-by-construction reconstruction of OHLCV from primitives",
        ],
        fit_windows=[{"path": train_ohlcv_path, "n_rows": len(train_real),
                      "start": TRAIN_START, "end": TRAIN_END}],
        gate_values=gate_summary,
        config_hash=audit_fit.get("config_hash"),
        registry_path=os.path.join(args.output_dir, "generator_family_registry.json"),
    )
    write_protocol_stub(
        family_id,
        description="Project 3 ETHUSDT 4h augmentation candidate (Phase 4 §4.1).",
        assumptions=["weak stationarity on 2017-09-28 → 2021-09-28"],
        gate_values=gate_summary,
        output_path=os.path.join(method_dir, f"{family_id}_protocol.md"),
    )

    summary = {
        "method": args.method,
        "family_id": family_id,
        "n_real_train": int(len(train_real)),
        "n_synthetic": int(len(syn_df)),
        "augmented_csv": aug_path,
        "model_file": model_path,
        "synthetic_csv": syn_path,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "heldout_boundary": HELDOUT_BOUNDARY,
        "gate_summary": gate_summary,
        "distribution_gates": dist.get("gates"),
        "memorization_gates": mem.get("gates"),
    }
    summary_path = os.path.join(method_dir, "augmentation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
