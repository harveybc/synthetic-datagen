"""Build a synthetic-augmented training CSV for Project 3 SAC training.

Pipeline:

    1. Load the 8-year ETH 4h ``model_ready`` CSV.
    2. Slice the 4-year training window (matches the agent-multi config).
    3. Train ``stationary_bootstrap_ohlcv_trainer`` on the OHLCV slice.
    4. Generate 1 year (≈2,190 bars) of synthetic OHLCV ending at the
       real-data start timestamp.
    5. Concatenate synthetic + real OHLCV (synthetic first, in time order).
    6. Abort unless algebraic, distribution, and memorization gates pass.
    7. Recompute the full ``tech_stat`` feature matrix on the
       pre-Stage-C concatenation via ``TechStatFeatureEngine``.
    8. Write the augmented ``model_ready`` CSV plus an audit-sidecar JSON.

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
from sdg_plugins.generator.regime_residual_bootstrap_ohlcv_generator import (
    RegimeResidualBootstrapOhlcvGenerator,
)
from sdg_plugins.trainer.stationary_bootstrap_ohlcv_trainer import (
    StationaryBootstrapOhlcvTrainer,
)
from sdg_plugins.trainer.regime_residual_bootstrap_ohlcv_trainer import (
    RegimeResidualBootstrapOhlcvTrainer,
)


_TRAINERS = {
    "stationary_bootstrap": StationaryBootstrapOhlcvTrainer,
    "regime_residual_bootstrap": RegimeResidualBootstrapOhlcvTrainer,
}
_GENERATORS = {
    "stationary_bootstrap": StationaryBootstrapOhlcvGenerator,
    "regime_residual_bootstrap": RegimeResidualBootstrapOhlcvGenerator,
}
_FAMILY_IDS = {
    "stationary_bootstrap": "stationary_bootstrap_v1",
    "regime_residual_bootstrap": "regime_residual_bootstrap_v1",
}


REAL_INPUT = "examples/data/ethusdt_4h_full_8yr.csv"
TRAIN_START = "2017-09-28 04:00:00"
TRAIN_END = "2021-09-28 00:00:00"
PROJECT3_HELDOUT_BOUNDARY = "2025-01-01 00:00:00"

OHLCV_COLS = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
DATETIME_COL = "DATE_TIME"


def _slice_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    mask = (df[DATETIME_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATETIME_COL] < pd.Timestamp(TRAIN_END))
    return df.loc[mask].reset_index(drop=True)


def _slice_pre_stage_c(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows allowed before the Project 3 Stage C firewall."""
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    mask = df[DATETIME_COL] < pd.Timestamp(PROJECT3_HELDOUT_BOUNDARY)
    return df.loc[mask].reset_index(drop=True)


def _build_pre_real_synthetic_timestamps(start_real: pd.Timestamp, n: int, freq: str = "4h") -> pd.DatetimeIndex:
    """N timestamps ending one bar before ``start_real``."""
    end = start_real - pd.Timedelta(freq)
    return pd.date_range(end=end, periods=n, freq=freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="stationary_bootstrap",
                    choices=sorted(_TRAINERS.keys()),
                    help="Synthetic generation method")
    ap.add_argument("--synthetic_years", type=float, default=1.0,
                    help="How many years of synthetic data to prepend (≈2190 rows/year at 4h)")
    ap.add_argument("--output_dir", default="experiments/synthetic_data/project3_eth_4h")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow_diagnostic_output",
        action="store_true",
        help=(
            "Write diagnostic artifacts even if quality gates fail. NEVER use "
            "the resulting CSV for Project 3 policy training."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    method_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_dir, exist_ok=True)

    # 1. Load + slice
    print(f"[1/7] Loading {REAL_INPUT}...")
    full = pd.read_csv(REAL_INPUT)
    full[DATETIME_COL] = pd.to_datetime(full[DATETIME_COL])
    train_real = _slice_train(full)
    pre_stage_c_real = _slice_pre_stage_c(full)
    print(f"      Real train slice: {len(train_real)} rows, "
          f"{train_real[DATETIME_COL].iloc[0]} -> {train_real[DATETIME_COL].iloc[-1]}")
    print(f"      Pre-Stage-C panel: {len(pre_stage_c_real)} rows, "
          f"{pre_stage_c_real[DATETIME_COL].iloc[0]} -> {pre_stage_c_real[DATETIME_COL].iloc[-1]}")

    # 2. OHLCV-only file for the trainer
    train_ohlcv_path = os.path.join(method_dir, "real_train_ohlcv.csv")
    train_real[[DATETIME_COL] + OHLCV_COLS].to_csv(train_ohlcv_path, index=False)

    # 3. Train
    print(f"[2/7] Training {args.method}...")
    family_id = _FAMILY_IDS[args.method]
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
        "heldout_boundary": PROJECT3_HELDOUT_BOUNDARY,
        "project3_mode": True,
        "synthetic_ledger_path": os.path.join(args.output_dir, "SYNTHETIC_LEDGER.csv"),
    }
    model_path = os.path.join(method_dir, "bootstrap.npz")
    trainer_cls = _TRAINERS[args.method]
    trainer_cfg = {
        **base_cfg,
        "train_data": train_ohlcv_path,
        "save_model": model_path,
        "block_length_mean": 24,
    }
    if args.method == "regime_residual_bootstrap":
        trainer_cfg.update({
            # Shorter blocks + per-row regime-conditioned residual sampling +
            # moderate jitter strikes the best balance on ETH 4h between
            # memorization gates (need duplicate_rate=0/200) and the
            # distribution KS p-value gate (need >0.01).
            "block_length_mean": 4,
            "n_regimes": 3,
            "vol_window": 24,
            "jitter_sigma": 0.20,
        })
    trainer = trainer_cls(trainer_cfg)
    trainer.train()
    audit_fit = build_audit_record(
        {**base_cfg, "trainer": trainer_cls.__name__,
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
    generator_cls = _GENERATORS[args.method]
    gen = generator_cls({
        **base_cfg,
        "load_model": model_path,
        "n_samples": n_syn,
        "output_file": syn_path,
        "start_timestamp": str(syn_timestamps[0]),
        "frequency": "4h",
    })
    gen.run_generate()
    audit_gen = build_audit_record(
        {**base_cfg, "generator": generator_cls.__name__,
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

    gates_pass = bool(alg.get("valid")) and bool(dist.get("gates", {}).get("all_pass")) and bool(mem.get("gates", {}).get("all_pass"))
    metrics_files = [
        os.path.join(method_dir, "algebraic_metrics.json"),
        os.path.join(method_dir, "distribution_metrics.json"),
        os.path.join(method_dir, "memorization_metrics.json"),
    ]
    append_ledger(
        base_cfg,
        kind="evaluate",
        audit=build_audit_record(
            {**base_cfg, "evaluator": "phase4_quality_gate_bundle", "valid": gates_pass},
            input_files={"synthetic_data": syn_path, "real_data": train_ohlcv_path},
        ),
        extra={
            "metrics_file": ";".join(metrics_files),
            "n_rows": len(syn_df),
            "valid": gates_pass,
            "notes": "quality gates passed" if gates_pass else "quality gates failed; diagnostic only",
        },
    )

    gate_summary = {
        "algebraic_violations": n_violations,
        "ks_returns_pvalue": dist.get("ks_returns", {}).get("pvalue"),
        "wasserstein_returns_ratio": dist.get("wasserstein_returns_ratio"),
        "drawdown_ks_pvalue": dist.get("drawdown_ks", {}).get("pvalue"),
        "duplicate_window_rate": mem.get("duplicate_window_rate"),
        "nn_overlap_rate": mem.get("nn_overlap_rate"),
        "copied_subseq_ratio": mem.get("copied_subseq_ratio"),
        "classifier_auc_window_std": mem.get("classifier_auc_window_std"),
    }
    if not gates_pass and not args.allow_diagnostic_output:
        stale_aug_path = os.path.join(method_dir, f"ethusdt_4h_tech_stat_augmented_{args.method}.csv")
        quarantined_aug_path = None
        if os.path.exists(stale_aug_path):
            quarantined_aug_path = stale_aug_path + ".invalid_quality_gates"
            os.replace(stale_aug_path, quarantined_aug_path)
        summary = {
            "method": args.method,
            "family_id": family_id,
            "n_real_train": int(len(train_real)),
            "n_synthetic": int(len(syn_df)),
            "augmented_csv": None,
            "quarantined_augmented_csv": quarantined_aug_path,
            "model_file": model_path,
            "synthetic_csv": syn_path,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "heldout_boundary": PROJECT3_HELDOUT_BOUNDARY,
            "project3_valid_for_training": False,
            "invalid_reason": "quality gates failed; rerun with --allow_diagnostic_output only for diagnostic artifacts",
            "gate_summary": gate_summary,
            "distribution_gates": dist.get("gates"),
            "memorization_gates": mem.get("gates"),
        }
        summary_path = os.path.join(method_dir, "augmentation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(json.dumps(summary, indent=2, default=str))
        print("      ABORT: quality gates failed; no Project 3 training CSV written.")
        sys.exit(3)

    # 6. Concatenate synthetic + pre-Stage-C real (synthetic first), recompute features
    print("[6/7] Concatenating + recomputing tech_stat features...")
    syn_ohlcv = syn_df[[DATETIME_COL] + OHLCV_COLS].copy()
    real_ohlcv = pre_stage_c_real[[DATETIME_COL] + OHLCV_COLS].copy()
    combined = pd.concat([syn_ohlcv, real_ohlcv], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)
    if (combined[DATETIME_COL] >= pd.Timestamp(PROJECT3_HELDOUT_BOUNDARY)).any():
        raise RuntimeError("internal error: augmented panel crosses Project 3 Stage C heldout boundary")
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
    family_descriptions = {
        "stationary_bootstrap_v1": "Politis-Romano stationary bootstrap on OHLCV primitives.",
        "regime_residual_bootstrap_v1": "Regime-conditional residual block bootstrap with Gaussian jitter on OHLCV primitives.",
    }
    family_assumptions = {
        "stationary_bootstrap_v1": [
            "log-returns are weakly stationary on the fit window",
            "block geometric distribution with mean=block_length_mean=24",
            "validity-by-construction reconstruction of OHLCV from primitives",
        ],
        "regime_residual_bootstrap_v1": [
            "volatility regimes are quantile-bin labels of rolling |r_close|",
            "primitives within a regime have approximately stationary residuals",
            "regime sequence is a Markov chain (Laplace-smoothed transitions)",
            "additive Gaussian jitter on residuals breaks exact-window memorization",
            "validity-by-construction reconstruction of OHLCV from primitives",
        ],
    }
    register_family(
        family_id,
        description=family_descriptions.get(family_id, args.method),
        assumptions=family_assumptions.get(family_id, []),
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
        "heldout_boundary": PROJECT3_HELDOUT_BOUNDARY,
        "pre_stage_c_real_end": str(pre_stage_c_real[DATETIME_COL].iloc[-1]),
        "project3_valid_for_training": gates_pass,
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
