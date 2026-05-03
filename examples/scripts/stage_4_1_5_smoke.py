"""Stage 4.1.5 smoke run.

Runs the minimal Phase 4 pipeline end-to-end on a tiny window
(<= 200 bars):

  1. Train ``stationary_bootstrap_ohlcv_trainer`` on a small CSV.
  2. Generate ``stationary_bootstrap_ohlcv_generator`` synthetic CSV.
  3. Run ``OhlcvAlgebraicEvaluator`` — must return zero violations.
  4. Run ``FinancialDistributionEvaluator`` and ``MemorizationEvaluator``.
  5. Append rows to ``SYNTHETIC_LEDGER.csv`` for fit / generate / evaluate.
  6. Register the family in ``generator_family_registry.json`` and emit
     a ``synthetic_data_protocol.md`` stub.

Exit code 0 only if all algebraic violations == 0.

Usage:
    python -m examples.scripts.stage_4_1_5_smoke
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

from app.audit import build_audit_record
from app.family_registry import register_family, write_protocol_stub
from app.synthetic_ledger import append_ledger
from sdg_plugins.evaluator.financial_distribution_evaluator import (
    FinancialDistributionEvaluator,
)
from sdg_plugins.evaluator.memorization_evaluator import MemorizationEvaluator
from sdg_plugins.evaluator.ohlcv_algebraic_evaluator import OhlcvAlgebraicEvaluator
from sdg_plugins.generator.stationary_bootstrap_ohlcv_generator import (
    StationaryBootstrapOhlcvGenerator,
)
from sdg_plugins.trainer.stationary_bootstrap_ohlcv_trainer import (
    StationaryBootstrapOhlcvTrainer,
)


SMOKE_DIR = os.path.join("experiments", "synthetic_data", "_smoke_4_1_5")
FAMILY_ID = "stationary_bootstrap_v1"


def _ensure_input(path: str) -> str:
    """Use the bundled fixture; truncate to 200 rows."""
    src = os.path.join("examples", "data", "financial_ohlcv_sample.csv")
    df = pd.read_csv(src).head(200)
    df.to_csv(path, index=False)
    return path


def main() -> int:
    os.makedirs(SMOKE_DIR, exist_ok=True)
    train_csv = os.path.join(SMOKE_DIR, "train.csv")
    model_npz = os.path.join(SMOKE_DIR, "bootstrap.npz")
    syn_csv = os.path.join(SMOKE_DIR, "synthetic.csv")
    metrics_alg = os.path.join(SMOKE_DIR, "algebraic_metrics.json")
    metrics_dist = os.path.join(SMOKE_DIR, "distribution_metrics.json")
    metrics_mem = os.path.join(SMOKE_DIR, "memorization_metrics.json")
    ledger = os.path.join(SMOKE_DIR, "SYNTHETIC_LEDGER.csv")

    _ensure_input(train_csv)

    base_cfg = {
        "financial_mode": True,
        "asset_id": "EURUSD",
        "timeframe": "1h",
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN", "high_col": "HIGH", "low_col": "LOW",
        "close_col": "CLOSE", "volume_col": "VOLUME",
        "seed": 42,
        "generator_family_id": FAMILY_ID,
        "synthetic_ablation_id": "smoke",
        "synthetic_use_case": "diagnostics",
        "synthetic_ledger_path": ledger,
    }

    # 1. train
    trainer = StationaryBootstrapOhlcvTrainer({
        **base_cfg,
        "train_data": train_csv,
        "save_model": model_npz,
        "block_length_mean": 24,
    })
    trainer.train()
    audit_fit = build_audit_record(
        {**base_cfg, "trainer": "stationary_bootstrap_ohlcv_trainer", "train_data": train_csv,
         "save_model": model_npz},
        input_files={"train_data": train_csv},
    )
    append_ledger({**base_cfg}, kind="fit", audit=audit_fit,
                  extra={"model_file": model_npz})

    # 2. generate
    gen = StationaryBootstrapOhlcvGenerator({
        **base_cfg,
        "load_model": model_npz,
        "n_samples": 200,
        "output_file": syn_csv,
        "start_timestamp": "2024-01-01 00:00:00",
        "frequency": "1h",
    })
    gen.run_generate()
    audit_gen = build_audit_record(
        {**base_cfg, "generator": "stationary_bootstrap_ohlcv_generator",
         "load_model": model_npz, "output_file": syn_csv},
        input_files={"model": model_npz},
    )
    append_ledger({**base_cfg}, kind="generate", audit=audit_gen,
                  extra={"output_file": syn_csv})

    # 3. algebraic
    alg = OhlcvAlgebraicEvaluator({
        **base_cfg,
        "synthetic_data": syn_csv,
        "metrics_file": metrics_alg,
    })
    alg_rep = alg.evaluate()
    total_violations = sum(int(v) for v in alg_rep.get("violations", {}).values())
    print("[smoke] algebraic violations:", total_violations, "valid:", alg_rep.get("valid"))

    # 4a. distribution
    dist = FinancialDistributionEvaluator({
        **base_cfg,
        "synthetic_data": syn_csv,
        "real_data": train_csv,
        "metrics_file": metrics_dist,
    })
    dist_rep = dist.evaluate()

    # 4b. memorization
    mem = MemorizationEvaluator({
        **base_cfg,
        "synthetic_data": syn_csv,
        "real_data": train_csv,
        "metrics_file": metrics_mem,
        "window": 16,
        "max_windows": 100,
    })
    mem_rep = mem.evaluate()

    # 5. ledger row for evaluate
    audit_eval = build_audit_record(
        {**base_cfg, "evaluator": "ohlcv_algebraic_evaluator",
         "synthetic_data": syn_csv},
        input_files={"synthetic": syn_csv, "real": train_csv},
    )
    append_ledger({**base_cfg}, kind="evaluate", audit=audit_eval,
                  extra={"metrics_file": metrics_alg,
                         "valid": bool(alg_rep.get("valid", False))})

    # 6. registry + protocol stub
    gate_summary = {
        "total_algebraic_violations": total_violations,
        "ks_returns_pvalue": dist_rep.get("ks_returns", {}).get("pvalue"),
        "wasserstein_returns_ratio": dist_rep.get("wasserstein_returns_ratio"),
        "duplicate_window_rate": mem_rep.get("duplicate_window_rate"),
        "copied_subseq_ratio": mem_rep.get("copied_subseq_ratio"),
        "classifier_auc_window_std": mem_rep.get("classifier_auc_window_std"),
    }
    reg_path = register_family(
        FAMILY_ID,
        description="Politis-Romano stationary bootstrap on transformed OHLCV primitives.",
        assumptions=[
            "log-returns are weakly stationary on the fit window",
            "block geometric distribution with mean = block_length_mean",
            "primitives r_close, r_open, d_high, d_low, v are jointly resampled",
        ],
        fit_windows=[{"path": train_csv, "n_rows": 200}],
        gate_values=gate_summary,
        config_hash=audit_fit.get("config_hash"),
        registry_path=os.path.join(SMOKE_DIR, "generator_family_registry.json"),
    )
    stub_path = write_protocol_stub(
        FAMILY_ID,
        description="Stage 4.1 baseline; non-parametric, validity-by-construction reconstruction.",
        assumptions=[
            "weak stationarity on fit window",
            "no regime shift between fit and generate windows",
        ],
        gate_values=gate_summary,
        output_path=os.path.join(SMOKE_DIR, f"{FAMILY_ID}_protocol.md"),
    )

    summary = {
        "smoke_dir": SMOKE_DIR,
        "ledger": ledger,
        "registry": reg_path,
        "protocol_stub": stub_path,
        "algebraic_violations": total_violations,
        "distribution_gates": dist_rep.get("gates"),
        "memorization_gates": mem_rep.get("gates"),
    }
    summary_path = os.path.join(SMOKE_DIR, "smoke_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

    return 0 if total_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
