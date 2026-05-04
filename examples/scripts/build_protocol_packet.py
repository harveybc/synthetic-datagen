#!/usr/bin/env python
"""Build the *locked* synthetic-augmentation protocol packet for
``regime_residual_bootstrap_v1`` with ``--anti_memorization`` enabled.

This script is **read-only** with respect to the synthetic generator
artifacts: it consumes the existing ``augmentation_summary.json``
emitted by :mod:`build_augmented_project3_training`, hashes the
input/output files, validates that all 7 Phase 4 §4.2 gates pass and
that no synthetic timestamp crosses the Project 3 Stage C heldout
boundary (2025-01-01 00:00:00), and emits two artifacts:

* ``regime_residual_bootstrap_v1_anti_mem_protocol.json`` — locked,
  versioned, machine-readable protocol packet.
* ``regime_residual_bootstrap_v1_anti_mem_protocol.md`` — human-readable
  rendering of the same packet.

It will refuse to emit either file if **any** gate fails or the
heldout boundary is violated. It does **not** launch agent-multi
training; it does not modify the synthetic data; and it does not
relax the Phase 4 gate thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd


HELDOUT_BOUNDARY = "2025-01-01 00:00:00"
DATETIME_COL = "DATE_TIME"

GATE_THRESHOLDS = {
    # Phase 4 §4.2.
    "algebraic_violations_max": 0,
    "ks_returns_pvalue_min": 0.01,
    "wasserstein_returns_ratio_max": 1.5,
    "classifier_auc_window_std_max": 0.70,
    "duplicate_window_rate_max": 1e-3,
    "nn_overlap_rate_max": 1e-3,
    "copied_subseq_ratio_max": 0.50,
}


# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_gates(summary: Dict[str, Any]) -> List[str]:
    """Return a list of gate-failure reasons; empty list = all pass."""
    gs = summary.get("gate_summary", {})
    reasons: List[str] = []
    if int(gs.get("algebraic_violations", 1)) > GATE_THRESHOLDS["algebraic_violations_max"]:
        reasons.append(f"algebraic_violations={gs.get('algebraic_violations')}>0")
    if float(gs.get("ks_returns_pvalue", 0.0)) < GATE_THRESHOLDS["ks_returns_pvalue_min"]:
        reasons.append(f"ks_returns_pvalue={gs.get('ks_returns_pvalue')}<0.01")
    if float(gs.get("wasserstein_returns_ratio", 1e9)) > GATE_THRESHOLDS["wasserstein_returns_ratio_max"]:
        reasons.append("wasserstein_returns_ratio>1.5")
    if float(gs.get("classifier_auc_window_std", 1.0)) > GATE_THRESHOLDS["classifier_auc_window_std_max"]:
        reasons.append("classifier_auc_window_std>0.70")
    if float(gs.get("duplicate_window_rate", 1.0)) > GATE_THRESHOLDS["duplicate_window_rate_max"]:
        reasons.append("duplicate_window_rate>1e-3")
    if float(gs.get("nn_overlap_rate", 1.0)) > GATE_THRESHOLDS["nn_overlap_rate_max"]:
        reasons.append("nn_overlap_rate>1e-3")
    if float(gs.get("copied_subseq_ratio", 1.0)) > GATE_THRESHOLDS["copied_subseq_ratio_max"]:
        reasons.append("copied_subseq_ratio>0.50")
    if not bool(summary.get("project3_valid_for_training", False)):
        reasons.append("project3_valid_for_training=false")
    return reasons


def _check_heldout(synthetic_csv: str) -> List[str]:
    df = pd.read_csv(synthetic_csv, usecols=[DATETIME_COL])
    ts = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    boundary = pd.Timestamp(HELDOUT_BOUNDARY)
    if ts.isna().any():
        return ["synthetic CSV contains unparseable DATE_TIME entries"]
    if (ts >= boundary).any():
        n_bad = int((ts >= boundary).sum())
        return [f"synthetic span crosses heldout boundary ({n_bad} rows >= {HELDOUT_BOUNDARY})"]
    return []


def _diagnostic_warnings(summary: Dict[str, Any]) -> List[str]:
    """Return non-blocking warnings for metrics outside the approved gate set."""
    gs = summary.get("gate_summary", {})
    warnings: List[str] = []
    drawdown_p = gs.get("drawdown_ks_pvalue")
    if drawdown_p is not None and float(drawdown_p) < 0.01:
        warnings.append(
            "drawdown_ks_pvalue is below 0.01; this is not an approved Phase 4 "
            "blocking gate, but Stage B approval must review drawdown-shape mismatch "
            "before any synthetic-pretraining launch."
        )
    return warnings


# ---------------------------------------------------------------------------
def build_protocol_packet(
    augmentation_summary_path: str,
    real_input_csv: str,
    *,
    family_id: str = "regime_residual_bootstrap_v1",
    family_revision: str = "anti_mem_v1",
    seed: int = 13,
    anti_mem_params: Dict[str, Any] | None = None,
    train_years: int = 5,
    val_years: int = 1,
    test_years: int = 1,
) -> Dict[str, Any]:
    """Construct the locked protocol packet dict.

    Raises ``RuntimeError`` if any gate fails or the heldout boundary
    is violated.
    """
    if anti_mem_params is None:
        anti_mem_params = {
            "anti_memorization": True,
            "anti_mem_window": 32,
            "anti_mem_max_real_windows": 4000,
            "anti_mem_dup_eps_quantile": 0.001,
            "anti_mem_safety_margin": 1.50,
            "anti_mem_boost_factor": 1.0,
            "anti_mem_max_passes": 16,
        }

    with open(augmentation_summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    synthetic_csv = summary["synthetic_csv"]
    augmented_csv = summary["augmented_csv"]
    model_file = summary["model_file"]

    gate_reasons = _check_gates(summary)
    heldout_reasons = _check_heldout(synthetic_csv)
    failures = gate_reasons + heldout_reasons
    if failures:
        raise RuntimeError(
            "protocol packet REFUSED — invariants violated: "
            + "; ".join(failures)
        )
    diagnostic_warnings = _diagnostic_warnings(summary)

    packet: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "packet_kind": "synthetic_augmentation_protocol",
        "asset": "ETHUSDT_4h",
        "generator": {
            "family_id": family_id,
            "family_revision": family_revision,
            "trainer_plugin": "regime_residual_bootstrap_ohlcv_trainer",
            "generator_plugin": "regime_residual_bootstrap_ohlcv_generator",
            "seed": int(seed),
            "anti_memorization_params": dict(anti_mem_params),
        },
        "windows": {
            "train_start": summary["train_start"],
            "train_end": summary["train_end"],
            "synthetic_n_rows": int(summary["n_synthetic"]),
            "real_train_n_rows": int(summary["n_real_train"]),
            "heldout_boundary": HELDOUT_BOUNDARY,
            "pre_stage_c_real_end": summary.get("pre_stage_c_real_end"),
            "downstream_split_template": {
                "train_years": int(train_years),
                "val_years": int(val_years),
                "test_years": int(test_years),
                "split_anchor": "start",
                "comment": "val + test windows must remain entirely real-data",
            },
        },
        "gate_thresholds": GATE_THRESHOLDS,
        "gate_table": summary["gate_summary"],
        "diagnostic_warnings": diagnostic_warnings,
        "distribution_gates": summary["distribution_gates"],
        "memorization_gates": summary["memorization_gates"],
        "project3_valid_for_training": True,
        "input_files": {
            "real_input_csv": {
                "path": real_input_csv,
                "sha256": _sha256(real_input_csv),
            },
        },
        "output_files": {
            "synthetic_ohlcv_csv": {
                "path": synthetic_csv,
                "sha256": _sha256(synthetic_csv),
            },
            "augmented_tech_stat_csv": {
                "path": augmented_csv,
                "sha256": _sha256(augmented_csv),
            },
            "trainer_npz": {
                "path": model_file,
                "sha256": _sha256(model_file),
            },
        },
        "build_metadata": {
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "augmentation_summary_path": augmentation_summary_path,
            "augmentation_summary_sha256": _sha256(augmentation_summary_path),
            "phase4_section": "§4.1, §4.2, §4.3 (pre-promotion)",
        },
        "stage_b_status": "PENDING_APPROVAL",
        "downstream_training": {
            "agent_multi_config_template":
                "examples/config/project3_ethusdt_4h_sac_synth_anti_mem_v1.json",
            "do_not_run_until": "Stage B approval recorded in SYNTHETIC_LEDGER.csv",
            "compare_plan": "experiments/synthetic_data/project3_eth_4h/COMPARE_PLAN.md",
        },
    }
    return packet


# ---------------------------------------------------------------------------
def render_md(packet: Dict[str, Any]) -> str:
    g = packet["gate_table"]
    am = packet["generator"]["anti_memorization_params"]
    w = packet["windows"]
    out = packet["output_files"]
    inp = packet["input_files"]
    lines = [
        "# Synthetic Augmentation Protocol — `regime_residual_bootstrap_v1` (anti_mem_v1)",
        "",
        "**LOCKED PACKET** — Phase 4 §4.1/§4.2/§4.3. Do not edit by hand;",
        "regenerate via `examples/scripts/build_protocol_packet.py`.",
        "",
        f"- Asset: **{packet['asset']}**",
        f"- Schema version: `{packet['schema_version']}`",
        f"- Built (UTC): `{packet['build_metadata']['built_at_utc']}`",
        f"- Stage B status: **{packet['stage_b_status']}**",
        "",
        "## Generator",
        f"- family_id: `{packet['generator']['family_id']}`",
        f"- family_revision: `{packet['generator']['family_revision']}`",
        f"- trainer plugin: `{packet['generator']['trainer_plugin']}`",
        f"- generator plugin: `{packet['generator']['generator_plugin']}`",
        f"- seed: `{packet['generator']['seed']}`",
        "",
        "### Anti-memorization params",
    ]
    for k, v in am.items():
        lines.append(f"- `{k}` = `{v}`")
    lines += [
        "",
        "## Windows",
        f"- train_start: `{w['train_start']}`",
        f"- train_end: `{w['train_end']}`",
        f"- real_train_n_rows: `{w['real_train_n_rows']}`",
        f"- synthetic_n_rows: `{w['synthetic_n_rows']}`",
        f"- heldout_boundary: **`{w['heldout_boundary']}`** (Stage C firewall)",
        f"- pre_stage_c_real_end: `{w['pre_stage_c_real_end']}`",
        "",
        "### Downstream split template",
        f"- train_years: `{w['downstream_split_template']['train_years']}`",
        f"- val_years: `{w['downstream_split_template']['val_years']}`",
        f"- test_years: `{w['downstream_split_template']['test_years']}`",
        f"- split_anchor: `{w['downstream_split_template']['split_anchor']}`",
        f"- note: {w['downstream_split_template']['comment']}",
        "",
        "## Gate table (Phase 4 §4.2)",
        "",
        "| Gate | Threshold | Value | Pass |",
        "|---|---|---|---|",
        f"| algebraic_violations | == 0 | {g['algebraic_violations']} | ✅ |",
        f"| ks_returns_pvalue | > 0.01 | {g['ks_returns_pvalue']:.6g} | ✅ |",
        f"| wasserstein_returns_ratio | < 1.5 | {g['wasserstein_returns_ratio']:.6g} | ✅ |",
        f"| classifier_auc_window_std | < 0.70 | {g['classifier_auc_window_std']:.6g} | ✅ |",
        f"| nn_overlap_rate | < 1e-3 | {g['nn_overlap_rate']:.6g} | ✅ |",
        f"| copied_subseq_ratio | < 0.50 | {g['copied_subseq_ratio']:.6g} | ✅ |",
        f"| duplicate_window_rate | < 1e-3 | {g['duplicate_window_rate']:.6g} | ✅ |",
        "",
        f"`project3_valid_for_training` = **{packet['project3_valid_for_training']}**",
        "",
        "## Diagnostic warnings",
        "",
    ]
    warnings = packet.get("diagnostic_warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Input files",
    ]
    for k, v in inp.items():
        lines.append(f"- `{k}` — `{v['path']}`")
        lines.append(f"  - sha256: `{v['sha256']}`")
    lines += ["", "## Output files"]
    for k, v in out.items():
        lines.append(f"- `{k}` — `{v['path']}`")
        lines.append(f"  - sha256: `{v['sha256']}`")
    lines += [
        "",
        "## Downstream training",
        f"- agent-multi config template: `{packet['downstream_training']['agent_multi_config_template']}`",
        f"- do_not_run_until: **{packet['downstream_training']['do_not_run_until']}**",
        f"- compare plan: `{packet['downstream_training']['compare_plan']}`",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--augmentation_summary",
        default="experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/augmentation_summary.json",
    )
    ap.add_argument(
        "--real_input_csv",
        default="examples/data/ethusdt_4h_full_8yr.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap",
    )
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    packet = build_protocol_packet(
        augmentation_summary_path=args.augmentation_summary,
        real_input_csv=args.real_input_csv,
        seed=args.seed,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "regime_residual_bootstrap_v1_anti_mem_protocol.json")
    md_path = os.path.join(args.out_dir, "regime_residual_bootstrap_v1_anti_mem_protocol.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(packet, fh, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_md(packet))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(
        {"project3_valid_for_training": packet["project3_valid_for_training"]},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
