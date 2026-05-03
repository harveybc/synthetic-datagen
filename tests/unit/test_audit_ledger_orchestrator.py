"""Tests for audit record extension, synthetic ledger, forbidden-paths,
and the Stage 4.3 augmentation manifest pipeline."""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from app.audit import build_audit_record
from app.forbidden_paths import assert_path_allowed, load_forbidden_globs
from app.synthetic_ledger import append_ledger
from sdg_plugins.pipeline.augmentation_manifest_pipeline import (
    AugmentationManifestPipeline,
)


# --- audit ------------------------------------------------------------------


def test_audit_record_includes_phase4_fields():
    cfg = {
        "seed": 7,
        "trainer": "stationary_bootstrap_ohlcv_trainer",
        "generator": "stationary_bootstrap_ohlcv_generator",
        "generator_family_id": "stationary_bootstrap_v1",
        "synthetic_ablation_id": "ratio_1_00x",
        "synthetic_use_case": "rl_pretrain",
        "augmentation_ratios": [0.25, 0.5, 1.0],
        "asset_id": "EURUSD",
        "timeframe": "1h",
        "username": "alice",
        "password": "should-not-be-here",
    }
    rec = build_audit_record(cfg)
    assert rec["generator_family_id"] == "stationary_bootstrap_v1"
    assert rec["synthetic_ablation_id"] == "ratio_1_00x"
    assert rec["synthetic_use_case"] == "rl_pretrain"
    assert rec["augmentation_ratios"] == [0.25, 0.5, 1.0]
    assert rec["asset_id"] == "EURUSD" and rec["timeframe"] == "1h"
    assert "username" not in rec and "password" not in rec


# --- ledger -----------------------------------------------------------------


def test_append_ledger_creates_csv_with_stable_columns(tmp_path):
    ledger = tmp_path / "SYNTHETIC_LEDGER.csv"
    cfg = {"synthetic_ledger_path": str(ledger)}
    audit = {
        "config_hash": "abc123",
        "git_commit": "deadbee",
        "asset_id": "EURUSD",
        "timeframe": "1h",
        "generator_family_id": "stationary_bootstrap_v1",
        "trainer": "tr",
        "generator": "gn",
        "seed": 0,
    }
    p1 = append_ledger(cfg, kind="fit", audit=audit, extra={"model_file": "m.npz"})
    p2 = append_ledger(cfg, kind="generate", audit=audit, extra={"output_file": "o.csv"})
    assert p1 == str(ledger) and p2 == str(ledger)
    df = pd.read_csv(ledger)
    assert len(df) == 2
    assert {"timestamp_utc", "kind", "config_hash", "generator_family_id",
            "model_file", "output_file"}.issubset(df.columns)
    assert df["kind"].tolist() == ["fit", "generate"]


# --- forbidden paths --------------------------------------------------------


def test_forbidden_paths_blocks_2025_heldout():
    with pytest.raises(ValueError):
        assert_path_allowed("data/heldout/2025/eurusd_1h.csv")
    # Non-forbidden paths must pass through.
    assert_path_allowed("data/train/2010_2024/eurusd_1h.csv")


def test_load_forbidden_globs_includes_repo_default():
    globs = load_forbidden_globs()
    assert any("2025" in g for g in globs)


# --- augmentation manifest pipeline -----------------------------------------


def test_augmentation_pipeline_emits_six_cells_x_three_costs(tmp_path):
    n = 200
    real = pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=n, freq="h"),
        "CLOSE": 100.0 + 0.01 * pd.Series(range(n)),
    })
    syn = real.copy()
    syn["CLOSE"] = syn["CLOSE"] + 0.5
    rp = tmp_path / "real.csv"; sp = tmp_path / "syn.csv"
    real.to_csv(rp, index=False); syn.to_csv(sp, index=False)
    out = tmp_path / "augmentation"

    pipe = AugmentationManifestPipeline({
        "real_train": str(rp),
        "synthetic_data": str(sp),
        "real_validation": str(rp),
        "output_dir": str(out),
        "candidate_run_id": "cand_001",
        "generator_family_id": "stationary_bootstrap_v1",
        "generator_config_hash": "abc",
        "asset_id": "EURUSD",
        "timeframe": "1h",
        "seeds": [0, 1, 2, 3, 4],
        "cost_scenarios": ["base", "plus_50pct", "plus_100pct"],
    })
    res = pipe.generate_manifests()
    assert res["n_cells"] == 6
    assert res["n_manifests"] == 6 * 3

    idx = json.loads((out / "ablation_index.json").read_text())
    assert idx["n_cells"] == 6 and idx["n_cost_scenarios"] == 3
    cell_ids = {row["cell_id"] for row in idx["manifests"]}
    assert cell_ids == {
        "real_train_only", "synthetic_train_only",
        "real_plus_synthetic_0_25x", "real_plus_synthetic_0_50x",
        "real_plus_synthetic_1_00x",
        "synthetic_pretrain_then_real_finetune",
    }
    sample = json.loads((out / "real_plus_synthetic_0_50x" / "manifest_base.json").read_text())
    assert sample["synthetic_ratio"] == 0.5
    assert sample["downstream_repo"] == "agent-multi"
    assert sample["seeds"] == [0, 1, 2, 3, 4]
    assert sample["training_data_sha256"]
