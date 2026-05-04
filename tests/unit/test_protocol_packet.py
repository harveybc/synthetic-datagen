"""Tests for the Phase 4 synthetic-augmentation protocol packet builder.

Invariants under test:
* Builder REFUSES to emit a packet if any §4.2 gate fails.
* Builder REFUSES to emit a packet if the synthetic CSV crosses the
  Project 3 Stage C heldout boundary (2025-01-01 00:00:00).
* On a valid input, the packet contains all required fields, file
  hashes are computed, and ``project3_valid_for_training`` is True.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "scripts"))

import build_protocol_packet as bpp  # noqa: E402


# ---------------------------------------------------------------------------
def _valid_summary(synthetic_csv: str, augmented_csv: str, model_file: str) -> dict:
    return {
        "method": "regime_residual_bootstrap",
        "family_id": "regime_residual_bootstrap_v1",
        "n_real_train": 8749,
        "n_synthetic": 2190,
        "augmented_csv": augmented_csv,
        "model_file": model_file,
        "synthetic_csv": synthetic_csv,
        "train_start": "2017-09-28 04:00:00",
        "train_end": "2021-09-28 00:00:00",
        "heldout_boundary": "2025-01-01 00:00:00",
        "pre_stage_c_real_end": "2024-12-31 20:00:00",
        "project3_valid_for_training": True,
        "gate_summary": {
            "algebraic_violations": 0,
            "ks_returns_pvalue": 0.32,
            "wasserstein_returns_ratio": 0.014,
            "drawdown_ks_pvalue": 1e-300,
            "duplicate_window_rate": 0.0,
            "nn_overlap_rate": 0.0,
            "copied_subseq_ratio": 0.0,
            "classifier_auc_window_std": 0.58,
        },
        "distribution_gates": {
            "ks_returns_pass": True,
            "wasserstein_pass": True,
            "all_pass": True,
        },
        "memorization_gates": {
            "classifier_auc_pass": True,
            "duplicate_rate_pass": True,
            "nn_overlap_pass": True,
            "copied_subseq_pass": True,
            "all_pass": True,
        },
    }


def _write_summary_and_artifacts(tmp_path, summary_overrides=None, syn_dt_end="2017-09-27 20:00:00"):
    syn = tmp_path / "synthetic.csv"
    aug = tmp_path / "augmented.csv"
    mdl = tmp_path / "bootstrap.npz"
    real = tmp_path / "real_input.csv"
    # Synthetic CSV: timestamps strictly before the heldout boundary.
    n = 50
    ts = pd.date_range(end=pd.Timestamp(syn_dt_end), periods=n, freq="4h")
    pd.DataFrame({
        "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "OPEN": 1.0, "HIGH": 1.0, "LOW": 1.0, "CLOSE": 1.0, "VOLUME": 1.0,
    }).to_csv(syn, index=False)
    aug.write_bytes(b"DATE_TIME,CLOSE\n2020-01-01 00:00:00,1.0\n")
    mdl.write_bytes(b"npz-stub")
    real.write_bytes(b"DATE_TIME,CLOSE\n2017-01-01 00:00:00,1.0\n")
    summary = _valid_summary(str(syn), str(aug), str(mdl))
    if summary_overrides:
        for k, v in summary_overrides.items():
            if isinstance(v, dict) and isinstance(summary.get(k), dict):
                summary[k] = {**summary[k], **v}
            else:
                summary[k] = v
    summary_path = tmp_path / "augmentation_summary.json"
    summary_path.write_text(json.dumps(summary))
    return str(summary_path), str(real), str(syn)


# ---------------------------------------------------------------------------
def test_protocol_builder_emits_locked_packet_on_valid_input(tmp_path):
    summary_path, real, _ = _write_summary_and_artifacts(tmp_path)
    packet = bpp.build_protocol_packet(
        augmentation_summary_path=summary_path,
        real_input_csv=real,
        seed=13,
    )
    assert packet["project3_valid_for_training"] is True
    assert packet["packet_kind"] == "synthetic_augmentation_protocol"
    assert packet["generator"]["family_id"] == "regime_residual_bootstrap_v1"
    assert packet["generator"]["family_revision"] == "anti_mem_v1"
    assert packet["generator"]["seed"] == 13
    assert packet["generator"]["anti_memorization_params"]["anti_memorization"] is True
    assert packet["windows"]["heldout_boundary"] == "2025-01-01 00:00:00"
    assert packet["stage_b_status"] == "PENDING_APPROVAL"
    assert packet["diagnostic_warnings"]
    assert "drawdown_ks_pvalue" in packet["diagnostic_warnings"][0]
    # All file hashes are 64-char hex.
    for blob in [packet["input_files"]["real_input_csv"]] + list(packet["output_files"].values()):
        assert len(blob["sha256"]) == 64 and all(c in "0123456789abcdef" for c in blob["sha256"])


@pytest.mark.parametrize("override", [
    {"gate_summary": {"duplicate_window_rate": 0.01}},
    {"gate_summary": {"ks_returns_pvalue": 1e-6}},
    {"gate_summary": {"classifier_auc_window_std": 0.95}},
    {"gate_summary": {"nn_overlap_rate": 0.5}},
    {"gate_summary": {"copied_subseq_ratio": 0.9}},
    {"gate_summary": {"algebraic_violations": 1}},
    {"gate_summary": {"wasserstein_returns_ratio": 5.0}},
    {"project3_valid_for_training": False},
])
def test_protocol_builder_refuses_when_any_gate_fails(tmp_path, override):
    summary_path, real, _ = _write_summary_and_artifacts(tmp_path, summary_overrides=override)
    with pytest.raises(RuntimeError, match="protocol packet REFUSED"):
        bpp.build_protocol_packet(
            augmentation_summary_path=summary_path,
            real_input_csv=real,
            seed=13,
        )


def test_protocol_builder_refuses_when_synthetic_crosses_heldout_boundary(tmp_path):
    # Place the last synthetic timestamp on/after 2025-01-01 to trip the firewall.
    summary_path, real, _ = _write_summary_and_artifacts(
        tmp_path, syn_dt_end="2025-06-01 00:00:00",
    )
    with pytest.raises(RuntimeError, match="heldout boundary"):
        bpp.build_protocol_packet(
            augmentation_summary_path=summary_path,
            real_input_csv=real,
            seed=13,
        )


def test_main_writes_json_and_md(tmp_path):
    summary_path, real, _ = _write_summary_and_artifacts(tmp_path)
    out_dir = tmp_path / "packet"
    rc = bpp.main([
        "--augmentation_summary", summary_path,
        "--real_input_csv", real,
        "--out_dir", str(out_dir),
        "--seed", "13",
    ])
    assert rc == 0
    json_path = out_dir / "regime_residual_bootstrap_v1_anti_mem_protocol.json"
    md_path = out_dir / "regime_residual_bootstrap_v1_anti_mem_protocol.md"
    assert json_path.exists() and md_path.exists()
    pkt = json.loads(json_path.read_text())
    assert pkt["project3_valid_for_training"] is True
    md = md_path.read_text()
    assert "PENDING_APPROVAL" in md
    assert "regime_residual_bootstrap_v1" in md
    assert "2025-01-01" in md
