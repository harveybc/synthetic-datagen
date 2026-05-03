"""Tests for the generator-family registry, diagnostics-only register,
and protocol-doc stub generator (Phase 4 §8 refinements #2, #6, #12)."""
from __future__ import annotations

import json
import os

from app.family_registry import (
    is_diagnostics_only,
    lookup_family,
    mark_diagnostics_only,
    register_family,
    write_protocol_stub,
)


def test_register_family_round_trip(tmp_path):
    reg = tmp_path / "registry.json"
    p = register_family(
        "stationary_bootstrap_v1",
        description="Politis-Romano bootstrap.",
        assumptions=["weak stationarity"],
        gate_values={"ks_pvalue": 0.5},
        registry_path=str(reg),
    )
    assert p == str(reg)
    data = json.loads(reg.read_text())
    assert "stationary_bootstrap_v1" in data["families"]
    entry = lookup_family("stationary_bootstrap_v1", registry_path=str(reg))
    assert entry["description"].startswith("Politis-Romano")
    assert entry["gate_values"] == {"ks_pvalue": 0.5}


def test_register_family_updates_existing_entry(tmp_path):
    reg = tmp_path / "registry.json"
    register_family(
        "fam_a", description="v1", assumptions=["A"],
        gate_values={"x": 1}, registry_path=str(reg),
    )
    register_family(
        "fam_a", description="v2", assumptions=["A", "B"],
        gate_values={"x": 2}, registry_path=str(reg),
    )
    entry = lookup_family("fam_a", registry_path=str(reg))
    assert entry["description"] == "v2"
    assert entry["assumptions"] == ["A", "B"]
    assert entry["gate_values"] == {"x": 2}


def test_diagnostics_only_register_idempotent(tmp_path):
    diag = tmp_path / "diagnostics.txt"
    mark_diagnostics_only("fam_b", reason="failed copied-subseq gate",
                          diagnostics_path=str(diag))
    mark_diagnostics_only("fam_b", reason="failed copied-subseq gate",
                          diagnostics_path=str(diag))
    body = diag.read_text()
    # one entry only despite two writes
    assert body.count("fam_b\t") == 1
    assert is_diagnostics_only("fam_b", diagnostics_path=str(diag))
    assert not is_diagnostics_only("fam_other", diagnostics_path=str(diag))


def test_protocol_stub_contains_required_sections(tmp_path):
    out = tmp_path / "protocol.md"
    p = write_protocol_stub(
        "fam_c",
        description="A test family.",
        assumptions=["assumption-1", "assumption-2"],
        gate_values={"ks_pvalue": 0.42},
        output_path=str(out),
    )
    assert p == str(out)
    body = out.read_text()
    assert "fam_c" in body
    assert "Mathematical assumptions" in body
    assert "assumption-1" in body
    assert "ks_pvalue" in body
    assert "Authorized Stage 4.3 cells" in body
