"""Audit metadata helpers.

Builds a reproducibility record (config hash, input hash, plugin names,
seed, git commit if available) for every synthetic-data run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional


def file_sha256(path: str, max_bytes: Optional[int] = None) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def config_hash(config: Dict[str, Any]) -> str:
    """Stable, secret-free hash of the effective config."""
    safe = {k: v for k, v in config.items() if k not in ("username", "password")}
    blob = json.dumps(safe, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def build_audit_record(
    config: Dict[str, Any],
    *,
    input_files: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "config_hash": config_hash(config),
        "git_commit": git_commit_short(),
        "seed": config.get("seed"),
        "trainer": config.get("trainer"),
        "generator": config.get("generator"),
        "evaluator": config.get("evaluator"),
        "optimizer": config.get("optimizer"),
        "train_start": config.get("train_start"),
        "train_end": config.get("train_end"),
        "heldout_boundary": config.get("heldout_boundary"),
        "project3_mode": config.get("project3_mode", False),
        # Phase 4 §3 fields:
        "generator_family_id": config.get("generator_family_id"),
        "synthetic_ablation_id": config.get("synthetic_ablation_id"),
        "synthetic_use_case": config.get("synthetic_use_case"),
        "augmentation_ratios": config.get("augmentation_ratios"),
        "asset_id": config.get("asset_id"),
        "timeframe": config.get("timeframe"),
    }
    if input_files:
        record["input_files"] = {
            label: {"path": path, "sha256": file_sha256(path)}
            for label, path in input_files.items()
        }
    if extra:
        record.update(extra)
    return record


__all__ = [
    "file_sha256", "config_hash", "git_commit_short", "build_audit_record",
]
