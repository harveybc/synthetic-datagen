"""Synthetic-data ledger.

Append-only CSV ledger that mirrors the Phase 3 immutable run ledger
for every generator-fit and every generation run.  Primary key is
``(generator_family_id, config_hash, seed, kind)``.

The ledger is written to ``experiments/synthetic_data/SYNTHETIC_LEDGER.csv``
by default; the location can be overridden via the
``synthetic_ledger_path`` config key or the ``SDG_SYNTHETIC_LEDGER`` env
var.  Writes are best-effort and never raise into the caller.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import Any, Dict, Optional


_DEFAULT_PATH = os.path.join(
    "experiments", "synthetic_data", "SYNTHETIC_LEDGER.csv"
)

_FIELDS = (
    "timestamp_utc",
    "kind",                  # "fit" | "generate" | "evaluate"
    "config_hash",
    "git_commit",
    "asset_id",
    "timeframe",
    "generator_family_id",
    "synthetic_ablation_id",
    "trainer",
    "generator",
    "evaluator",
    "seed",
    "train_start",
    "train_end",
    "heldout_boundary",
    "project3_mode",
    "synthetic_use_case",
    "augmentation_ratios",
    "model_file",
    "output_file",
    "metrics_file",
    "n_rows",
    "valid",
    "notes",
)


def _resolve_path(config: Dict[str, Any]) -> str:
    return (
        config.get("synthetic_ledger_path")
        or os.environ.get("SDG_SYNTHETIC_LEDGER")
        or _DEFAULT_PATH
    )


def append_ledger(
    config: Dict[str, Any],
    *,
    kind: str,
    audit: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Append one row to the synthetic ledger.  Returns the ledger path."""
    try:
        path = _resolve_path(config)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        is_new = not os.path.exists(path)
        row = {f: "" for f in _FIELDS}
        row["timestamp_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        row["kind"] = kind
        for k in (
            "config_hash", "git_commit", "asset_id", "timeframe",
            "generator_family_id", "synthetic_ablation_id",
            "trainer", "generator", "evaluator", "seed",
            "train_start", "train_end", "heldout_boundary",
            "project3_mode", "synthetic_use_case", "augmentation_ratios",
        ):
            v = audit.get(k)
            if v is not None:
                row[k] = str(v)
        if extra:
            for k, v in extra.items():
                if k in row:
                    row[k] = "" if v is None else str(v)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            if is_new:
                w.writeheader()
            w.writerow(row)
        return path
    except Exception:
        return None


__all__ = ["append_ledger"]
