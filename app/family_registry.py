"""Generator-family registry and diagnostics-only register.

Phase 4 §8 refinement #2 (registry) and #6 (diagnostics-only exit ramp).

The registry is a JSON file keyed by ``generator_family_id`` that stores
the family's mathematical assumptions, the windows it was fit on, the
§4.2 gate values it most recently produced, and the cells (Stage 4.3)
it has been authorized for.

The diagnostics-only register is a plain text file (one ID per line)
listing families that failed §4.2 and are therefore permanently barred
from contributing to a real-deployment training mix.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


_REGISTRY_DEFAULT = os.path.join(
    "experiments", "synthetic_data", "generator_family_registry.json"
)
_DIAGNOSTICS_DEFAULT = os.path.join(
    "experiments", "synthetic_data", "synthetic_diagnostics_only.txt"
)


def _read_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"families": {}}
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {"families": {}}
    data.setdefault("families", {})
    return data


def register_family(
    family_id: str,
    *,
    description: str,
    assumptions: List[str],
    fit_windows: Optional[List[Dict[str, Any]]] = None,
    gate_values: Optional[Dict[str, Any]] = None,
    authorized_cells: Optional[List[str]] = None,
    config_hash: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> str:
    """Insert / update a generator-family entry. Returns the registry path."""
    path = registry_path or _REGISTRY_DEFAULT
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = _read_registry(path)
    entry = data["families"].get(family_id, {})
    entry.update({
        "family_id": family_id,
        "description": description,
        "assumptions": list(assumptions),
        "fit_windows": list(fit_windows or entry.get("fit_windows", [])),
        "gate_values": dict(gate_values or entry.get("gate_values", {})),
        "authorized_cells": list(authorized_cells or entry.get("authorized_cells", [])),
        "config_hash": config_hash or entry.get("config_hash"),
    })
    data["families"][family_id] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def lookup_family(family_id: str, *, registry_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = registry_path or _REGISTRY_DEFAULT
    if not os.path.exists(path):
        return None
    return _read_registry(path)["families"].get(family_id)


def mark_diagnostics_only(
    family_id: str,
    *,
    reason: str,
    diagnostics_path: Optional[str] = None,
) -> str:
    """Append ``family_id`` (with reason) to the diagnostics-only register."""
    path = diagnostics_path or _DIAGNOSTICS_DEFAULT
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    existing = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    existing.add(s.split("\t", 1)[0])
    if family_id in existing:
        return path
    with open(path, "a") as f:
        if not existing:
            f.write("# Phase 4 §8 refinement #6: families barred from real-deployment training mixes.\n")
            f.write("# Format: <family_id>\\t<reason>\n")
        f.write(f"{family_id}\t{reason}\n")
    return path


def is_diagnostics_only(family_id: str, *, diagnostics_path: Optional[str] = None) -> bool:
    path = diagnostics_path or _DIAGNOSTICS_DEFAULT
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.split("\t", 1)[0] == family_id:
                return True
    return False


def write_protocol_stub(
    family_id: str,
    *,
    description: str,
    assumptions: List[str],
    gate_values: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Phase 4 §8 refinement #12: emit a ``synthetic_data_protocol.md`` stub."""
    path = output_path or os.path.join(
        "experiments", "synthetic_data", family_id, "synthetic_data_protocol.md"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Synthetic Data Protocol — `{family_id}`")
    lines.append("")
    lines.append(f"_{description}_")
    lines.append("")
    lines.append("## Mathematical assumptions")
    for a in assumptions:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("## §4.2 gate values (most recent run)")
    if gate_values:
        for k, v in gate_values.items():
            lines.append(f"- `{k}` = {v}")
    else:
        lines.append("- _not yet evaluated_")
    lines.append("")
    lines.append("## Authorized Stage 4.3 cells")
    lines.append("- _to be filled in once Stage 4.3 promotion succeeds_")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


__all__ = [
    "register_family",
    "lookup_family",
    "mark_diagnostics_only",
    "is_diagnostics_only",
    "write_protocol_stub",
]
