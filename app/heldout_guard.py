"""Project 3 heldout-boundary guard.

Centralizes the rule "the synthetic generator must never see rows on or
after the configured heldout boundary".  Importable from CLI dispatch
and from individual trainer plugins.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def assert_no_heldout_violation(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    datetime_column: str = "DATE_TIME",
) -> None:
    """Raise ``ValueError`` if the dataframe contains rows on/after heldout.

    Behaviour:

    * ``project3_mode=True``  -> heldout_boundary is *required*; any row
      ``>=`` boundary is fatal; the ``allow_non_research_mode`` escape
      hatch is rejected.
    * ``project3_mode=False`` -> if a heldout_boundary is set and rows
      cross it, raise unless ``allow_non_research_mode=True`` (explicit
      opt-in for synthetic-only research workflows).
    """
    project3 = bool(config.get("project3_mode", False))
    boundary: Optional[str] = config.get("heldout_boundary")
    allow_escape = bool(config.get("allow_non_research_mode", False))
    enforce = bool(config.get("reject_if_input_crosses_heldout", True))

    if not enforce:
        return
    if project3 and boundary is None:
        raise ValueError(
            "project3_mode=True requires heldout_boundary to be set."
        )
    if boundary is None:
        return
    if datetime_column not in df.columns:
        return
    ts = pd.to_datetime(df[datetime_column])
    bts = pd.Timestamp(boundary)
    n_violations = int((ts >= bts).sum())
    if n_violations == 0:
        return
    if project3:
        raise ValueError(
            f"[project3 leakage guard] {n_violations} input rows cross "
            f"heldout_boundary={bts}; refusing to train the generator."
        )
    if allow_escape:
        return
    raise ValueError(
        f"{n_violations} input rows cross heldout_boundary={bts}; "
        "set allow_non_research_mode=True to bypass (NEVER for Project 3)."
    )


__all__ = ["assert_no_heldout_violation"]
