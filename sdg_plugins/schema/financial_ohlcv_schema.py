"""Financial OHLCV column-classification schema utility.

Classifies the columns of an input dataframe into three buckets:

* ``primitive``     - the raw market fields the synthetic generator must
  emit (``OPEN``, ``HIGH``, ``LOW``, ``CLOSE``, ``VOLUME``).
* ``derived``       - deterministic functions of OHLCV (returns, ATR,
  RSI, MACD, Bollinger, OBV, VWAP, rolling moments, ...).  These MUST
  be recomputed by a causal feature engine after reconstruction; they
  are NEVER generated independently.
* ``contextual``    - timestamp, regime flags, calendar/seasonality
  columns; preserved or recomputed but not synthesized as primitives.

The classifier is intentionally rule-based and conservative: any column
that is not explicitly recognized falls into ``derived`` so that it is
guaranteed to be recomputed (never synthesized).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List


PRIMITIVE_COLUMNS_DEFAULT: tuple[str, ...] = (
    "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
)

CONTEXTUAL_PATTERNS: tuple[str, ...] = (
    r"^DATE_TIME$", r"^TIMESTAMP$", r"^DATE$", r"^TIME$",
    r"^HOUR$", r"^DAY_OF_WEEK$", r"^SESSION$", r"^MONTH$", r"^WEEK$",
    r"^REGIME$", r".*_REGIME$", r".*_FLAG$",
)

# Anything matching these is *certainly* derived/deterministic and must be
# recomputed.  This list is authoritative; unknown columns default to derived.
DERIVED_PATTERNS: tuple[str, ...] = (
    r"^TYPICAL_PRICE$", r"^RETURN.*", r"^LOG_RET.*", r"^.*_RETURN$",
    r"^SMA.*", r"^EMA.*", r"^MACD.*", r"^RSI.*", r"^STOCH.*",
    r"^WILLIAMS.*", r"^CCI.*", r"^ROC.*", r"^MOMENTUM.*",
    r"^BB_.*", r"^BOLL.*", r"^ATR.*", r"^NATR.*", r"^HIST_VOL.*",
    r"^OBV.*", r"^VOLUME_RATIO.*", r"^VWAP.*", r"^MFI.*",
    r"^ROLL.*", r"^REALIZED_VAR.*", r"^AUTOCORR.*", r"^HURST.*",
    r"^Z_.*", r".*_ZSCORE$",
)


@dataclass
class SchemaClassification:
    primitive: List[str] = field(default_factory=list)
    derived: List[str] = field(default_factory=list)
    contextual: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primitive": list(self.primitive),
            "derived": list(self.derived),
            "contextual": list(self.contextual),
            "unknown": list(self.unknown),
        }


def _matches_any(col: str, patterns: Iterable[str]) -> bool:
    up = col.upper()
    return any(re.match(p, up) for p in patterns)


def classify_columns(
    columns: Iterable[str],
    primitive_columns: Iterable[str] = PRIMITIVE_COLUMNS_DEFAULT,
    datetime_column: str = "DATE_TIME",
) -> SchemaClassification:
    """Classify every column name into primitive / derived / contextual.

    Unknown columns fall into ``derived`` so that the synthetic pipeline
    will always recompute them rather than synthesize them.
    """
    primitive_set = {c.upper() for c in primitive_columns}
    out = SchemaClassification()
    for col in columns:
        cu = col.upper()
        if cu == datetime_column.upper():
            out.contextual.append(col)
        elif cu in primitive_set:
            out.primitive.append(col)
        elif _matches_any(col, CONTEXTUAL_PATTERNS):
            out.contextual.append(col)
        elif _matches_any(col, DERIVED_PATTERNS):
            out.derived.append(col)
        else:
            # Conservative: unknown columns are treated as derived so that
            # they are recomputed by the feature engine after reconstruction
            # and never independently synthesized.
            out.derived.append(col)
            out.unknown.append(col)
    return out


__all__ = [
    "PRIMITIVE_COLUMNS_DEFAULT",
    "SchemaClassification",
    "classify_columns",
]
