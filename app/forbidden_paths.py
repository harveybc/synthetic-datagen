"""Forbidden-paths guard.

Reads ``forbidden_paths.txt`` (relative to CWD or repo root) and
refuses to open any input file matching one of those globs.  Used to
prevent Stage 4.2 evaluators (and Stage 4.3 generator-selection logic)
from accidentally loading the 2025 heldout window when picking the
winning generator family.

Each line is either a literal path or a fnmatch-style glob.  Blank
lines and ``#`` comments are ignored.
"""
from __future__ import annotations

import fnmatch
import os
from typing import Iterable, List, Optional


def _candidate_locations() -> List[str]:
    cwd = os.getcwd()
    return [
        os.path.join(cwd, "forbidden_paths.txt"),
        os.path.join(os.path.dirname(__file__), "..", "forbidden_paths.txt"),
    ]


def load_forbidden_globs(extra: Optional[Iterable[str]] = None) -> List[str]:
    """Return the (deduplicated) list of forbidden-path globs."""
    out: List[str] = []
    seen = set()
    for cand in _candidate_locations():
        if not os.path.exists(cand):
            continue
        with open(cand, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
    if extra:
        for s in extra:
            if s and s not in seen:
                seen.add(s); out.append(s)
    return out


def assert_path_allowed(path: str, *, extra_globs: Optional[Iterable[str]] = None) -> None:
    """Raise ``ValueError`` if ``path`` matches any forbidden glob."""
    if not path:
        return
    globs = load_forbidden_globs(extra_globs)
    if not globs:
        return
    abspath = os.path.abspath(path)
    for g in globs:
        if fnmatch.fnmatch(path, g) or fnmatch.fnmatch(abspath, g):
            raise ValueError(
                f"Refusing to open '{path}' — matches forbidden_paths "
                f"glob '{g}'. Generator selection MUST NOT see Stage C / "
                f"2025 heldout data (Phase 4 §7)."
            )


__all__ = ["assert_path_allowed", "load_forbidden_globs"]
