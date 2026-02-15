"""Plugin discovery via setuptools entry-points."""

from importlib.metadata import entry_points
from typing import Any, Optional


def load_plugin(group: str, name: str) -> Any:
    """
    Load a plugin class by entry-point *group* and *name*.

    Returns the plugin **class** (not an instance).
    Raises RuntimeError if not found.
    """
    eps = entry_points()
    # Python 3.12+ returns a dict-like; 3.10–3.11 returns SelectableGroups
    if hasattr(eps, "select"):
        matches = eps.select(group=group, name=name)
    else:
        matches = [ep for ep in eps.get(group, []) if ep.name == name]

    for ep in matches:
        return ep.load()

    raise RuntimeError(
        f"Plugin '{name}' not found in group '{group}'. "
        f"Install the package or check entry_points in pyproject.toml."
    )


def list_plugins(group: str) -> list[str]:
    """Return available plugin names for *group*."""
    eps = entry_points()
    if hasattr(eps, "select"):
        return [ep.name for ep in eps.select(group=group)]
    return [ep.name for ep in eps.get(group, [])]
