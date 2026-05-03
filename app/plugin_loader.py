#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugin_loader.py

Plugin discovery via ``importlib.metadata`` entry points, mirroring the
predictor / agent-multi / preprocessor repos so plugins can be exchanged
across projects that share the same plugin architecture.

Public API:
    load_plugin(group, name) -> (plugin_class, required_params_list)
    get_plugin_params(group, name) -> dict
    list_plugins(group) -> list[str]
"""

from importlib.metadata import entry_points
from typing import Any, Dict, List, Tuple


def _select(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # type: ignore[attr-defined]


def load_plugin(plugin_group: str, plugin_name: str) -> Tuple[Any, List[str]]:
    """Load a plugin class and return ``(class, required_param_keys)``.

    Raises ImportError when the plugin is not installed under
    ``plugin_group``. The returned ``required_param_keys`` is the list of
    keys declared by the plugin's class-level ``plugin_params`` dict (may
    be empty if the plugin has no declared defaults).
    """
    try:
        group_entries = _select(plugin_group)
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        required_params = list(getattr(plugin_class, "plugin_params", {}).keys())
        return plugin_class, required_params
    except StopIteration as exc:
        available = [ep.name for ep in _select(plugin_group)]
        raise ImportError(
            f"Plugin '{plugin_name}' not found in group '{plugin_group}'. "
            f"Available: {available}"
        ) from exc
    except Exception as exc:
        raise ImportError(
            f"Failed to load plugin '{plugin_name}' from group '{plugin_group}': {exc}"
        ) from exc


def get_plugin_params(plugin_group: str, plugin_name: str) -> Dict[str, Any]:
    """Return the class-level ``plugin_params`` dict for the given plugin."""
    plugin_class, _ = load_plugin(plugin_group, plugin_name)
    return dict(getattr(plugin_class, "plugin_params", {}))


def list_plugins(group: str) -> List[str]:
    """Return available plugin names under ``group``."""
    return [ep.name for ep in _select(group)]
