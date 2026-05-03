# config_handler.py
#
# Local + remote configuration I/O with plugin-aware compaction.
# Mirrors the predictor / agent-multi pattern so a saved config_out.json
# only contains the deltas vs. defaults + plugin defaults.

import json
import sys
from typing import Any, Dict, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover — optional, only needed for remote ops
    requests = None  # type: ignore

from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin


# Plugin groups that may contribute defaults to the saved config delta.
# Each entry: (config_key_for_plugin_name, entry_point_group).
_PLUGIN_GROUPS = (
    ("trainer", "sdg.trainer"),
    ("generator", "sdg.generator"),
    ("evaluator", "sdg.evaluator"),
    ("optimizer", "sdg.optimizer"),
)


def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def get_plugin_default_params(group: str, plugin_name: str,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Instantiate a plugin and return its ``plugin_params`` dict (best-effort)."""
    try:
        plugin_class, _ = load_plugin(group, plugin_name)
    except Exception:
        return {}
    try:
        instance = plugin_class(config) if config is not None else plugin_class()
        params = getattr(instance, "plugin_params", None)
        if not params:
            params = getattr(plugin_class, "plugin_params", {}) or {}
        return dict(params)
    except Exception:
        return dict(getattr(plugin_class, "plugin_params", {}) or {})


def compose_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a minimal config dict (only values different from defaults).

    Removes entries that exactly match either DEFAULT_VALUES or the active
    plugins' ``plugin_params``, keeping the saved config compact and focused
    on what the user actually changed.
    """
    plugin_defaults: Dict[str, Any] = {}
    for cfg_key, group in _PLUGIN_GROUPS:
        name = config.get(cfg_key)
        if not name:
            continue
        plugin_defaults.update(get_plugin_default_params(group, name, config))

    out: Dict[str, Any] = {}
    for k, v in config.items():
        if k in DEFAULT_VALUES and v == DEFAULT_VALUES[k]:
            continue
        if k in plugin_defaults and v == plugin_defaults[k]:
            continue
        out[k] = v
    return out


def save_config(config: Dict[str, Any], path: str = "config_out.json"):
    config_to_save = compose_config(config)
    with open(path, "w") as f:
        json.dump(config_to_save, f, indent=4, default=str)
    return config, path


def save_debug_info(debug_info: Dict[str, Any], path: str = "debug_out.json") -> None:
    with open(path, "w") as f:
        json.dump(debug_info, f, indent=4, default=str)


def remote_load_config(url: str, username: Optional[str] = None,
                       password: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if requests is None:
        print("remote_load_config: 'requests' is not installed", file=sys.stderr)
        return None
    try:
        if username and password:
            response = requests.get(url, auth=(username, password))
        else:
            response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:  # pragma: no cover
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None


def remote_save_config(config: Dict[str, Any], url: str,
                       username: Optional[str], password: Optional[str]) -> bool:
    if requests is None:
        print("remote_save_config: 'requests' is not installed", file=sys.stderr)
        return False
    config_to_save = compose_config(config)
    try:
        response = requests.post(
            url,
            auth=(username, password) if username else None,
            data={"json_config": json.dumps(config_to_save, default=str)},
        )
        response.raise_for_status()
        return True
    except Exception as e:  # pragma: no cover
        print(f"Failed to save remote configuration: {e}", file=sys.stderr)
        return False


def remote_log(config: Dict[str, Any], debug_info: Dict[str, Any], url: str,
               username: Optional[str], password: Optional[str]) -> bool:
    if requests is None:
        print("remote_log: 'requests' is not installed", file=sys.stderr)
        return False
    config_to_save = compose_config(config)
    try:
        response = requests.post(
            url,
            auth=(username, password) if username else None,
            data={
                "json_config": json.dumps(config_to_save, default=str),
                "json_result": json.dumps(debug_info, default=str),
            },
        )
        response.raise_for_status()
        return True
    except Exception as e:  # pragma: no cover
        print(f"Failed to log remote information: {e}", file=sys.stderr)
        return False
