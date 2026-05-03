# config_merger.py
#
# Six-source configuration merger, matching the predictor / agent-multi /
# preprocessor repos so plugins and configs are interchangeable across the
# Harvey-bc plugin ecosystem.
#
# Merge precedence (lowest -> highest):
#     plugin_params1 < plugin_params2 < repository defaults
#         < file_config < CLI args / unknown CLI overrides

import sys
from typing import Any, Dict, Iterable

from app.config import DEFAULT_VALUES  # noqa: F401  (re-exported for callers)


def process_unknown_args(unknown_args: Iterable[str]) -> Dict[str, str]:
    """Convert ``["--key", "value", ...]`` into ``{"key": "value", ...}``.

    Pairs without a matching value are ignored (defensive: argparse passes
    through trailing flags unchanged).
    """
    items: Dict[str, str] = {}
    pairs = list(unknown_args)
    for i in range(0, len(pairs) - 1, 2):
        key = pairs[i]
        if not key.startswith("--"):
            continue
        items[key.lstrip("--")] = pairs[i + 1]
    return items


def convert_type(value: Any) -> Any:
    """Best-effort coerce CLI string values to int/float, fall back to str."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def merge_config(
    defaults: Dict[str, Any],
    plugin_params1: Dict[str, Any],
    plugin_params2: Dict[str, Any],
    file_config: Dict[str, Any],
    cli_args: Dict[str, Any],
    unknown_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge configuration from six sources with explicit precedence.

    Step 1: plugin_params1 then plugin_params2 (lowest priority).
    Step 2: repository defaults (override plugin defaults).
    Step 3: file_config (CLI ``--load_config`` JSON or remote payload).
    Step 4: CLI args explicitly passed on the command line, plus any unknown
            ``--key value`` overrides not declared in argparse.

    The function intentionally inspects ``sys.argv`` to know which CLI keys
    were *explicitly* provided, so argparse defaults (which mirror
    DEFAULT_VALUES) do not silently overwrite file_config values.
    """
    merged: Dict[str, Any] = {}

    # Step 1: plugin defaults
    for k, v in (plugin_params1 or {}).items():
        merged[k] = v
    for k, v in (plugin_params2 or {}).items():
        merged[k] = v

    # Step 2: repository defaults
    for k, v in (defaults or {}).items():
        merged[k] = v

    # Step 3: file/remote config overrides defaults & plugin params
    for k, v in (file_config or {}).items():
        merged[k] = v

    # Step 4: CLI args (only those *explicitly* passed) and unknown args
    cli_keys = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]
    for key in cli_keys:
        if key in (cli_args or {}) and cli_args[key] is not None:
            merged[key] = cli_args[key]
        elif key in (unknown_args or {}):
            merged[key] = convert_type(unknown_args[key])

    return merged
