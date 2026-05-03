#!/usr/bin/env python3
"""
Entry point for synthetic-datagen (sdg).

Dispatches to four modes (train | generate | optimize | evaluate) using the
same plugin discovery + config-merging pattern as the predictor and
agent-multi repositories. This keeps configurations and plugins
interchangeable across the Harvey-bc plugin ecosystem.

Effective merge precedence (lowest -> highest):
    plugin defaults < repository defaults < file/remote config
        < CLI args / unknown CLI overrides

Two passes are performed:
    1. Pre-plugin pass — needed to know *which* plugin to load (the user can
       set ``"trainer": "..."`` in the config file or on the CLI).
    2. Post-plugin pass — re-merges so the freshly-loaded plugin's
       ``plugin_params`` defaults sit at the bottom of the precedence stack.
"""

import json
import logging
import os
import sys
from typing import Any, Dict

from app.audit import build_audit_record
from app.cli import parse_args
from app.synthetic_ledger import append_ledger
from app.config import DEFAULT_VALUES
from app.config_handler import (
    load_config,
    remote_load_config,
    remote_log,
    remote_save_config,
    save_config,
    save_debug_info,
)
from app.config_merger import merge_config, process_unknown_args
from app.plugin_loader import list_plugins, load_plugin


# Plugin-group lookup: which config key selects the plugin name for a mode.
_MODE_GROUPS = {
    "train":    ("trainer",   "sdg.trainer"),
    "generate": ("generator", "sdg.generator"),
    "optimize": ("optimizer", "sdg.optimizer"),
    "evaluate": ("evaluator", "sdg.evaluator"),
}

_ALL_GROUPS = (
    ("sdg.trainer",   "Trainers"),
    ("sdg.generator", "Generators"),
    ("sdg.evaluator", "Evaluators"),
    ("sdg.optimizer", "Optimizers"),
)


def _print_plugin_catalog() -> None:
    """Print every entry-point known to importlib.metadata under sdg.* groups."""
    print("Available plugins:")
    for group, label in _ALL_GROUPS:
        names = list_plugins(group)
        print(f"  [{label}] ({group}):")
        if not names:
            print("    (none installed)")
        for n in sorted(names):
            print(f"    - {n}")


def _initialize_plugin(group: str, name: str, config: Dict[str, Any]):
    """Load and instantiate a plugin (predictor-style: pass config, then set_params)."""
    plugin_class, _ = load_plugin(group, name)
    try:
        instance = plugin_class(config)
    except TypeError:
        # Plugins that don't accept config in __init__ — fall back to no-arg.
        instance = plugin_class()
    set_params = getattr(instance, "set_params", None)
    if callable(set_params):
        try:
            set_params(**config)
        except TypeError:
            # Strict signatures: pass only declared keys.
            declared = list(getattr(instance, "plugin_params", {}).keys())
            set_params(**{k: config[k] for k in declared if k in config})
    return instance


def main(argv=None) -> None:
    args, unknown = parse_args(argv)
    cli_args: Dict[str, Any] = vars(args)

    # Quick exit: --list_plugins
    if cli_args.get("list_plugins"):
        _print_plugin_catalog()
        return

    # ---- Step 1: load file/remote config (no plugin params yet) -----------
    file_config: Dict[str, Any] = {}
    if cli_args.get("remote_load_config"):
        rc = remote_load_config(
            cli_args["remote_load_config"],
            cli_args.get("username"),
            cli_args.get("password"),
        )
        if rc is None:
            print("Failed to load remote configuration", file=sys.stderr)
            sys.exit(1)
        file_config = rc
    if cli_args.get("load_config"):
        try:
            file_config.update(load_config(cli_args["load_config"]))
        except Exception as e:
            print(f"Failed to load local configuration: {e}", file=sys.stderr)
            sys.exit(1)

    unknown_args_dict = process_unknown_args(unknown)

    # First-pass merge: defaults + file_config + CLI (no plugin defaults yet).
    config = merge_config(
        DEFAULT_VALUES.copy(), {}, {}, file_config, cli_args, unknown_args_dict,
    )

    # ---- Step 2: configure logging ----------------------------------------
    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("sdg")
    if config.get("quiet_mode"):
        logging.getLogger().setLevel(logging.WARNING)

    mode = config.get("mode", "train")
    log.info(f"Mode: {mode}")

    if mode not in _MODE_GROUPS:
        log.error(f"Unknown mode: {mode}")
        sys.exit(1)

    plugin_key, plugin_group = _MODE_GROUPS[mode]
    plugin_name = config.get(plugin_key)
    if not plugin_name:
        log.error(f"No plugin specified for mode '{mode}' (config key '{plugin_key}')")
        sys.exit(1)

    # ---- Step 3: load + initialize the plugin -----------------------------
    log.info(f"Loading plugin: {plugin_group}::{plugin_name}")
    try:
        plugin = _initialize_plugin(plugin_group, plugin_name, config)
    except Exception as e:
        log.error(f"Failed to load plugin '{plugin_name}' from '{plugin_group}': {e}")
        sys.exit(1)

    # Second-pass merge: plugin defaults now at the bottom of the stack.
    plugin_params = getattr(plugin, "plugin_params", {}) or {}
    config = merge_config(
        DEFAULT_VALUES.copy(), plugin_params, {}, file_config,
        cli_args, unknown_args_dict,
    )
    # Re-apply set_params with the final config (in case anything changed).
    if hasattr(plugin, "set_params"):
        try:
            plugin.set_params(**config)
        except TypeError:
            declared = list(plugin_params.keys())
            plugin.set_params(**{k: config[k] for k in declared if k in config})

    # ---- Step 4: dispatch -------------------------------------------------
    debug_info: Dict[str, Any] = {"mode": mode, "plugin": plugin_name}

    if mode == "train":
        if not config.get("train_data"):
            log.error("--train_data required for train mode")
            sys.exit(1)
        plugin.train()
        log.info(f"Model saved -> {config.get('save_model')}")

    elif mode == "generate":
        if not config.get("load_model"):
            log.error("--load_model (--model) required for generate mode")
            sys.exit(1)
        plugin.run_generate()
        log.info(f"Synthetic data -> {config.get('output_file')}")

    elif mode == "optimize":
        if not config.get("train_data"):
            log.error("--train_data required for optimize mode")
            sys.exit(1)
        best = plugin.optimize()
        debug_info["best_params"] = best
        log.info(f"Best params: {best}")

    elif mode == "evaluate":
        if not config.get("synthetic_data"):
            log.error("--synthetic_data required for evaluate mode")
            sys.exit(1)
        metrics = plugin.evaluate()
        out = config.get("metrics_file", "metrics.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        debug_info["metrics_file"] = out
        log.info(f"Metrics -> {out}")

    # ---- Step 5: persist artifacts ---------------------------------------
    # Write a synthetic-data audit sidecar in financial_mode (always cheap
    # and required by the spec for traceability).
    if config.get("financial_mode"):
        inputs: Dict[str, Any] = {}
        if config.get("train_data"):
            td = config["train_data"]
            inputs["train_data"] = td if isinstance(td, str) else td[0]
        if config.get("synthetic_data"):
            inputs["synthetic_data"] = config["synthetic_data"]
        if config.get("real_data"):
            inputs["real_data"] = config["real_data"]
        audit = build_audit_record(config, input_files=inputs, extra=debug_info)
        meta_path = config.get("synthetic_metadata_file") or config.get("metadata_file")
        if meta_path:
            try:
                with open(meta_path, "w") as f:
                    json.dump(audit, f, indent=2, default=str)
                log.info(f"Audit metadata -> {meta_path}")
            except Exception as e:
                log.warning(f"Failed to write audit metadata: {e}")
        debug_info["audit"] = audit

        # Append a row to the synthetic ledger (Phase 4 §3 audit trail).
        ledger_path = append_ledger(
            config, kind=mode, audit=audit,
            extra={
                "model_file": config.get("save_model") or config.get("load_model"),
                "output_file": config.get("output_file"),
                "metrics_file": config.get("metrics_file"),
                "valid": (debug_info.get("best_params") is not None
                          or mode in ("train", "generate", "evaluate")),
            },
        )
        if ledger_path:
            log.info(f"Synthetic ledger appended -> {ledger_path}")

    if config.get("save_config"):
        try:
            save_config(config, config["save_config"])
            log.info(f"Config saved -> {config['save_config']}")
        except Exception as e:
            log.warning(f"Failed to save configuration locally: {e}")

    if config.get("save_log"):
        try:
            save_debug_info(debug_info, config["save_log"])
        except Exception as e:
            log.warning(f"Failed to save debug log: {e}")

    if config.get("remote_save_config"):
        ok = remote_save_config(
            config, config["remote_save_config"],
            config.get("username"), config.get("password"),
        )
        log.info("Remote config save: %s", "ok" if ok else "FAILED")

    if config.get("remote_log"):
        ok = remote_log(
            config, debug_info, config["remote_log"],
            config.get("username"), config.get("password"),
        )
        log.info("Remote log push: %s", "ok" if ok else "FAILED")


if __name__ == "__main__":
    main()
