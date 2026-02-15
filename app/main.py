#!/usr/bin/env python3
"""
Entry point for synthetic-datagen (sdg).

Thin CLI wrapper around plugin-first architecture.
All plugins have clean programmatic APIs — the CLI just wires config → plugin.
"""

import json
import logging
import sys
from typing import Dict

from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin


def _merge_config(defaults: Dict, cli: Dict) -> Dict:
    cfg = defaults.copy()
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main(argv=None):
    args, unknown = parse_args(argv)
    cli = vars(args)

    config = DEFAULT_VALUES.copy()
    if cli.get("load_config"):
        with open(cli["load_config"]) as f:
            config.update(json.load(f))

    config = _merge_config(config, cli)

    logging.basicConfig(
        level=getattr(logging, config["log_level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("sdg")
    log.info(f"Mode: {config['mode']}")

    mode = config["mode"]

    # ── TRAIN ───────────────────────────────────────────────────────────
    if mode == "train":
        if not config["train_data"]:
            log.error("--train_data required for train mode")
            sys.exit(1)
        trainer_cls = load_plugin("sdg.trainer", config["trainer"])
        trainer = trainer_cls()
        trainer.configure(config)
        trainer.train(
            train_data=config["train_data"],
            save_model=config["save_model"],
        )
        log.info(f"Model saved → {config['save_model']}")

    # ── GENERATE ────────────────────────────────────────────────────────
    elif mode == "generate":
        if not config.get("load_model"):
            log.error("--load_model (--model) required for generate mode")
            sys.exit(1)
        gen_cls = load_plugin("sdg.generator", config["generator"])
        gen = gen_cls()
        gen.configure(config)
        gen.run_generate()

    # ── OPTIMIZE ────────────────────────────────────────────────────────
    elif mode == "optimize":
        if not config["train_data"]:
            log.error("--train_data required for optimize mode")
            sys.exit(1)
        opt_cls = load_plugin("sdg.optimizer", config["optimizer"])
        opt = opt_cls()
        opt.configure(config)
        best = opt.optimize()
        log.info(f"Best params: {best}")

    # ── EVALUATE ────────────────────────────────────────────────────────
    elif mode == "evaluate":
        if not config.get("synthetic_data") or not config.get("real_data"):
            log.error("--synthetic_data and --real_data required for evaluate mode")
            sys.exit(1)
        eval_cls = load_plugin("sdg.evaluator", config["evaluator"])
        ev = eval_cls()
        ev.configure(config)
        metrics = ev.evaluate()
        out = config["metrics_file"]
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Metrics → {out}")

    # Save config if requested
    if config.get("save_config"):
        with open(config["save_config"], "w") as f:
            json.dump(config, f, indent=2, default=str)
        log.info(f"Config saved → {config['save_config']}")


if __name__ == "__main__":
    main()
