"""CLI argument parsing for synthetic-datagen."""

import argparse
from app.config import DEFAULT_VALUES as D


def parse_args(argv=None):
    """Return (namespace, unknown_args)."""
    p = argparse.ArgumentParser(
        prog="sdg",
        description="Synthetic typical-price timeseries generator",
    )

    # Mode
    p.add_argument("--mode", choices=["train", "generate", "optimize", "evaluate"],
                   default=D["mode"])

    # Plugin selection
    p.add_argument("--trainer", default=D["trainer"])
    p.add_argument("--generator", default=D["generator"])
    p.add_argument("--evaluator", default=D["evaluator"])
    p.add_argument("--optimizer", default=D["optimizer"])

    # Data paths
    p.add_argument("--train_data", nargs="+", default=D["train_data"],
                   help="CSV file(s) for training")
    p.add_argument("--real_data", default=D["real_data"])
    p.add_argument("--synthetic_data", default=D["synthetic_data"])
    p.add_argument("--output_file", default=D["output_file"])
    p.add_argument("--metrics_file", default=D["metrics_file"])
    p.add_argument("--real_train", default=D["real_train"],
                   help="Real training CSV (d4) for predictive evaluation")
    p.add_argument("--real_val", default=D["real_val"],
                   help="Real validation CSV (d5) for predictive evaluation")
    p.add_argument("--real_test", default=D["real_test"],
                   help="Real test CSV (d6) for predictive evaluation")
    p.add_argument("--predictor_dir", default=D["predictor_dir"],
                   help="Path to external predictor repo (optional)")
    p.add_argument("--eval_epochs", type=int, default=D["eval_epochs"])
    p.add_argument("--eval_batch_size", type=int, default=D["eval_batch_size"])
    p.add_argument("--horizon", type=int, default=D["horizon"])

    # Model I/O
    p.add_argument("--save_model", default=D["save_model"])
    p.add_argument("--load_model", "--model", default=D["load_model"])

    # Training
    p.add_argument("--window_size", type=int, default=D["window_size"])
    p.add_argument("--batch_size", type=int, default=D["batch_size"])
    p.add_argument("--epochs", type=int, default=D["epochs"])
    p.add_argument("--learning_rate", type=float, default=D["learning_rate"])
    p.add_argument("--latent_dim", type=int, default=D["latent_dim"])
    p.add_argument("--activation", default=D["activation"])
    p.add_argument("--intermediate_layers", type=int, default=D["intermediate_layers"])
    p.add_argument("--initial_layer_size", type=int, default=D["initial_layer_size"])
    p.add_argument("--layer_size_divisor", type=int, default=D["layer_size_divisor"])
    p.add_argument("--kl_weight", type=float, default=D["kl_weight"])
    p.add_argument("--kl_anneal_epochs", type=int, default=D["kl_anneal_epochs"])
    p.add_argument("--mmd_lambda", type=float, default=D["mmd_lambda"])
    p.add_argument("--l2_reg", type=float, default=D["l2_reg"])
    p.add_argument("--use_returns", type=bool, default=D["use_returns"])
    p.add_argument("--early_patience", type=int, default=D["early_patience"])

    # GAN
    p.add_argument("--discriminator_lr", type=float, default=D["discriminator_lr"])
    p.add_argument("--generator_lr", type=float, default=D["generator_lr"])

    # Generation
    p.add_argument("--n_samples", type=int, default=D["n_samples"])
    p.add_argument("--seed", type=int, default=D["seed"])
    p.add_argument("--start_datetime", default=D["start_datetime"])
    p.add_argument("--interval_hours", type=int, default=D["interval_hours"])

    # Optimizer
    p.add_argument("--population_size", type=int, default=D["population_size"])
    p.add_argument("--n_generations", type=int, default=D["n_generations"])

    # Augmentation evaluator
    p.add_argument("--d4_file", default=D.get("d4_file"))
    p.add_argument("--d5_file", default=D.get("d5_file"))
    p.add_argument("--d6_file", default=D.get("d6_file"))
    p.add_argument("--predictor_root", default=D.get("predictor_root"))
    p.add_argument("--baseline_file", default=D.get("baseline_file"))

    # Config I/O
    p.add_argument("--load_config", default=D["load_config"])
    p.add_argument("--save_config", default=D["save_config"])
    p.add_argument("--save_log", default=D.get("save_log"))
    p.add_argument("--log_level", default=D["log_level"])
    p.add_argument("--quiet_mode", "--quiet", action="store_true",
                   default=D.get("quiet_mode", False))

    # Remote config / logging (interoperable with predictor + agent-multi)
    p.add_argument("--remote_load_config", default=D.get("remote_load_config"))
    p.add_argument("--remote_save_config", default=D.get("remote_save_config"))
    p.add_argument("--remote_log", default=D.get("remote_log"))
    p.add_argument("--username", default=D.get("username"))
    p.add_argument("--password", default=D.get("password"))

    # Plugin discovery (lists available plugins per group then exits)
    p.add_argument("--list_plugins", action="store_true",
                   help="List all available plugins (per group) and exit.")

    # --- Financial OHLCV mode ---
    p.add_argument("--financial_mode", action="store_true", default=D.get("financial_mode", False))
    p.add_argument("--data_format", choices=["csv", "parquet", "auto"], default=D.get("data_format", "auto"))
    p.add_argument("--datetime_column", default=D.get("datetime_column", "DATE_TIME"))
    p.add_argument("--asset_id", default=D.get("asset_id"))
    p.add_argument("--timeframe", default=D.get("timeframe"))
    p.add_argument("--train_start", default=D.get("train_start"))
    p.add_argument("--train_end", default=D.get("train_end"))
    p.add_argument("--validation_start", default=D.get("validation_start"))
    p.add_argument("--validation_end", default=D.get("validation_end"))
    p.add_argument("--heldout_start", default=D.get("heldout_start"))
    p.add_argument("--heldout_end", default=D.get("heldout_end"))
    p.add_argument("--block_length_mean", type=int, default=D.get("block_length_mean", 32))
    p.add_argument("--metadata_file", default=D.get("metadata_file"))
    p.add_argument("--synthetic_metadata_file", default=D.get("synthetic_metadata_file"))
    p.add_argument("--generated_feature_file", default=D.get("generated_feature_file"))

    # --- Project 3 augmentation governance ---
    p.add_argument("--project3_mode", action="store_true", default=D.get("project3_mode", False))
    p.add_argument("--heldout_boundary", default=D.get("heldout_boundary"))
    p.add_argument("--synthetic_use_case",
                   choices=["augmentation", "pretraining", "stress_test", "diagnostics"],
                   default=D.get("synthetic_use_case", "diagnostics"))
    p.add_argument("--allow_non_research_mode", action="store_true",
                   default=D.get("allow_non_research_mode", False))
    p.add_argument("--generator_family_id", default=D.get("generator_family_id"))
    p.add_argument("--synthetic_ablation_id", default=D.get("synthetic_ablation_id"))

    return p.parse_known_args(argv)
