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

    # Config I/O
    p.add_argument("--load_config", default=D["load_config"])
    p.add_argument("--save_config", default=D["save_config"])
    p.add_argument("--log_level", default=D["log_level"])

    return p.parse_known_args(argv)
