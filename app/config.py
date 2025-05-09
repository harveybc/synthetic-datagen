"""
config.py for SDG (Synthetic Data Generator)

This module defines the default values for every command-line parameter
supported by the sdg application. These defaults are used when no value
is provided via CLI, config file, or remote config.
"""

DEFAULT_VALUES = {
    # Plugin selection
    "feeder": "default_feeder",
    "generator": "default_generator",
    "evaluator": "default_evaluator",
    "optimizer": "default_optimizer",

    # Generation parameters
    "n_samples": 1000,
    "latent_dim": 16,
    "batch_size": 32,
    "decoder_model_file": "examples/results/phase_4_1/decoder_model.keras",

    # Data for evaluation
    "real_data_file": "examples/data/phase_4_1/normalized_d1.csv",

    # Output paths
    "output_file": "examples/results/phase_4_1/synthetic_data.csv",
    "metrics_file": "examples/results/phase_4_1/evaluation_metrics.json",

    # Optimizer parameters
    "latent_dim_range": [8, 64],
    "iterations": 10,

    # Remote config & logging
    "remote_log": None,
    "remote_load_config": None,
    "remote_save_config": None,
    "username": None,
    "password": None,

    # Local config persistence
    "load_config": "examples/config/phase_4_1/sdg_config.json",
    "save_config": "examples/results/phase_4_1/config_out.json",
    "save_log": "examples/results/phase_4_1/debug_out.json",

    # CLI behavior
    "quiet_mode": False,
}
