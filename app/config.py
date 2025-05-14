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

    # Data for evaluation
    "real_data_file": "examples/data/phase_3/normalized_d4.csv",
    "x_train_file": "examples/data/phase_3/normalized_d4.csv",
    "y_train_file": "examples/data/phase_3/normalized_d4.csv",
    "x_validation_file": "examples/data/phase_3/normalized_d5.csv",
    "y_validation_file": "examples/data/phase_3/normalized_d5.csv",
    "x_test_file": "examples/data/phase_3/normalized_d6.csv",
    "y_test_file": "examples/data/phase_3/normalized_d6.csv",
    "target_column": "CLOSE",
    "stl_period":24,
    "predicted_horizons": [24,48,72,96,120,144],
    "use_stl": True,
    "use_wavelets": True,
    "use_multi_tapper": True,

    "dataset_periodicity": "1h", 

     # Generation parameters
    "n_samples": 6300,  
    "latent_dim": 16,
    "batch_size": 32,
    "decoder_model_file": "examples/results/phase_4_1/phase_4_1_cnn_small_decoder_model.h5.keras",
    "encoder_model_file": "examples/results/phase_4_1/phase_4_1_cnn_small_encoder_model.h5.keras",
    "max_steps_train": 6300,
    "max_steps_val": 6300,
    "max_steps_test": 6300,

    # Output paths
    "output_file": "examples/results/phase_4_1/normalized_d4_25200_synthetic_12600_prepended.csv",
    "metrics_file": "examples/results/phase_4_1/normalized_d4_25200_synthetic_12600_metrics.json",

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
    "load_config": None,
    "save_config": "examples/results/phase_4_1/config_out.json",
    "save_log": "examples/results/phase_4_1/debug_out.json",

    # CLI behavior
    "quiet_mode": False,
}
