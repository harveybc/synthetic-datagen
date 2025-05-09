"""
cli.py for SDG (Synthetic Data Generator)

This module defines the command-line interface for the sdg application,
including arguments for feeder, generator, evaluator, optimizer plugins,
generation and evaluation parameters, and remote configuration/logging.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="SDG: Synthetic Data Generator with plugin-based architecture."
    )

    # Plugin selection
    parser.add_argument('--feeder', type=str,
                        help='Name of the feeder plugin to use.')
    parser.add_argument('--generator', type=str,
                        help='Name of the generator plugin to use.')
    parser.add_argument('--evaluator', type=str,
                        help='Name of the evaluator plugin to use.')
    parser.add_argument('--optimizer', type=str,
                        help='Name of the optimizer plugin to use.')

    # Generation parameters
    parser.add_argument('-ns', '--n_samples', type=int,
                        help='Number of synthetic samples (windows) to generate.')
    parser.add_argument('-ld', '--latent_dim', type=int,
                        help='Dimension of the latent space to use for generation.')
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Batch size for the generation process.')
    parser.add_argument('-dm', '--decoder_model_file', type=str,
                        help='Path to the pretrained decoder model file.')

    # Evaluation parameters
    parser.add_argument('-rdf', '--real_data_file', type=str,
                        help='Path to the CSV file with real reference data for evaluation.')
    parser.add_argument('-of', '--output_file', type=str,
                        help='Path to the output CSV file for synthetic data.')
    parser.add_argument('-mf', '--metrics_file', type=str,
                        help='Path to the output file for evaluation metrics (JSON or CSV).')

    # Optimizer parameters
    parser.add_argument('-le', '--latent_dim_range', type=int, nargs=2,
                        help='Min and max latent dimensions for optimization.')
    parser.add_argument('-it', '--iterations', type=int,
                        help='Number of optimizer iterations (generations).')

    # Remote config & logging
    parser.add_argument('-rl', '--remote_log', type=str,
                        help='URL of remote API endpoint for saving debug information.')
    parser.add_argument('-rlc', '--remote_load_config', type=str,
                        help='URL of remote JSON configuration file to download and execute.')
    parser.add_argument('-rsc', '--remote_save_config', type=str,
                        help='URL of remote API endpoint to save configuration in JSON format.')
    parser.add_argument('-u', '--username', type=str,
                        help='Username for the API endpoint.')
    parser.add_argument('-p', '--password', type=str,
                        help='Password for the API endpoint.')
    parser.add_argument('-lc', '--load_config', type=str,
                        help='Path to load a configuration file.')
    parser.add_argument('-sc', '--save_config', type=str,
                        help='Path to save the current configuration.')
    parser.add_argument('-sl', '--save_log', type=str,
                        help='Path to save the current debug information.')
    parser.add_argument('-qm', '--quiet_mode', action='store_true',
                        help='Suppress output messages.')

    return parser.parse_known_args()
