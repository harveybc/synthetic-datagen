#!/usr/bin/env python3
"""
main.py

Punto de entrada de la aplicación de predicción de EUR/USD. Este script orquesta:
    - La carga y fusión de configuraciones (CLI, archivos locales y remotos).
    - La inicialización de los plugins: Predictor, Optimizer, Pipeline y Preprocessor.
    - La selección entre ejecutar la optimización de hiperparámetros o entrenar y evaluar directamente.
    - El guardado de la configuración resultante de forma local y/o remota.
"""

import sys
import json
import pandas as pd
from typing import Any, Dict

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    """
    Orquesta la ejecución completa del sistema, incluyendo la optimización (si se configura)
    y la ejecución del pipeline completo (preprocesamiento, entrenamiento, predicción y evaluación).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # Selección del plugin Feeder
    if not cli_args.get('feeder'):
        cli_args['feeder'] = config.get('feeder', 'default_feeder')
    plugin_name = config.get('feeder', 'default_feeder')
    print(f"Loading Feeder Plugin: {plugin_name}")
    try:
        feeder_class, _ = load_plugin('feeder.plugins', plugin_name)
        feeder_plugin = feeder_class(config)
        feeder_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Feeder Plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Selección del plugin Generator
    if not cli_args.get('generator'):
        cli_args['generator'] = config.get('generator', 'default_generator')
    plugin_name = config.get('generator', 'default_generator')
    print(f"Loading Generator Plugin: {plugin_name}")
    try:
        # ADD THIS DEBUG BLOCK
        returned_value_for_generator = load_plugin('generator.plugins', plugin_name)
        print(f"DEBUG main.py: Received from load_plugin for generator: {returned_value_for_generator}")
        print(f"DEBUG main.py: Type of received value: {type(returned_value_for_generator)}")
        if isinstance(returned_value_for_generator, (tuple, list)): # Check if it's an iterable
            print(f"DEBUG main.py: Length of received iterable: {len(returned_value_for_generator)}")
        # END DEBUG BLOCK
        
        generator_class, _ = returned_value_for_generator # Use the inspected value
        generator_plugin = generator_class(config)
        generator_plugin.set_params(**config)

        # --- INSERTAR AQUÍ: Inferir latent_dim desde el decoder y actualizar feeder_plugin ---
        decoder_model = getattr(generator_plugin, "model", None)
        if decoder_model is None:
            raise RuntimeError("GeneratorPlugin must expose attribute 'model'.")
        # decoder_model.input_shape == (None, latent_dim)
        _, inferred_latent = decoder_model.input_shape
        feeder_plugin.set_params(latent_dim=int(inferred_latent))
        # -------------------------------------------------------------------------------
    except Exception as e:
        print(f"Failed to load or initialize Generator Plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Selección del plugin Evaluator
    if not cli_args.get('evaluator'):
        cli_args['evaluator'] = config.get('evaluator', 'default_evaluator')
    plugin_name = config.get('evaluator', 'default_evaluator')
    print(f"Loading Evaluator Plugin: {plugin_name}")
    try:
        evaluator_class, _ = load_plugin('evaluator.plugins', plugin_name)
        evaluator_plugin = evaluator_class(config)
        evaluator_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Evaluator Plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Selección del plugin Optimizer
    if not cli_args.get('optimizer'):
        cli_args['optimizer'] = config.get('optimizer', 'default_optimizer')
    plugin_name = config.get('optimizer', 'default_optimizer')
    print(f"Loading Optimizer Plugin: {plugin_name}")
    try:
        optimizer_class, _ = load_plugin('optimizer.plugins', plugin_name)
        optimizer_plugin = optimizer_class(config)
        optimizer_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Optimizer Plugin '{plugin_name}': {e}")
        sys.exit(1)

    print("Merging configuration with CLI arguments and plugin parameters...")
    config = merge_config(config, feeder_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, generator_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, evaluator_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, optimizer_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    # --- DECISIÓN DE EJECUCIÓN ---
    if config.get('use_optimizer', False):
        print("Running hyperparameter optimization with Optimizer Plugin...")
        try:
            optimal_params = optimizer_plugin.optimize(
                feeder_plugin,
                generator_plugin,
                evaluator_plugin,
                config
            )
            optimizer_output_file = config.get(
                "optimizer_output_file",
                "examples/results/phase_4_1/optimizer_output.json"
            )
            with open(optimizer_output_file, "w") as f:
                json.dump(optimal_params, f, indent=4)
            print(f"Optimized parameters saved to {optimizer_output_file}.")
            config.update(optimal_params)
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            sys.exit(1)
    else:
        print("Skipping hyperparameter optimization.")
        print("Generating synthetic data and evaluating...")

        try:
            # 1. Muestreo latente (usar solo n_samples; latent_dim ya fue seteado)
            Z = feeder_plugin.generate(config['n_samples'])
            # 2. Decodificación a ventanas sintéticas
            X_syn = generator_plugin.generate(Z)
            # 3. Guardado de datos sintéticos
            output_file = config['output_file']
            pd.DataFrame(X_syn.reshape(-1, X_syn.shape[-1])).to_csv(
                output_file, index=False, header=False
            )
            print(f"Synthetic data saved to {output_file}.")
            # 4. Evaluación de las ventanas generadas
            metrics = evaluator_plugin.evaluate(
                synthetic_data=X_syn,
                real_data_file=config['real_data_file']
            )
            metrics_file = config['metrics_file']
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Evaluation metrics saved to {metrics_file}.")
        except Exception as e:
            print(f"Synthetic data generation or evaluation failed: {e}")
            sys.exit(1)

    if config.get('save_config'):
        try:
            save_config(config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")

    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(
                config,
                config['remote_save_config'],
                config.get('username'),
                config.get('password')
            )
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")

if __name__ == "__main__":
    main()
