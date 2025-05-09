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

# Se asume que los siguientes plugins se cargan desde sus respectivos namespaces:
# - predictor.plugins
# - optimizer.plugins
# - pipeline.plugins
# - preprocessor.plugins

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
    # Carga remota de configuración si se solicita
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Carga local de configuración si se solicita
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # Primera fusión de la configuración (sin parámetros específicos de plugins)
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
        feeder_plugin = feeder_class()
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
        generator_class, _ = load_plugin('generator.plugins', plugin_name)
        generator_plugin = generator_class()
        generator_plugin.set_params(**config)
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
        evaluator_plugin = evaluator_class()
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
        optimizer_plugin = optimizer_class()
        optimizer_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Optimizer Plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Fusión de configuración con parámetros específicos de cada plugin
    print("Merging configuration with CLI arguments and plugin parameters...")
    config = merge_config(config, feeder_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, generator_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, evaluator_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, optimizer_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    

    # --- DECISIÓN DE EJECUCIÓN ---
    # Si se activa el optimizador, ejecutar optimización antes de generar datos
    if config.get('use_optimizer', False):
        print("Running hyperparameter optimization with Optimizer Plugin...")
        try:
            # El optimizador recibe los plugins y la configuración
            optimal_params = optimizer_plugin.optimize(
                feeder_plugin,
                generator_plugin,
                evaluator_plugin,
                config
            )
            # Guardar parámetros óptimos en JSON
            optimizer_output_file = config.get(
                "optimizer_output_file",
                "examples/results/phase_4_1/optimizer_output.json"
            )
            with open(optimizer_output_file, "w") as f:
                json.dump(optimal_params, f, indent=4)
            print(f"Optimized parameters saved to {optimizer_output_file}.")
            # Actualizar configuración con los parámetros optimizados
            config.update(optimal_params)
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            sys.exit(1)
    else:
        print("Skipping hyperparameter optimization.")
        print("Generating synthetic data and evaluating...")

        try:
            # 1. Muestreo latente
            Z = feeder_plugin.generate(
                n_samples=config['n_samples'],
                latent_dim=config['latent_dim']
            )
            # 2. Decodificación a ventanas sintéticas
            X_syn = generator_plugin.generate(Z)
            # 3. Guardado de datos sintéticos
            output_file = config['output_file']
            pd.DataFrame(
                X_syn.reshape(-1, X_syn.shape[-1])
            ).to_csv(output_file, index=False, header=False)
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

    # Guardado de la configuración local y remota
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
