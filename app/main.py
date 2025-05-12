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
import numpy as np
import os # For path operations and temp file removal
import tempfile # For creating temporary files

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

        # --- Inferir latent_dim desde el decoder y actualizar feeder_plugin ---
        decoder_model = getattr(generator_plugin, "model", None)
        if decoder_model is None:
            raise RuntimeError("GeneratorPlugin must expose attribute 'model'.")
        
        print(f"DEBUG main.py: Raw decoder_model.input_shape: {decoder_model.input_shape}") # Add this line to see the shape

        input_shape = decoder_model.input_shape
        if not (isinstance(input_shape, tuple) and len(input_shape) >= 1):
            raise RuntimeError(f"Unexpected decoder_model.input_shape: {input_shape}. Expected a tuple.")

        # Assuming the actual latent dimension is the last element of the shape.
        # Handles shapes like (None, latent_dim) or (None, ..., latent_dim)
        inferred_latent_val = input_shape[-1]
        
        # If the last dimension is None (e.g. for variable length sequences, not typical for latent_dim itself),
        # and there's a preceding dimension, try using that.
        if inferred_latent_val is None and len(input_shape) > 1:
            print(f"WARNING main.py: Last element of input_shape {input_shape} is None. Attempting to use second to last.")
            inferred_latent_val = input_shape[-2] 

        if inferred_latent_val is None:
            raise RuntimeError(f"Could not determine a valid inferred latent dimension from shape: {input_shape}")

        print(f"DEBUG main.py: Inferred latent_dim value: {inferred_latent_val} from shape: {input_shape}")
        feeder_plugin.set_params(latent_dim=int(inferred_latent_val))
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

    # Carga del Preprocessor Plugin (para process_data, ventanas deslizantes y STL)
    # Using the exact loading mechanism you provided:
    plugin_name = config.get('preprocessor_plugin', 'stl_preprocessor') # Default name from your snippet
    print(f"Loading Plugin ..{plugin_name}") # Print format from your snippet
    try:
        # Ensure 'preprocessor.plugins' is the correct group name for your preprocessor
        preprocessor_class, _ = load_plugin('preprocessor.plugins', plugin_name) # Group from your snippet
        preprocessor_plugin = preprocessor_class() # Instantiation from your snippet
        preprocessor_plugin.set_params(**config) # Param setting from your snippet
    except Exception as e:
        print(f"Failed to load or initialize Preprocessor Plugin: {e}") # Error message format from your snippet
        sys.exit(1)

    print("Merging configuration with CLI arguments and plugin parameters...")
    config = merge_config(config, feeder_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, generator_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, evaluator_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, optimizer_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # Add preprocessor plugin params to merge if it has 'plugin_params' attribute
    if hasattr(preprocessor_plugin, 'plugin_params'):
        config = merge_config(config, preprocessor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)


    # --- DECISIÓN DE EJECUCIÓN ---
    if config.get('use_optimizer', False):
        print("Running hyperparameter optimization with Optimizer Plugin...")
        try:
            optimal_params = optimizer_plugin.optimize(
                feeder_plugin,
                generator_plugin,
                evaluator_plugin, # Pass the evaluator instance
                preprocessor_plugin, # Pass the preprocessor instance
                config
            )
            optimizer_output_file = config.get(
                "optimizer_output_file",
                "examples/results/phase_4_1/optimizer_output.json"
            )
            with open(optimizer_output_file, "w") as f:
                json.dump(optimal_params, f, indent=4)
            print(f"Optimized parameters saved to {optimizer_output_file}.")
            config.update(optimal_params) # Update main config with optimal params
            # Re-set params for plugins if they were optimized
            feeder_plugin.set_params(**config)
            generator_plugin.set_params(**config)
            # Evaluator and Preprocessor might not have optimizable params, but good practice if they could
            evaluator_plugin.set_params(**config)
            preprocessor_plugin.set_params(**config)

        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            sys.exit(1)
    else:
        print("Skipping hyperparameter optimization.")
        print("Generating synthetic data and evaluating...")

        try:
            # 0. Preprocess real data for evaluation using the loaded PreprocessorPlugin
            print("Preprocessing real data for evaluation via PreprocessorPlugin...")
            
            if not hasattr(preprocessor_plugin, 'run_preprocessing'):
                raise AttributeError("PreprocessorPlugin does not have a 'run_preprocessing' method.")
            
            config_for_preprocessor_run = config.copy()
            print(f"DEBUG main.py: Initial 'config_for_preprocessor_run' before any workarounds: {config_for_preprocessor_run}")


            # WORKAROUND 1: For "Baseline train indices invalid"
            # If use_stl is False, the external stl_preprocessor might still use
            # effective_stl_window in original_offset calculation.
            # Setting stl_window = 1 forces effective_stl_window to 1, correcting the offset.
            if not config_for_preprocessor_run.get('use_stl', False): 
                if 'stl_window' in preprocessor_plugin.plugin_params or 'stl_window' in config_for_preprocessor_run:
                    original_stl_window = config_for_preprocessor_run.get('stl_window')
                    config_for_preprocessor_run['stl_window'] = 1 
                    print(f"INFO: synthetic-datagen/main.py: WORKAROUND 1: 'use_stl' is False. Original 'stl_window': {original_stl_window}. Temporarily setting 'stl_window' to 1 to prevent 'Baseline train indices invalid' error.")
            else:
                print(f"DEBUG main.py: WORKAROUND 1: 'use_stl' is True or not set, 'stl_window' workaround not applied.")


            # WORKAROUND 2: For "Wavelet: Length of data must be even"
            # The part below attempts to ensure the input file has an even number of rows.
            # The part that disables wavelets via ACTUAL_WAVELET_CONFIG_KEY should be removed
            # if wavelets are mandatory and you are relying solely on this file truncation.
            
            # --- Ensure input file has even length ---
            temp_data_file_to_delete = None
            original_real_data_file_path = config_for_preprocessor_run.get('real_data_file')
            print(f"DEBUG main.py: Wavelet Workaround: Original 'real_data_file' path from config: {original_real_data_file_path}")

            if original_real_data_file_path and os.path.exists(original_real_data_file_path):
                try:
                    df_real_data = pd.read_csv(original_real_data_file_path)
                    data_len = len(df_real_data)
                    print(f"DEBUG main.py: Wavelet Workaround: Read '{original_real_data_file_path}', original length: {data_len}")
                    if data_len > 0 and data_len % 2 != 0: 
                        print(f"INFO: synthetic-datagen/main.py: Wavelet Workaround: Data in '{original_real_data_file_path}' has odd length ({data_len}). Truncating last row.")
                        df_real_data_truncated = df_real_data.iloc[:-1]
                        
                        if not df_real_data_truncated.empty: # Ensure not empty after truncation
                            with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as tmp_file_obj:
                                df_real_data_truncated.to_csv(tmp_file_obj.name, index=False)
                                temp_data_file_to_delete = tmp_file_obj.name
                            
                            config_for_preprocessor_run['real_data_file'] = temp_data_file_to_delete
                            print(f"INFO: synthetic-datagen/main.py: Wavelet Workaround: Preprocessor will use temporary even-length data file: {temp_data_file_to_delete}. New length: {len(df_real_data_truncated)}")
                        else:
                            # This case means original length was 1. Truncating makes it 0.
                            # pywt.swt will also fail on 0-length. Let preprocessor handle original 1-row file.
                            print(f"WARN: synthetic-datagen/main.py: Wavelet Workaround: Original data in '{original_real_data_file_path}' had 1 row. Truncating made it empty. Preprocessor will use original file (which has odd length). Wavelet error may still occur.")
                    else:
                        print(f"INFO: synthetic-datagen/main.py: Wavelet Workaround: Data in '{original_real_data_file_path}' has length {data_len} (even or zero). No truncation of input file needed.")
                except Exception as e_data_processing:
                    print(f"WARN: synthetic-datagen/main.py: Wavelet Workaround: Error during data check/truncation for '{original_real_data_file_path}': {e_data_processing}. Preprocessor will use original file path.")
            else:
                print(f"DEBUG main.py: Wavelet Workaround: 'real_data_file' path '{original_real_data_file_path}' not found or not specified. Skipping input file length adjustment.")
            
            print(f"DEBUG main.py: Final 'config_for_preprocessor_run' being passed to preprocessor (after potential file truncation): {config_for_preprocessor_run}")
            try:
                datasets = preprocessor_plugin.run_preprocessing(config=config_for_preprocessor_run)
            finally:
                if temp_data_file_to_delete:
                    try:
                        os.remove(temp_data_file_to_delete)
                        print(f"INFO: synthetic-datagen/main.py: Wavelet Workaround: Successfully removed temporary data file: {temp_data_file_to_delete}")
                    except OSError as e_remove:
                        print(f"WARN: synthetic-datagen/main.py: Wavelet Workaround: Failed to remove temporary data file '{temp_data_file_to_delete}': {e_remove}")
            
            print("PreprocessorPlugin.run_preprocessing finished.")

            # Extract the relevant processed real data, dates, and feature names
            # Adjust these keys based on what your preprocessor_plugin.run_preprocessing returns
            # For example, if your preprocessor prepares data specifically for this kind of evaluation,
            # it might return a single processed array, dates, and feature names directly,
            # or they might be under specific keys in the 'datasets' dictionary.
            
            # Scenario 1: Preprocessor's run_preprocessing is adapted for this use case and
            # returns a structure similar to the previous 'preprocess_for_evaluation'
            if "processed_eval_data" in datasets and "eval_dates" in datasets and "eval_feature_names" in datasets:
                X_real_processed = datasets["processed_eval_data"]
                real_dates = datasets["eval_dates"]
                real_feature_names = datasets["eval_feature_names"]
            # Scenario 2: Using "x_train" or similar from a more general preprocessor output
            # YOU MIGHT NEED TO ADJUST THESE KEYS AND LOGIC
            elif "x_train" in datasets:
                X_real_processed = datasets["x_train"]
                # Attempt to get corresponding dates and feature names
                # This assumes your preprocessor stores them in a way that can be aligned with x_train
                # For example, if 'x_train_dates' and 'feature_names' are provided:
                real_dates = datasets.get("x_train_dates") # Or "train_dates", "y_train_dates" etc.
                
                # Feature names might come from a general key or be inferred
                if "feature_names" in datasets:
                    real_feature_names = datasets["feature_names"]
                elif hasattr(X_real_processed, 'columns') and isinstance(X_real_processed, pd.DataFrame): # If it's a DataFrame
                    real_feature_names = list(X_real_processed.columns)
                    X_real_processed = X_real_processed.values # Convert to NumPy array for evaluator
                elif X_real_processed.ndim == 2: # If NumPy array, create generic feature names
                    real_feature_names = [f"feature_{i}" for i in range(X_real_processed.shape[1])]
                else:
                    raise ValueError("Could not determine feature names for the processed real data.")

                # Ensure X_real_processed is a NumPy array
                if not isinstance(X_real_processed, np.ndarray):
                    if hasattr(X_real_processed, 'values'): # e.g. Pandas DataFrame/Series
                        X_real_processed = X_real_processed.values
                    else:
                        try:
                            X_real_processed = np.array(X_real_processed)
                        except Exception as e_conv:
                            raise TypeError(f"Could not convert processed real data to NumPy array: {e_conv}")
            else:
                raise KeyError("Could not find 'x_train' or 'processed_eval_data' in the datasets returned by PreprocessorPlugin.run_preprocessing.")

            # Ensure X_real_processed is 2D (samples, features)
            if X_real_processed.ndim == 1:
                X_real_processed = X_real_processed.reshape(-1, 1)
            elif X_real_processed.ndim == 3:
                # If data is (samples, window_size, features) from preprocessor,
                # and generator produces (samples, features) representing the first tick,
                # you need to ensure X_real_processed is also (samples, features)
                # This might mean your preprocessor needs a specific mode for this evaluation,
                # or you extract the relevant part here.
                # Example: if X_real_processed is (samples, window, features) and you need first tick:
                # X_real_processed = X_real_processed[:, 0, :]
                print(f"Warning: X_real_processed has shape {X_real_processed.shape}. Assuming it's (samples, window, features) and taking [:, 0, :] for comparison.")
                X_real_processed = X_real_processed[:, 0, :] # Adjust if this assumption is wrong

            print(f"Real data extracted for evaluation. Shape: {X_real_processed.shape}, Number of dates: {len(real_dates) if real_dates is not None else 'N/A'}")

            # 1. Muestreo latente (usar solo n_samples; latent_dim ya fue seteado)
            n_synthetic_samples = config['n_samples']
            if X_real_processed.shape[0] < n_synthetic_samples and X_real_processed.shape[0] > 0 :
                print(f"Warning: Number of preprocessed real samples ({X_real_processed.shape[0]}) is less than requested synthetic samples ({n_synthetic_samples}). Adjusting synthetic samples to match real data length.")
                n_synthetic_samples = X_real_processed.shape[0]
            elif X_real_processed.shape[0] == 0:
                 print(f"Warning: Preprocessed real data has 0 samples. Using configured n_samples ({n_synthetic_samples}) for synthetic data.")

            if n_synthetic_samples == 0:
                raise ValueError("Cannot generate 0 synthetic samples. Check real data preprocessing or n_samples config.")

            Z = feeder_plugin.generate(n_synthetic_samples)
            
            X_syn = generator_plugin.generate(Z) # X_syn is (n_samples, features)
            
            if X_syn.shape[1] != X_real_processed.shape[1]:
                raise ValueError(
                    f"Feature mismatch after generation/preprocessing: "
                    f"Synthetic data has {X_syn.shape[1]} features, "
                    f"Preprocessed real data has {X_real_processed.shape[1]} features. "
                    f"Feature names from preprocessor: {real_feature_names}"
                )

            output_file = config['output_file']
            pd.DataFrame(X_syn, columns=real_feature_names if len(real_feature_names) == X_syn.shape[1] else None).to_csv(
                output_file, index=False
            )
            print(f"Synthetic data saved to {output_file}.")
            
            metrics = evaluator_plugin.evaluate(
                synthetic_data=X_syn,
                real_data_processed=X_real_processed,
                real_dates=real_dates,
                feature_names=real_feature_names,
                config=config
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
