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
from datetime import datetime, timedelta # Ensure these are imported

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


    # --- Helper function to generate DATE_TIME column ---
    def generate_datetime_column(start_datetime_str: str, num_samples: int, periodicity_str: str) -> list:
        """
        Generates a list of datetime strings based on start_datetime, num_samples, and periodicity.
        Skips weekends for non-daily periodicities. For daily, only includes weekdays.
        """
        date_times = []
        
        try:
            current_dt = pd.to_datetime(start_datetime_str)
        except Exception as e:
            print(f"Error parsing 'start_date_time' ('{start_datetime_str}'): {e}. DATE_TIME column will be empty.")
            return []

        time_delta_map = {
            "1h": timedelta(hours=1),
            "15min": timedelta(minutes=15),
            "1min": timedelta(minutes=1),
            "daily": timedelta(days=1)
            # Add other supported periodicities here
        }
        time_step = time_delta_map.get(periodicity_str)

        if not time_step:
            print(f"Warning: Unsupported 'dataset_periodicity' ('{periodicity_str}') for DATE_TIME generation. Column will be empty.")
            return []

        generated_count = 0
        
        # Initial adjustment for non-daily: if start_dt is weekend, move to next Monday 00:00
        if periodicity_str != "daily":
            while current_dt.weekday() >= 5: # 5 = Saturday, 6 = Sunday
                current_dt += timedelta(days=1)
                current_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        while generated_count < num_samples:
            if periodicity_str == "daily":
                if current_dt.weekday() < 5: # Monday to Friday
                    date_times.append(current_dt.strftime('%Y-%m-%d %H:%M:%S'))
                    generated_count += 1
                current_dt += time_step # Increment by one day
            else: # For hourly, minutely, etc.
                date_times.append(current_dt.strftime('%Y-%m-%d %H:%M:%S'))
                generated_count += 1
                current_dt += time_step
                # Skip to next weekday if current_dt lands on a weekend
                while current_dt.weekday() >= 5:
                    current_dt += timedelta(days=1)
                    current_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if len(date_times) > num_samples * 2 and num_samples > 0 : # Safety break for potential infinite loops with complex date logic
                print(f"Warning: DATE_TIME generation seems to be in a long loop. Breaking after generating {len(date_times)} dates for {num_samples} requested samples.")
                break
        
        if len(date_times) > num_samples: # Trim if overshot (can happen if last step before weekend skip was not needed)
            date_times = date_times[:num_samples]

        if len(date_times) != num_samples:
            print(f"Warning: Could not generate the exact number of timestamps for DATE_TIME. Expected {num_samples}, got {len(date_times)}. Check start_date and periodicity.")
            # Optionally, fill with NaNs or handle differently if exact match is critical
            # For now, it will save with fewer date_times if this happens.

        return date_times

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
            # Ensure all relevant dataset files have an even number of rows.
            
            print("DEBUG main.py: WORKAROUND 2: Starting process to ensure even row counts for relevant data files.")
            
            # List of config keys that might hold paths to data files used by the preprocessor.
            # Add any other keys your stl_preprocessor.py might use.
            data_file_keys = [
                'real_data_file', # Often a primary input before splits
                'x_train_file', 'y_train_file',
                'x_validation_file', 'y_validation_file', 'x_val_file', 'y_val_file', # Added common val aliases
                'x_test_file', 'y_test_file'
            ]
            
            temp_files_created_paths = [] # Keep track of temp files for cleanup

            for file_key in data_file_keys:
                original_file_path = config_for_preprocessor_run.get(file_key)
                
                if original_file_path and isinstance(original_file_path, str) and os.path.exists(original_file_path):
                    print(f"DEBUG main.py: WORKAROUND 2: Checking file for key '{file_key}': '{original_file_path}'")
                    try:
                        df_data = pd.read_csv(original_file_path)
                        data_len = len(df_data)
                        print(f"DEBUG main.py: WORKAROUND 2: Read '{original_file_path}', original length: {data_len}")
                        
                        if data_len > 0 and data_len % 2 != 0:
                            print(f"INFO: synthetic-datagen/main.py: WORKAROUND 2: Data in '{original_file_path}' (key: '{file_key}') has odd length ({data_len}). Truncating last row.")
                            df_data_truncated = df_data.iloc[:-1]
                            
                            if not df_data_truncated.empty:
                                with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix=f'_{file_key}.csv') as tmp_file_obj:
                                    df_data_truncated.to_csv(tmp_file_obj.name, index=False)
                                    temp_file_path = tmp_file_obj.name
                                temp_files_created_paths.append(temp_file_path)
                                config_for_preprocessor_run[file_key] = temp_file_path
                                print(f"INFO: synthetic-datagen/main.py: WORKAROUND 2: Preprocessor will use temporary even-length file for key '{file_key}': '{temp_file_path}'. New length: {len(df_data_truncated)}")
                            else:
                                print(f"WARN: synthetic-datagen/main.py: WORKAROUND 2: Original data in '{original_file_path}' (key: '{file_key}') had 1 row. Truncating made it empty. Preprocessor will use original file. Wavelet error may still occur for this file.")
                        elif data_len == 0:
                            print(f"INFO: synthetic-datagen/main.py: WORKAROUND 2: Data in '{original_file_path}' (key: '{file_key}') is empty. No truncation needed.")
                        else: # Even length
                            print(f"INFO: synthetic-datagen/main.py: WORKAROUND 2: Data in '{original_file_path}' (key: '{file_key}') has even length ({data_len}). No truncation needed.")
                    except Exception as e_data_processing:
                        print(f"WARN: synthetic-datagen/main.py: WORKAROUND 2: Error during data check/truncation for '{original_file_path}' (key: '{file_key}'): {e_data_processing}. Preprocessor will use original file path for this key.")
                elif original_file_path:
                    print(f"DEBUG main.py: WORKAROUND 2: File path for key '{file_key}' ('{original_file_path}') not found or not a string. Skipping.")
                # else:
                #    print(f"DEBUG main.py: WORKAROUND 2: Key '{file_key}' not found in config_for_preprocessor_run. Skipping.")

            print(f"DEBUG main.py: Final 'config_for_preprocessor_run' being passed to preprocessor (after potential file truncations for multiple keys): {config_for_preprocessor_run}")
            
            try:
                datasets = preprocessor_plugin.run_preprocessing(config=config_for_preprocessor_run)
            finally:
                if temp_files_created_paths:
                    print(f"INFO: synthetic-datagen/main.py: WORKAROUND 2: Cleaning up temporary files: {temp_files_created_paths}")
                    for temp_file_path in temp_files_created_paths:
                        try:
                            os.remove(temp_file_path)
                            print(f"INFO: synthetic-datagen/main.py: WORKAROUND 2: Successfully removed temporary file: {temp_file_path}")
                        except OSError as e_remove:
                            print(f"WARN: synthetic-datagen/main.py: WORKAROUND 2: Failed to remove temporary file '{temp_file_path}': {e_remove}")
            
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
            
            # --- Define Target Column Order ---
            TARGET_COLUMN_ORDER = [
                "DATE_TIME", "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
                "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI",
                "WilliamsR", "Momentum", "ROC", "OPEN", "HIGH", "LOW", "CLOSE",
                "BC-BO", "BH-BL", "BH-BO", "BO-BL", "S&P500_Close", "vix_close",
                "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
                "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
                "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
                "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8",
                "day_of_month", "hour_of_day", "day_of_week"
            ]

            # --- Determine initial datetime for DATE_TIME column generation ---
            start_datetime_str_for_generation = ""
            try:
                base_date_str = config.get("start_date", "2022-01-01") # Default to Jan 1, 2022
                base_dt = pd.to_datetime(base_date_str)
                print(f"DEBUG: Base date for DATE_TIME generation: {base_dt.strftime('%Y-%m-%d')}")

                if 'day_of_month' in real_feature_names and \
                   'hour_of_day' in real_feature_names and \
                   X_syn.shape[0] > 0:
                    print(f"DEBUG: Attempting to use day/hour from X_syn[0]. Features 'day_of_month', 'hour_of_day' found in real_feature_names. X_syn has {X_syn.shape[0]} samples.")
                    
                    day_of_month_idx = real_feature_names.index('day_of_month')
                    hour_of_day_idx = real_feature_names.index('hour_of_day')
                    
                    raw_day_val = X_syn[0, day_of_month_idx]
                    raw_hour_val = X_syn[0, hour_of_day_idx]
                    print(f"DEBUG: Raw X_syn[0] values for day_of_month: {raw_day_val}, hour_of_day: {raw_hour_val}")

                    day_val = pd.NA # Using pandas NA for missing/unconvertible
                    hour_val = pd.NA

                    try:
                        if pd.notna(raw_day_val): day_val = int(float(raw_day_val)) # Convert to float first, then int
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert X_syn day_of_month value '{raw_day_val}' to int.")
                    
                    try:
                        if pd.notna(raw_hour_val): hour_val = int(float(raw_hour_val))
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert X_syn hour_of_day value '{raw_hour_val}' to int.")

                    print(f"DEBUG: Converted X_syn[0] day_val: {day_val}, hour_val: {hour_val}")

                    # Validate day and hour values
                    if pd.notna(day_val) and (1 <= day_val <= 31):
                        # Day is plausible, use it
                        pass
                    else:
                        print(f"Warning: Extracted day_of_month ({day_val}) from synthetic data is invalid, NA, or out of range (1-31). Using day from base_date: {base_dt.day}.")
                        day_val = base_dt.day # Fallback to day from base_dt (e.g., 1st of the month)
                    
                    if pd.notna(hour_val) and (0 <= hour_val <= 23):
                        # Hour is plausible, use it
                        pass
                    else:
                        print(f"Warning: Extracted hour_of_day ({hour_val}) from synthetic data is invalid, NA, or out of range (0-23). Using hour 0.")
                        hour_val = 0 # Fallback to hour 0
                    
                    try:
                        specific_start_dt = datetime(base_dt.year, base_dt.month, day_val, hour_val, 0, 0)
                        start_datetime_str_for_generation = specific_start_dt.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"DEBUG: Constructed start_datetime for generation (from X_syn day/hour logic): {start_datetime_str_for_generation}")
                    except ValueError as e_dt:
                        print(f"Warning: Could not construct specific start datetime (Year: {base_dt.year}, Month: {base_dt.month}, Day: {day_val}, Hour: {hour_val}): {e_dt}. Falling back to configured/default start_date at 00:00.")
                        start_datetime_str_for_generation = base_dt.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"DEBUG: Fallback start_datetime (ValueError during construction): {start_datetime_str_for_generation}")
                else:
                    missing_info = []
                    if 'day_of_month' not in real_feature_names: missing_info.append("'day_of_month' not in real_feature_names")
                    if 'hour_of_day' not in real_feature_names: missing_info.append("'hour_of_day' not in real_feature_names")
                    if X_syn.shape[0] == 0: missing_info.append("X_syn is empty")
                    
                    print(f"Warning: Cannot use day/hour from synthetic data ({'; '.join(missing_info)}). Using configured/default start_date '{base_date_str}' at 00:00 for DATE_TIME generation.")
                    start_datetime_str_for_generation = base_dt.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"DEBUG: Fallback start_datetime (features unavailable or X_syn empty): {start_datetime_str_for_generation}")

            except Exception as e_init_dt:
                print(f"CRITICAL Error determining initial datetime: {e_init_dt}. Defaulting to current system time for DATE_TIME generation as a last resort.")
                start_datetime_str_for_generation = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"DEBUG: Fallback start_datetime (exception in outer try): {start_datetime_str_for_generation}")

            # --- Generate DATE_TIME column values ---
            datetime_column_values = [pd.NaT] * n_synthetic_samples
            dataset_periodicity_str = config.get("dataset_periodicity")

            if dataset_periodicity_str and start_datetime_str_for_generation:
                print(f"DEBUG: Attempting to generate DATE_TIME column for {n_synthetic_samples} samples, starting from '{start_datetime_str_for_generation}' with periodicity '{dataset_periodicity_str}'.")
                generated_dates = generate_datetime_column(
                    start_datetime_str_for_generation,
                    n_synthetic_samples,
                    dataset_periodicity_str
                )
                if generated_dates and len(generated_dates) == n_synthetic_samples:
                    datetime_column_values = generated_dates
                    print(f"DEBUG: DATE_TIME column generated successfully with {len(datetime_column_values)} entries.")
                else:
                    print(f"Warning: DATE_TIME column generation did not produce the expected {n_synthetic_samples} valid entries (got {len(generated_dates) if generated_dates else 0}). The DATE_TIME column will contain placeholders (NaT). Review messages from date generation function.")
            else:
                if not dataset_periodicity_str: print("Information: 'dataset_periodicity' not found in config. DATE_TIME column will contain placeholders (NaT).")
                if not start_datetime_str_for_generation: print("Information: Could not determine a valid start_datetime_str for generation. DATE_TIME column will contain placeholders (NaT).")
            
            # --- Create DataFrame with synthetic data and feature names from preprocessor ---
            if len(real_feature_names) != X_syn.shape[1]:
                print(f"CRITICAL WARNING: Mismatch between number of real_feature_names ({len(real_feature_names)}) and X_syn columns ({X_syn.shape[1]}). DataFrame creation might be incorrect or fail. Feature names from preprocessor: {real_feature_names}")
            
            df_synthetic_data = pd.DataFrame(X_syn, columns=real_feature_names)
            print(f"DEBUG: df_synthetic_data created. Columns: {list(df_synthetic_data.columns)}")

            # --- Add DATE_TIME column as the first column ---
            df_synthetic_data.insert(0, "DATE_TIME", datetime_column_values)
            print(f"DEBUG: DATE_TIME column inserted. Current columns: {list(df_synthetic_data.columns)}")
            if datetime_column_values:
                 print(f"DEBUG: First few DATE_TIME values after insertion: {datetime_column_values[:5]}...")
            else:
                 print(f"DEBUG: datetime_column_values is empty or None after generation attempt.")


            # --- Reorder columns to the specified TARGET_COLUMN_ORDER ---
            print(f"DEBUG: Reordering columns to TARGET_COLUMN_ORDER. Target: {TARGET_COLUMN_ORDER}")
            
            # Ensure all columns in TARGET_COLUMN_ORDER exist in the DataFrame, adding NaNs if not.
            # Then select them in the specified order.
            df_data_to_save = pd.DataFrame() # Start with an empty DataFrame
            for col_name in TARGET_COLUMN_ORDER:
                if col_name in df_synthetic_data:
                    df_data_to_save[col_name] = df_synthetic_data[col_name]
                else:
                    df_data_to_save[col_name] = np.nan # Or pd.NaT for DATE_TIME if it was truly missing
                    if col_name == "DATE_TIME" and not ("DATE_TIME" in df_synthetic_data): # Should not happen due to insert
                         print(f"DEBUG: DATE_TIME column was not in df_synthetic_data during reorder; adding as NaN/NaT.")
                    else:
                         print(f"DEBUG: Column '{col_name}' from TARGET_COLUMN_ORDER not found in generated data; adding as NaN.")
            
            # Verify final column order
            final_columns = list(df_data_to_save.columns)
            if final_columns != TARGET_COLUMN_ORDER:
                print(f"CRITICAL WARNING: Final column order {final_columns} does not match TARGET_COLUMN_ORDER {TARGET_COLUMN_ORDER}. Reindexing might have failed.")
            else:
                print(f"DEBUG: df_data_to_save after reordering. Columns: {final_columns}")
            
            # --- Save to CSV ---
            df_data_to_save.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            print(f"Synthetic data saved to {output_file}.")
            
            metrics = evaluator_plugin.evaluate(
                synthetic_data=X_syn, # Pass original X_syn without datetime for numeric evaluation
                real_data_processed=X_real_processed,
                real_dates=real_dates,
                feature_names=real_feature_names, # Numeric feature names
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
