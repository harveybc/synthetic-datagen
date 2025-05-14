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
    if hasattr(preprocessor_plugin, 'plugin_params'):
        config = merge_config(config, preprocessor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)


    # --- Helper function to generate DATE_TIME column ---
    def generate_datetime_column(start_datetime_str: str, num_samples: int, periodicity_str: str) -> list:
        """
        Generates a list of datetime strings. Forcefully ensures num_samples valid date strings are returned.
        Skips weekends for non-daily periodicities, attempting to preserve the time of day.
        For daily periodicity, only includes weekdays, using the time from start_datetime_str.
        """
        date_times_objs = [] # Store datetime objects
        if num_samples == 0:
            return []

        try:
            current_dt = pd.to_datetime(start_datetime_str)
            # print(f"DEBUG generate_datetime_column: Initial current_dt: {current_dt}, type: {type(current_dt)}")
        except Exception as e:
            print(f"ERROR generate_datetime_column: Error parsing 'start_datetime_str' ('{start_datetime_str}'): {e}. Defaulting to current system time.")
            current_dt = pd.to_datetime(datetime.now().replace(microsecond=0))

        time_delta_map = {
            "1h": timedelta(hours=1), "1H": timedelta(hours=1),
            "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
            "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
            "daily": timedelta(days=1), "1D": timedelta(days=1)
            # Add other supported periodicities here
        }
        time_step = time_delta_map.get(periodicity_str)

        original_periodicity_str = periodicity_str # Keep for reference
        if not time_step:
            print(f"WARNING generate_datetime_column: Unsupported 'dataset_periodicity' ('{original_periodicity_str}'). Defaulting to hourly ('1h').")
            time_step = timedelta(hours=1)
            periodicity_str = "1h" # Update for internal logic

        generated_count = 0
        initial_hour, initial_minute, initial_second = current_dt.hour, current_dt.minute, current_dt.second

        if periodicity_str != "daily":
            if current_dt.weekday() >= 5: # 5 = Saturday, 6 = Sunday
                # print(f"DEBUG generate_datetime_column: Initial start_datetime {current_dt} is on a weekend. Adjusting.")
                while current_dt.weekday() >= 5:
                    current_dt += timedelta(days=1)
                current_dt = current_dt.replace(hour=initial_hour, minute=initial_minute, second=initial_second, microsecond=0)
                # print(f"DEBUG generate_datetime_column: Adjusted start_datetime to {current_dt} (next weekday at original time).")

        loop_iterations = 0
        # Increased max_loop_iterations to be more generous, especially with weekend skipping
        max_loop_iterations = num_samples * 7 if num_samples > 0 else 100

        while generated_count < num_samples:
            loop_iterations += 1
            if loop_iterations > max_loop_iterations:
                print(f"WARNING generate_datetime_column: Exceeded max_loop_iterations ({max_loop_iterations}) while generating dates. Generated {generated_count}/{num_samples}. Will forcefully complete.")
                break

            if periodicity_str == "daily":
                if current_dt.weekday() < 5:
                    date_times_objs.append(current_dt.replace(hour=initial_hour, minute=initial_minute, second=initial_second, microsecond=0))
                    generated_count += 1
                current_dt += time_step
            else: # For hourly, minutely, etc.
                date_times_objs.append(current_dt)
                generated_count += 1
                if generated_count >= num_samples: break
                
                current_dt += time_step
                if current_dt.weekday() >= 5:
                    target_h, target_m, target_s = current_dt.hour, current_dt.minute, current_dt.second
                    # print(f"DEBUG generate_datetime_column: Incremented to {current_dt} (weekend). Adjusting.")
                    while current_dt.weekday() >= 5:
                        current_dt += timedelta(days=1)
                    current_dt = current_dt.replace(hour=target_h, minute=target_m, second=target_s, microsecond=0)
                    # print(f"DEBUG generate_datetime_column: Adjusted incremented time to {current_dt}.")
        
        # Forcefully complete if not enough dates were generated
        if generated_count < num_samples:
            print(f"INFO generate_datetime_column: Loop generated {generated_count}/{num_samples} dates. Forcefully completing remaining dates.")
            # Determine the last valid datetime object to continue from, or use current_dt
            if date_times_objs:
                last_dt_obj = date_times_objs[-1]
            else: # If date_times_objs is empty, current_dt is the starting point for filling
                last_dt_obj = current_dt - time_step # current_dt would be the *next* one, so step back once

            # Ensure last_dt_obj is a valid datetime before proceeding
            if not isinstance(last_dt_obj, (datetime, pd.Timestamp)):
                 print(f"CRITICAL WARNING generate_datetime_column: last_dt_obj for fill is invalid ({last_dt_obj}). Resetting to current time.")
                 last_dt_obj = pd.to_datetime(datetime.now().replace(microsecond=0))


            while generated_count < num_samples:
                last_dt_obj += time_step # Increment from the last known good datetime

                # Apply weekend skipping for the fill part as well for consistency
                if periodicity_str != "daily":
                    if last_dt_obj.weekday() >= 5:
                        target_h, target_m, target_s = last_dt_obj.hour, last_dt_obj.minute, last_dt_obj.second
                        while last_dt_obj.weekday() >= 5:
                            last_dt_obj += timedelta(days=1)
                        last_dt_obj = last_dt_obj.replace(hour=target_h, minute=target_m, second=target_s, microsecond=0)
                    date_times_objs.append(last_dt_obj)
                    generated_count += 1
                elif periodicity_str == "daily":
                    # For daily, keep advancing until a weekday is found
                    while last_dt_obj.weekday() >= 5:
                        last_dt_obj += time_step # time_step is 1 day
                    # Apply initial time for daily
                    date_times_objs.append(last_dt_obj.replace(hour=initial_hour, minute=initial_minute, second=initial_second, microsecond=0))
                    generated_count += 1
                
                if generated_count >= num_samples: break


        # Ensure the list is exactly num_samples long, truncating if somehow over (shouldn't happen with this logic)
        if len(date_times_objs) > num_samples:
            date_times_objs = date_times_objs[:num_samples]
        
        # If still not enough (highly unlikely now), fill with emergency datetimes
        emergency_fill_count = 0
        while len(date_times_objs) < num_samples:
            emergency_fill_count +=1
            emergency_dt_val = (pd.to_datetime(datetime.now().replace(microsecond=0)) + timedelta(hours=len(date_times_objs)))
            date_times_objs.append(emergency_dt_val)
        if emergency_fill_count > 0:
            print(f"CRITICAL WARNING generate_datetime_column: Had to emergency fill {emergency_fill_count} dates. This indicates a flaw in generation logic.")


        # Convert all datetime objects to string representation
        output_dates_str = []
        for dt_obj in date_times_objs:
            if isinstance(dt_obj, (datetime, pd.Timestamp)):
                output_dates_str.append(dt_obj.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                # This is an absolute last resort if an item is not a datetime object
                print(f"CRITICAL ERROR generate_datetime_column: Non-datetime object '{dt_obj}' (type: {type(dt_obj)}) found in list. Using emergency string.")
                fallback_dt_str = (pd.to_datetime(datetime.now().replace(microsecond=0)) + timedelta(seconds=len(output_dates_str))).strftime('%Y-%m-%d %H:%M:%S')
                output_dates_str.append(fallback_dt_str)
        
        # Final check on counts
        if len(output_dates_str) != num_samples and num_samples > 0:
            print(f"CRITICAL ERROR generate_datetime_column: Final date string list length ({len(output_dates_str)}) does not match num_samples ({num_samples}).")
            # Pad with emergency strings if short, truncate if long
            while len(output_dates_str) < num_samples:
                output_dates_str.append((pd.to_datetime(datetime.now().replace(microsecond=0)) + timedelta(seconds=len(output_dates_str))).strftime('%Y-%m-%d %H:%M:%S'))
            if len(output_dates_str) > num_samples:
                output_dates_str = output_dates_str[:num_samples]

        # print(f"DEBUG generate_datetime_column: Generated {len(output_dates_str)} date strings. First few: {output_dates_str[:5]}")
        return output_dates_str

    # --- Helper function to generate DATE_TIME values for synthetic data ---
    def generate_synthetic_datetimes_before_real(
        real_start_dt: pd.Timestamp,
        num_synthetic_samples: int,
        time_delta_val: timedelta,
        periodicity_str_val: str
    ) -> list[pd.Timestamp]:
        """
        Generates a list of 'num_synthetic_samples' datetime objects in chronological order,
        ending such that the next tick after the last synthetic datetime would lead into 'real_start_dt'.
        Skips weekends.
        """
        if num_synthetic_samples == 0:
            return []

        generated_datetimes_reversed = [] # Store in reverse chronological order first
        current_reference_dt = real_start_dt

        # Determine the target time of day for 'daily' periodicity from real_start_dt
        daily_target_time = None
        if periodicity_str_val == "daily":
            daily_target_time = real_start_dt.time()

        for _ in range(num_synthetic_samples):
            # Calculate the datetime one tick before current_reference_dt
            prev_dt_candidate = current_reference_dt - time_delta_val

            if periodicity_str_val == "daily":
                # For daily, ensure the time component matches daily_target_time
                # and that the date is a weekday.
                prev_dt_candidate = prev_dt_candidate.replace(
                    hour=daily_target_time.hour,
                    minute=daily_target_time.minute,
                    second=daily_target_time.second,
                    microsecond=0
                )
                while prev_dt_candidate.weekday() >= 5: # Monday is 0, Saturday is 5, Sunday is 6
                    prev_dt_candidate -= timedelta(days=1)
                    # Re-apply target time after day change
                    prev_dt_candidate = prev_dt_candidate.replace(
                        hour=daily_target_time.hour,
                        minute=daily_target_time.minute,
                        second=daily_target_time.second,
                        microsecond=0
                    )
            else: # For hourly, minutely, etc.
                # Preserve the time of day from the simple subtraction,
                # then adjust the date part if it falls on a weekend.
                target_h, target_m, target_s = prev_dt_candidate.hour, prev_dt_candidate.minute, prev_dt_candidate.second
                while prev_dt_candidate.weekday() >= 5:
                    prev_dt_candidate -= timedelta(days=1) # Move to the previous day
                # After finding a weekday, set the time to what it was after the initial timedelta subtraction.
                prev_dt_candidate = prev_dt_candidate.replace(hour=target_h, minute=target_m, second=target_s, microsecond=0)

            generated_datetimes_reversed.append(prev_dt_candidate)
            current_reference_dt = prev_dt_candidate # The next iteration calculates the tick before this new dt

        return list(reversed(generated_datetimes_reversed)) # Return in chronological order

    # --- (The old generate_datetime_column function can be removed or kept if used elsewhere) ---

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
            # This part remains largely the same as it prepares 'datasets', 'X_real_processed', 
            # 'real_dates', and 'real_feature_names' which are crucial.
            print("Preprocessing real data for evaluation via PreprocessorPlugin...")
            
            if not hasattr(preprocessor_plugin, 'run_preprocessing'):
                raise AttributeError("PreprocessorPlugin does not have a 'run_preprocessing' method.")
            
            config_for_preprocessor_run = config.copy()
            # ... (Keep your existing WORKAROUND 1 and WORKAROUND 2 logic for config_for_preprocessor_run here) ...
            # WORKAROUND 1: For "Baseline train indices invalid"
            if not config_for_preprocessor_run.get('use_stl', False): 
                if 'stl_window' in preprocessor_plugin.plugin_params or 'stl_window' in config_for_preprocessor_run:
                    original_stl_window = config_for_preprocessor_run.get('stl_window')
                    config_for_preprocessor_run['stl_window'] = 1 
                    print(f"INFO: synthetic-datagen/main.py: WORKAROUND 1: 'use_stl' is False. Original 'stl_window': {original_stl_window}. Temporarily setting 'stl_window' to 1 to prevent 'Baseline train indices invalid' error.")
            else:
                print(f"DEBUG main.py: WORKAROUND 1: 'use_stl' is True or not set, 'stl_window' workaround not applied.")

            # WORKAROUND 2: For "Wavelet: Length of data must be even"
            print("DEBUG main.py: WORKAROUND 2: Starting process to ensure even row counts for relevant data files.")
            data_file_keys = [
                'real_data_file', 'x_train_file', 'y_train_file',
                'x_validation_file', 'y_validation_file', 'x_val_file', 'y_val_file',
                'x_test_file', 'y_test_file'
            ]
            temp_files_created_paths = [] 
            for file_key in data_file_keys:
                original_file_path = config_for_preprocessor_run.get(file_key)
                if original_file_path and isinstance(original_file_path, str) and os.path.exists(original_file_path):
                    try:
                        df_data = pd.read_csv(original_file_path)
                        data_len = len(df_data)
                        if data_len > 0 and data_len % 2 != 0:
                            df_data_truncated = df_data.iloc[:-1]
                            if not df_data_truncated.empty:
                                with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix=f'_{file_key}.csv') as tmp_file_obj:
                                    df_data_truncated.to_csv(tmp_file_obj.name, index=False)
                                    temp_file_path = tmp_file_obj.name
                                temp_files_created_paths.append(temp_file_path)
                                config_for_preprocessor_run[file_key] = temp_file_path
                                print(f"INFO: WORKAROUND 2: Using temp even-length file for '{file_key}': '{temp_file_path}'. New length: {len(df_data_truncated)}")
                            # ... (rest of your print statements for this workaround) ...
                    except Exception as e_data_processing:
                        print(f"WARN: WORKAROUND 2: Error for '{original_file_path}' (key: '{file_key}'): {e_data_processing}.")
            
            try:
                datasets = preprocessor_plugin.run_preprocessing(config=config_for_preprocessor_run)
            finally:
                if temp_files_created_paths: # Cleanup temp files
                    for temp_file_path in temp_files_created_paths:
                        try: os.remove(temp_file_path)
                        except OSError as e_remove: print(f"WARN: Failed to remove temp file '{temp_file_path}': {e_remove}")
            
            print("PreprocessorPlugin.run_preprocessing finished.")

            # Extract feature names from preprocessor output (primarily for evaluation purposes)
            if "feature_names" in datasets:
                eval_feature_names = datasets["feature_names"] # Renamed from real_feature_names
            elif hasattr(datasets.get("x_train"), 'columns') and isinstance(datasets.get("x_train"), pd.DataFrame):
                eval_feature_names = list(datasets.get("x_train").columns) # Renamed
            elif datasets.get("x_train") is not None and datasets.get("x_train").ndim == 2:
                eval_feature_names = [f"feature_{i}" for i in range(datasets.get("x_train").shape[1])] # Renamed
            else: # Fallback if preprocessor doesn't provide clear feature names
                print("WARNING: Could not reliably get 'eval_feature_names' from preprocessor output. Attempting to use TARGET_COLUMN_ORDER excluding DATE_TIME.")
                # Ensure TARGET_COLUMN_ORDER is defined or accessible here if used as a fallback
                # For safety, define it before this block or ensure it's globally available.
                # temp_target_cols = [col for col in TARGET_COLUMN_ORDER if col != "DATE_TIME"] 
                # if not temp_target_cols: raise ValueError("TARGET_COLUMN_ORDER is not defined or only contains DATE_TIME.")
                # eval_feature_names = temp_target_cols # Renamed
                # A safer fallback if TARGET_COLUMN_ORDER is problematic here:
                if datasets.get("x_train") is not None: # Check if x_train exists
                     print("ERROR: Could not determine eval_feature_names and x_train shape is not helpful. Exiting.")
                     sys.exit(1)
                else: # If x_train doesn't exist, this is a deeper issue with preprocessor output
                     print("ERROR: 'x_train' not found in datasets from preprocessor. Cannot determine eval_feature_names. Exiting.")
                     sys.exit(1)

            initial_preprocessor_feature_names = list(eval_feature_names) # ADD THIS LINE
            print(f"DEBUG: Stored initial_preprocessor_feature_names: {initial_preprocessor_feature_names}")

            # Ensure X_real_processed is correctly shaped for evaluation (as in your existing code)
            X_real_processed = datasets.get("x_train") # Or your specific key for processed evaluation data
            if X_real_processed is None: raise KeyError("Processed real data for evaluation (e.g., 'x_train') not found in 'datasets'.")
            if not isinstance(X_real_processed, np.ndarray): X_real_processed = np.array(X_real_processed)
            if X_real_processed.ndim == 1: X_real_processed = X_real_processed.reshape(-1, 1)
            elif X_real_processed.ndim == 3: X_real_processed = X_real_processed[:, 0, :] # Assuming (samples, window, features) -> (samples, features)

            # --- Load Real Data for Concatenation ---
            real_data_filepath_for_concat = config.get("real_data_file")
            if not real_data_filepath_for_concat or not os.path.exists(real_data_filepath_for_concat):
                print(f"ERROR: Main real data file '{real_data_filepath_for_concat}' for concatenation not found. Exiting.")
                sys.exit(1)
            try:
                # 1. Load FULL real data
                real_df_for_concat_full = pd.read_csv(real_data_filepath_for_concat)
                if real_df_for_concat_full.empty:
                    print(f"ERROR: Real data file '{real_data_filepath_for_concat}' is empty. Cannot proceed. Exiting.")
                    sys.exit(1)

                # 2. Handle DATE_TIME column and sort for the FULL data
                if 'DATE_TIME' not in real_df_for_concat_full.columns:
                    if real_df_for_concat_full.shape[1] > 0 and \
                       (pd.api.types.is_datetime64_any_dtype(real_df_for_concat_full.iloc[:, 0]) or \
                        real_df_for_concat_full.iloc[:, 0].astype(str).str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?').all()):
                        real_df_for_concat_full.rename(columns={real_df_for_concat_full.columns[0]: 'DATE_TIME'}, inplace=True)
                    else:
                        print(f"ERROR: 'DATE_TIME' column not found in '{real_data_filepath_for_concat}'. Exiting.")
                        sys.exit(1)
                real_df_for_concat_full['DATE_TIME'] = pd.to_datetime(real_df_for_concat_full['DATE_TIME'])
                real_df_for_concat_full.sort_values('DATE_TIME', inplace=True)
                
                # 3. Define output_feature_names from the FULL real data
                output_feature_names = [col for col in real_df_for_concat_full.columns if col != 'DATE_TIME']
                print(f"DEBUG: output_feature_names (for CSV, from full '{real_data_filepath_for_concat}'): {output_feature_names}")

                # 4. Slice the real data for concatenation based on max_steps_train
                real_df_for_concat = real_df_for_concat_full # Default to full if no slicing
                max_steps_train_config = config.get("max_steps_train")
                
                if max_steps_train_config is not None:
                    try:
                        num_real_rows_to_keep = int(max_steps_train_config)
                        if num_real_rows_to_keep > 0:
                            if len(real_df_for_concat_full) > num_real_rows_to_keep:
                                print(f"INFO: Slicing real data for concatenation to its last {num_real_rows_to_keep} rows (original real data length: {len(real_df_for_concat_full)}).")
                                real_df_for_concat = real_df_for_concat_full.iloc[-num_real_rows_to_keep:].copy()
                                real_df_for_concat.reset_index(drop=True, inplace=True)
                            else:
                                print(f"INFO: max_steps_train ({num_real_rows_to_keep}) is >= length of real data ({len(real_df_for_concat_full)}). Using all available real data for concatenation.")
                        elif num_real_rows_to_keep <= 0:
                            print(f"INFO: max_steps_train ({num_real_rows_to_keep}) is <= 0. Using all available real data for concatenation (no slicing).")
                    except ValueError:
                        print(f"WARNING: Could not convert max_steps_train ('{max_steps_train_config}') to int. Using all available real data for concatenation (no slicing).")
                
                if real_df_for_concat.empty: # Check after potential slicing
                    print(f"ERROR: Real data for concatenation ('{real_data_filepath_for_concat}') is effectively empty after slicing. Cannot determine start datetime. Exiting.")
                    sys.exit(1)
                real_data_first_datetime_for_concat = real_df_for_concat['DATE_TIME'].iloc[0]
                print(f"DEBUG: real_data_first_datetime_for_concat (for synthetic generation, from (potentially sliced) real data): {real_data_first_datetime_for_concat}")

            except Exception as e:
                print(f"ERROR: Loading or processing real data for concatenation from '{real_data_filepath_for_concat}': {e}. Exiting.")
                sys.exit(1)

            # --- Determine Periodicity and Time Step (re-ensure for clarity) ---
            dataset_periodicity_str = config.get("dataset_periodicity", "1h")
            time_delta_map = {
                "1h": timedelta(hours=1), "1H": timedelta(hours=1),
                "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
                "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
                "daily": timedelta(days=1), "1D": timedelta(days=1)
            }
            time_step = time_delta_map.get(dataset_periodicity_str)
            if not time_step:
                print(f"WARNING: Unsupported 'dataset_periodicity' ('{dataset_periodicity_str}'). Defaulting to hourly ('1h').")
                time_step = timedelta(hours=1)
                dataset_periodicity_str = "1h"

            # --- Generate Synthetic Data for Output File ---
            n_synthetic_samples = config.get('n_samples', 0)
            if n_synthetic_samples <= 0:
                print("INFO: n_samples is 0. No synthetic data will be generated for the output file.")
                df_synthetic_for_output = pd.DataFrame()
                X_syn_raw_preprocessed_space = None # No raw synthetic data generated
            else:
                synthetic_datetime_objects = generate_synthetic_datetimes_before_real(
                    real_data_first_datetime_for_concat,
                    n_synthetic_samples,
                    time_step,
                    dataset_periodicity_str # Pass the determined periodicity
                )
                if not synthetic_datetime_objects or len(synthetic_datetime_objects) != n_synthetic_samples:
                    print(f"ERROR: Failed to generate synthetic datetimes. Expected {n_synthetic_samples}, got {len(synthetic_datetime_objects)}. Exiting.")
                    sys.exit(1)

                Z_for_output = feeder_plugin.generate(n_synthetic_samples)
                # X_syn_raw_preprocessed_space contains all features generated by the model (e.g., 54 preprocessed features)
                X_syn_raw_preprocessed_space = generator_plugin.generate(Z_for_output)

                #// ...existing code...
                print(f"DEBUG: Generator produced X_syn_raw_preprocessed_space with shape {X_syn_raw_preprocessed_space.shape} (matches eval_feature_names count).")

                # Create a DataFrame of the raw synthetic data with the preprocessed feature names
                df_synthetic_raw_full_preprocessed = pd.DataFrame(X_syn_raw_preprocessed_space, columns=eval_feature_names)

                # --- BEGIN PATCH: Calculate 'CLOSE' if needed for output and not present in generated preprocessed data ---
                # output_feature_names is the list of columns desired for the final CSV output.
                # df_synthetic_raw_full_preprocessed.columns currently reflects the initially generated features (eval_feature_names).
                if 'CLOSE' in output_feature_names and 'CLOSE' not in df_synthetic_raw_full_preprocessed.columns:
                    print("INFO: 'CLOSE' is required for output CSV but not directly generated. Attempting to calculate from 'BC-BO' and 'OPEN'.")
                    if 'BC-BO' in df_synthetic_raw_full_preprocessed.columns and 'OPEN' in df_synthetic_raw_full_preprocessed.columns:
                        df_synthetic_raw_full_preprocessed['CLOSE'] = df_synthetic_raw_full_preprocessed['BC-BO'] + df_synthetic_raw_full_preprocessed['OPEN']
                        print("INFO: Successfully calculated and added 'CLOSE' column to df_synthetic_raw_full_preprocessed.")
                        
                        # CRITICAL: Update eval_feature_names to reflect the new set of columns in the DataFrame.
                        # This ensures consistency for subsequent operations that rely on eval_feature_names
                        # to describe the columns of the (potentially modified) synthetic data.
                        eval_feature_names = df_synthetic_raw_full_preprocessed.columns.tolist()
                        print(f"DEBUG: eval_feature_names updated to: {eval_feature_names}")

                        # CRITICAL: Also update X_syn_raw_preprocessed_space if it's used later with the new eval_feature_names
                        # This is important for the evaluator_plugin.evaluate call.
                        X_syn_raw_preprocessed_space = df_synthetic_raw_full_preprocessed.to_numpy()
                        print(f"DEBUG: X_syn_raw_preprocessed_space updated to shape {X_syn_raw_preprocessed_space.shape} to include 'CLOSE'.")
                    else:
                        # If 'CLOSE' is required for output but cannot be calculated, the subsequent check
                        # for missing_original_in_preprocessed will catch it and exit, which is correct.
                        print("WARNING: 'CLOSE' is required for output, but 'BC-BO' or 'OPEN' (or both) are missing from generated features. 'CLOSE' cannot be calculated.")
                # --- END PATCH ---

                #// ...existing code...
                # Now, eval_feature_names and df_synthetic_raw_full_preprocessed.columns are consistent,
                # and X_syn_raw_preprocessed_space (if modified) matches this new structure.

                # Select only the columns that correspond to the original features desired in the output CSV
                # These selected features are still in their generated (potentially scaled/normalized) form.
                # Ensure all output_feature_names are present in the (potentially augmented) eval_feature_names
                missing_original_in_preprocessed = [name for name in output_feature_names if name not in eval_feature_names]
                if missing_original_in_preprocessed:
                    # This error will now correctly trigger if 'CLOSE' was in output_feature_names,
                    # was NOT in the original eval_feature_names, AND could NOT be calculated (e.g. BC-BO or OPEN missing).
                    print(f"ERROR: The following original output features are not found in the preprocessed feature list (eval_feature_names): {missing_original_in_preprocessed}. Cannot select them from generated data. This might be because 'CLOSE' was required but could not be calculated due to missing 'BC-BO' or 'OPEN'.")
                    sys.exit(1) # Exit if essential columns for output are missing

                X_syn_selected_original_scaled = df_synthetic_raw_full_preprocessed[output_feature_names].values
                print(f"DEBUG: Selected original features (scaled) X_syn_selected_original_scaled with shape {X_syn_selected_original_scaled.shape}.")


                # This will be the data for the synthetic part of the output CSV, ideally de-scaled.
                X_syn_for_output_final = X_syn_selected_original_scaled 
                try:
                    if hasattr(preprocessor_plugin, 'inverse_transform_synthetic_data'):
                        # This method should be designed to take the selected *original* features (but still scaled)
                        # and de-scale them using the appropriate scalers from 'datasets'.
                        # It should operate on data with 'len(output_feature_names)' columns.
                        # It might need 'output_feature_names' to correctly identify which scalers to use from 'datasets'.
                        print(f"DEBUG: Attempting to call inverse_transform_synthetic_data with data of shape {X_syn_selected_original_scaled.shape} and target_features: {output_feature_names}")
                        X_syn_for_output_final = preprocessor_plugin.inverse_transform_synthetic_data(
                            X_syn_selected_original_scaled, 
                            datasets,
                            target_features=output_feature_names 
                        )
                        print("INFO: Called preprocessor_plugin.inverse_transform_synthetic_data on selected original features.")
                        if X_syn_for_output_final.shape[1] != len(output_feature_names):
                            print(f"ERROR: inverse_transform_synthetic_data changed column count from {len(output_feature_names)} to {X_syn_for_output_final.shape[1]}. Reverting to using scaled selected features for output.")
                            X_syn_for_output_final = X_syn_selected_original_scaled 
                    else:
                        print(f"WARNING: PreprocessorPlugin '{preprocessor_plugin.__class__.__name__}' has no 'inverse_transform_synthetic_data' method. Using selected (but potentially scaled/normalized) original features for the output CSV.")
                except Exception as e_inv:
                    print(f"WARNING: Error during preprocessor_plugin.inverse_transform_synthetic_data: {e_inv}. Using selected (but potentially scaled/normalized) original features for the output CSV.")
                    X_syn_for_output_final = X_syn_selected_original_scaled 

                # Final check for the data intended for the CSV output
                if X_syn_for_output_final.shape[1] != len(output_feature_names):
                    print(f"ERROR: Final synthetic data for output (X_syn_for_output_final) has {X_syn_for_output_final.shape[1]} features, but expected {len(output_feature_names)} (original features). This indicates an issue after selection and attempted inverse transform.")
                    sys.exit(1)
                
                df_synthetic_for_output = pd.DataFrame(X_syn_for_output_final, columns=output_feature_names)
                df_synthetic_for_output.insert(0, "DATE_TIME", synthetic_datetime_objects)

            # --- Prepare Real Data for Concatenation ---
            aligned_real_columns = ['DATE_TIME'] + output_feature_names # Use output_feature_names
            
            missing_cols_in_real_concat = [col for col in output_feature_names if col not in real_df_for_concat.columns]
            if missing_cols_in_real_concat: 
                print(f"INTERNAL LOGIC ERROR: Columns {missing_cols_in_real_concat} (derived from real_df_for_concat) are somehow missing from real_df_for_concat. This indicates a bug. Exiting.")
                sys.exit(1)
            
            real_df_prepared_for_concat = real_df_for_concat[aligned_real_columns].copy()
            # DATE_TIME column in real_df_prepared_for_concat is already pd.to_datetime

            # --- Concatenate Synthetic and Real Data ---
            if not df_synthetic_for_output.empty:
                combined_df = pd.concat([df_synthetic_for_output, real_df_prepared_for_concat], ignore_index=True)
                print(f"INFO: Synthetic and real data concatenated. Synthetic rows: {len(df_synthetic_for_output)}, Real rows: {len(real_df_prepared_for_concat)}.")
            else:
                print("INFO: No synthetic data generated, output will contain only real data.")
                combined_df = real_df_prepared_for_concat.copy()
            
            # --- Define TARGET_COLUMN_ORDER (ensure it's defined before use) ---
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
            
            # --- Reorder columns and Format DATE_TIME for CSV ---
            df_data_to_save = pd.DataFrame()
            for col_name in TARGET_COLUMN_ORDER:
                if col_name in combined_df:
                    df_data_to_save[col_name] = combined_df[col_name]
                else:
                    df_data_to_save[col_name] = np.nan 
                    print(f"DEBUG: Column '{col_name}' from TARGET_COLUMN_ORDER not found in combined_df; adding as NaN.")
            
            # Format DATE_TIME to string 'YYYY-MM-DD HH:MM:SS' for CSV output
            if "DATE_TIME" in df_data_to_save:
                 df_data_to_save['DATE_TIME'] = pd.to_datetime(df_data_to_save['DATE_TIME']).dt.strftime('%Y-%m-%d %H:%M:%S')


            output_file = config['output_file']
            df_data_to_save.to_csv(output_file, index=False, na_rep='NaN') # Using na_rep for clarity
            print(f"Combined data saved to {output_file}. Total rows: {len(df_data_to_save)}")

            # --- Evaluation (uses X_syn_for_output_processed and X_real_processed from earlier) ---
            if n_synthetic_samples > 0 and X_syn_raw_preprocessed_space is not None:
                # At this point:
                # - X_syn_raw_preprocessed_space might have 'CLOSE' added.
                # - eval_feature_names reflects the columns of X_syn_raw_preprocessed_space (potentially including 'CLOSE').
                # - X_real_processed corresponds to 'initial_preprocessor_feature_names' (which does NOT have 'CLOSE' added by the synthetic patch).
                
                X_real_processed_for_eval = X_real_processed # Start with the original real processed data.

                # Align X_real_processed_for_eval with the (potentially augmented) eval_feature_names
                if 'CLOSE' in eval_feature_names and 'CLOSE' not in initial_preprocessor_feature_names:
                    # This implies 'CLOSE' was added to synthetic data and its eval_feature_names,
                    # but X_real_processed (described by initial_preprocessor_feature_names) doesn't have it yet.
                    print("INFO: Evaluation: 'CLOSE' is in target eval_feature_names and was added to synthetic data. Aligning X_real_processed.")
                    
                    df_real_temp = None
                    is_real_numpy = isinstance(X_real_processed, np.ndarray)

                    if is_real_numpy:
                        if X_real_processed.shape[1] == len(initial_preprocessor_feature_names):
                            df_real_temp = pd.DataFrame(X_real_processed, columns=initial_preprocessor_feature_names)
                        else:
                            print(f"ERROR: Evaluation: Shape mismatch for X_real_processed (NumPy array with {X_real_processed.shape[1]} cols) and initial_preprocessor_feature_names (count: {len(initial_preprocessor_feature_names)}). Cannot add 'CLOSE'.")
                    elif isinstance(X_real_processed, pd.DataFrame):
                        # If X_real_processed is already a DataFrame, ensure its columns match the expected original set.
                        if list(X_real_processed.columns) == initial_preprocessor_feature_names:
                            df_real_temp = X_real_processed.copy()
                        else:
                            # This is a less ideal state, try to proceed if BC-BO and OPEN exist.
                            print(f"WARNING: Evaluation: Columns of X_real_processed DataFrame do not exactly match initial_preprocessor_feature_names. Will attempt to find 'BC-BO' and 'OPEN' in existing columns: {list(X_real_processed.columns)}")
                            df_real_temp = X_real_processed.copy() 
                    else:
                        print(f"ERROR: Evaluation: X_real_processed is of unexpected type {type(X_real_processed)}. Cannot align for 'CLOSE'.")

                    if df_real_temp is not None:
                        if 'BC-BO' in df_real_temp.columns and 'OPEN' in df_real_temp.columns:
                            if 'CLOSE' not in df_real_temp.columns: # Add only if not somehow already there
                                df_real_temp['CLOSE'] = df_real_temp['BC-BO'] + df_real_temp['OPEN']
                                print("INFO: Evaluation: Successfully calculated and added 'CLOSE' column to a temporary DataFrame for X_real_processed.")
                                X_real_processed_for_eval = df_real_temp.to_numpy() if is_real_numpy else df_real_temp
                            else:
                                print("DEBUG: Evaluation: 'CLOSE' column already exists in df_real_temp. No addition needed.")
                                X_real_processed_for_eval = df_real_temp.to_numpy() if is_real_numpy else df_real_temp # Ensure it's assigned
                        else:
                            print("WARNING: Evaluation: Could not add 'CLOSE' to X_real_processed as 'BC-BO' or 'OPEN' (or both) are missing in its features. Evaluation might fail due to feature mismatch.")
                    else:
                        print("WARNING: Evaluation: df_real_temp was not created. Cannot add 'CLOSE' to X_real_processed. Evaluation might fail.")
                
                print(f"DEBUG: For evaluation - synthetic_data (raw preprocessed) shape: {X_syn_raw_preprocessed_space.shape}, real_data_processed_for_eval shape: {X_real_processed_for_eval.shape}, feature_names for eval (eval_feature_names): {eval_feature_names}")
                
                if X_syn_raw_preprocessed_space.shape[1] != X_real_processed_for_eval.shape[1]:
                     print(f"CRITICAL WARNING: Mismatch in feature counts for evaluation. Synthetic preprocessed features: {X_syn_raw_preprocessed_space.shape[1]}, Real preprocessed features for eval: {X_real_processed_for_eval.shape[1]}. Evaluation may fail or be incorrect.")
                
                metrics = evaluator_plugin.evaluate(
                    synthetic_data=X_syn_raw_preprocessed_space, 
                    real_data_processed=X_real_processed_for_eval, # Use the aligned real data
                    real_dates=datasets.get("eval_dates") or datasets.get("y_test_dates"), 
                    feature_names=eval_feature_names,        # This list now accurately describes both datasets
                    config=config
                )
                metrics_file = config['metrics_file']
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                print(f"Evaluation metrics saved to {metrics_file}.")
            else:
                print("Skipping evaluation as no synthetic data was generated.")

        except Exception as e:
            print(f"Synthetic data generation, combination, or evaluation failed: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
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
