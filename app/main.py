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


    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, {}, cli_args, unknown_args_dict)


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
    # -------------------------------------------------------------------------
    # GAN TRAINING OR VAE‐GENERATE + EVALUATE
    # -------------------------------------------------------------------------
    # Use presence of --optimizer to decide, not the old --use_optimizer flag
    if cli_args.get('optimizer') and \
        getattr(optimizer_plugin, "__class__", None).__name__ == "GANTrainerPlugin":
         print("▶ Running GAN training via Optimizer Plugin...")
         try:
             optimizer_plugin.optimize(
                 feeder=feeder_plugin,
                 generator=generator_plugin,
                 evaluator=evaluator_plugin,
                 preprocessor=preprocessor_plugin,
                 config=config
             )
             print("✔︎ GAN training completed.")

             # Swap in the trained generator for downstream generation
             trained_gen = optimizer_plugin.get_trained_generator()
             generator_plugin.update_model(trained_gen)
             print("✔︎ Generator plugin updated with GAN‐trained weights.")
         except Exception as e:
             print(f"❌ GAN training failed: {e}")
             sys.exit(1)

         print("▶ Proceeding to synthetic data generation and evaluation…")
    else:
         if cli_args.get('optimizer'):
             print("▶ Running hyperparameter optimization with Optimizer Plugin…")
             try:
                 optimal_params = optimizer_plugin.optimize(
                     feeder_plugin,
                     generator_plugin,
                     evaluator_plugin,
                     preprocessor_plugin,
                     config
                 )
                 print("✔︎ Hyperparameter optimization completed.")
                 sys.exit(0)
             except Exception as e:
                 print(f"❌ Hyperparameter optimization failed: {e}")
                 sys.exit(1)
         else:
             print("▶ Skipping optimization, generating synthetic data and evaluating…")
 
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
               