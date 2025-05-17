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
from typing import Any, Dict, List # Added List
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

# Attempt to import pandas_ta, if not available, TI calculation will be skipped.
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("WARNING: pandas_ta library not found. Technical indicator calculation will be skipped.")

# Assume these are defined in your config or constants file and loaded into `config`
# For example:
# TARGET_COLUMN_ORDER = config.get("target_column_order", ['DATETIME', 'High', 'Low', 'Open', 'Close', 'Close_Open', 'High_Low', 'RSI_14', ...])
# DATE_TIME_COLUMN_NAME = config.get("datetime_col_name", "DATETIME")
# NUM_BASE_FEATURES_GENERATED = config.get("num_base_features_generated", 6) # e.g., H,L,O,C, C-O, H-L
# BASE_FEATURE_NAMES = TARGET_COLUMN_ORDER[1:NUM_BASE_FEATURES_GENERATED+1] # e.g., ['High', ..., 'High_Low']

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
                print("Preprocessing real data for evaluation via PreprocessorPlugin...")

                if not hasattr(preprocessor_plugin, 'run_preprocessing'):
                    raise AttributeError("PreprocessorPlugin does not have a 'run_preprocessing' method.")

                config_for_preprocessor_run = config.copy()
                # --- WORKAROUND 1: For "Baseline train indices invalid" ---
                if not config_for_preprocessor_run.get('use_stl', False):
                    if 'stl_window' in preprocessor_plugin.plugin_params or 'stl_window' in config_for_preprocessor_run:
                        original_stl_window = config_for_preprocessor_run.get('stl_window')
                        config_for_preprocessor_run['stl_window'] = 1
                        print(f"INFO: WORKAROUND 1: 'use_stl' is False. "
                              f"Original 'stl_window': {original_stl_window}. "
                              "Temporarily setting 'stl_window' to 1 to prevent 'Baseline train indices invalid' error.")
                else:
                    print("DEBUG main.py: WORKAROUND 1: 'use_stl' is True or not set, 'stl_window' workaround not applied.")

                # --- WORKAROUND 2: For "Wavelet: Length of data must be even" ---
                print("DEBUG main.py: WORKAROUND 2: Ensuring even row counts for relevant data files.")
                data_file_keys = [
                    'real_data_file', 'x_train_file', 'y_train_file',
                    'x_validation_file', 'y_validation_file',
                    'x_val_file', 'y_val_file',
                    'x_test_file', 'y_test_file'
                ]
                temp_files_created_paths = []
                for file_key in data_file_keys:
                    original_file_path = config_for_preprocessor_run.get(file_key)
                    if (
                        original_file_path
                        and isinstance(original_file_path, str)
                        and os.path.exists(original_file_path)
                    ):
                        try:
                            df_data = pd.read_csv(original_file_path)
                            if len(df_data) % 2 != 0:
                                df_trunc = df_data.iloc[:-1]
                                if not df_trunc.empty:
                                    with tempfile.NamedTemporaryFile(
                                        delete=False,
                                        mode='w',
                                        newline='',
                                        suffix=f'_{file_key}.csv'
                                    ) as tmpf:
                                        df_trunc.to_csv(tmpf.name, index=False)
                                        temp_path = tmpf.name
                                    temp_files_created_paths.append(temp_path)
                                    config_for_preprocessor_run[file_key] = temp_path
                                    print(f"INFO: WORKAROUND 2: Using temp even-length file "
                                          f"'{temp_path}' for key '{file_key}'. New length: {len(df_trunc)}")
                        except Exception as e2:
                            print(f"WARN: WORKAROUND 2: Error processing '{original_file_path}': {e2}")

                try:
                    datasets = preprocessor_plugin.run_preprocessing(
                        config=config_for_preprocessor_run
                    )
                finally:
                    for tmp_path in temp_files_created_paths:
                        try:
                            os.remove(tmp_path)
                        except OSError as rm_err:
                            print(f"WARN: Failed to remove temp file '{tmp_path}': {rm_err}")

                print("PreprocessorPlugin.run_preprocessing finished.")

                # Extract feature names for evaluation
                if "feature_names" in datasets:
                    eval_feature_names = datasets["feature_names"]
                elif (
                    isinstance(datasets.get("x_train"), pd.DataFrame)
                    and hasattr(datasets["x_train"], 'columns')
                ):
                    eval_feature_names = list(datasets["x_train"].columns)
                elif (
                    datasets.get("x_train") is not None
                    and getattr(datasets["x_train"], 'ndim', None) == 2
                ):
                    n_feats = datasets["x_train"].shape[1]
                    eval_feature_names = [f"feature_{i}" for i in range(n_feats)]
                else:
                    print(
                        "WARNING: Could not reliably get 'eval_feature_names' "
                        "from preprocessor output. Will attempt to construct from TARGET_COLUMN_ORDER."
                    )
                    # Fallback: Use TARGET_COLUMN_ORDER if eval_feature_names couldn't be determined
                    # Ensure TARGET_COLUMN_ORDER is defined (e.g., from config)
                    target_column_order_from_config = config.get("target_column_order")
                    datetime_col_name_from_config = config.get("datetime_col_name", "DATETIME")
                    if target_column_order_from_config:
                        eval_feature_names = [col for col in target_column_order_from_config if col != datetime_col_name_from_config]
                    else:
                        raise ValueError("TARGET_COLUMN_ORDER not found in config, and could not infer feature names.")
                
                # Ensure eval_feature_names is available for the rest of the script
                if not eval_feature_names:
                    raise ValueError("eval_feature_names could not be determined after preprocessing.")


                # Determine the real data and datetimes to use for evaluation reference
                # Prioritize validation set, then test set, then train set from preprocessor output
                real_data_for_eval_key = None
                real_datetimes_for_eval_key = None

                if "x_validation" in datasets and datasets["x_validation"] is not None:
                    real_data_for_eval_key = "x_validation"
                    real_datetimes_for_eval_key = "validation_datetimes"
                elif "x_test" in datasets and datasets["x_test"] is not None:
                    real_data_for_eval_key = "x_test"
                    real_datetimes_for_eval_key = "test_datetimes"
                elif "x_train" in datasets and datasets["x_train"] is not None: # Fallback to train if others not present
                    real_data_for_eval_key = "x_train"
                    real_datetimes_for_eval_key = "train_datetimes"
                
                if not real_data_for_eval_key:
                    raise ValueError("No suitable real data (x_validation, x_test, or x_train) found in preprocessor output for evaluation.")

                X_real_eval_source = datasets[real_data_for_eval_key]
                # Ensure datetimes are pandas Series of Timestamps
                if real_datetimes_for_eval_key and real_datetimes_for_eval_key in datasets and datasets[real_datetimes_for_eval_key] is not None:
                    datetimes_eval_source = pd.Series(pd.to_datetime(datasets[real_datetimes_for_eval_key]))
                elif isinstance(X_real_eval_source, pd.DataFrame) and X_real_eval_source.index.inferred_type == 'datetime64':
                     datetimes_eval_source = pd.Series(X_real_eval_source.index)
                else:
                    raise ValueError(f"Could not find or infer datetimes for the evaluation data source '{real_data_for_eval_key}'.")

                if isinstance(X_real_eval_source, pd.DataFrame):
                    X_real_eval_source_np = X_real_eval_source.values
                elif isinstance(X_real_eval_source, np.ndarray):
                    X_real_eval_source_np = X_real_eval_source
                else:
                    raise TypeError(f"Unsupported data type for '{real_data_for_eval_key}': {type(X_real_eval_source)}")

                print(f"Using '{real_data_for_eval_key}' (shape: {X_real_eval_source_np.shape}) and its datetimes for evaluation reference.")

            except Exception as e:
                print(f"❌ Preprocessing or evaluation setup failed: {e}")
                sys.exit(1)

            # --- SEQUENTIAL SYNTHETIC DATA GENERATION ---
            print("▶ Starting sequential synthetic data generation...")
            try:
                sequence_length_T = config.get('sequence_length_T', len(datetimes_eval_source))
                if sequence_length_T > len(datetimes_eval_source):
                    print(f"WARNING: Requested sequence_length_T ({sequence_length_T}) is greater than available evaluation datetimes ({len(datetimes_eval_source)}). Clamping to available length.")
                    sequence_length_T = len(datetimes_eval_source)
                
                if sequence_length_T == 0:
                    raise ValueError("Cannot generate sequence of length 0. Check 'sequence_length_T' config or available evaluation data.")

                target_datetimes_for_generation = datetimes_eval_source.iloc[:sequence_length_T]

                # Prepare initial history (example: using zeros or a segment of real data)
                # This needs careful consideration based on your specific use case.
                history_k = generator_plugin.params.get("history_k", 10)
                num_base_features = generator_plugin.params.get("num_base_features", 6) # From GeneratorPlugin params
                num_fundamental_features = generator_plugin.params.get("num_fundamental_features", 2) # From GeneratorPlugin params

                initial_history_base = np.zeros((history_k, num_base_features))
                initial_history_fundamentals = np.zeros((history_k, num_fundamental_features))
                
                # Optionally, populate initial_history from the end of training data or start of real_eval_source
                # Example: if X_real_train_processed is available and has same feature structure
                # initial_history_base = X_real_train_processed[-history_k:, :num_base_features]
                # initial_history_fundamentals = X_real_train_processed[-history_k:, num_base_features:num_base_features+num_fundamental_features]
                print(f"Using initial_history_base shape: {initial_history_base.shape}, initial_history_fundamentals shape: {initial_history_fundamentals.shape}")

                feeder_outputs_for_sequence: List[Dict[str, np.ndarray]] = []
                print(f"Generating feeder outputs for {sequence_length_T} steps...")
                for t_idx in range(sequence_length_T):
                    current_dt_for_step = target_datetimes_for_generation.iloc[t_idx]
                    # Feeder expects n_samples and optional datetimes_for_conditioning (as pd.Series)
                    feeder_step_output = feeder_plugin.generate(
                        n_samples=1, # Generating one step at a time for the sequence
                        datetimes_for_conditioning=pd.Series([current_dt_for_step])
                    )
                    feeder_outputs_for_sequence.append(feeder_step_output)
                
                print("Generating synthetic base sequence via GeneratorPlugin...")
                # GeneratorPlugin's generate method is now sequential
                generated_base_sequence_batch = generator_plugin.generate(
                    feeder_outputs_sequence=feeder_outputs_for_sequence,
                    sequence_length_T=sequence_length_T,
                    initial_history_base=initial_history_base,
                    initial_history_fundamentals=initial_history_fundamentals
                ) # Expected output shape: (1, sequence_length_T, num_base_features)

                if not (isinstance(generated_base_sequence_batch, np.ndarray) and generated_base_sequence_batch.ndim == 3):
                    raise ValueError(f"GeneratorPlugin output has unexpected shape or type: {generated_base_sequence_batch.shape if isinstance(generated_base_sequence_batch, np.ndarray) else type(generated_base_sequence_batch)}")

                generated_base_sequence_np = generated_base_sequence_batch[0] # Shape: (sequence_length_T, num_base_features)
                print(f"Generated synthetic base sequence with shape: {generated_base_sequence_np.shape}")

                # --- Calculate Technical Indicators on Generated Base Sequence ---
                # Define base feature names based on TARGET_COLUMN_ORDER and num_base_features
                # These names must match what pandas_ta expects (e.g., 'high', 'low', 'open', 'close', 'volume')
                # For simplicity, assume the first `num_base_features` in `eval_feature_names` (excluding date)
                # correspond to the generated base features. This needs to be robust.
                
                target_column_order_from_config = config.get("target_column_order", eval_feature_names)
                datetime_col_name_from_config = config.get("datetime_col_name", "DATETIME")
                
                # Assuming the first N columns of eval_feature_names (if it's ordered like target_column_order)
                # are the base features. Or, use a predefined list.
                # For pandas-ta, column names like 'open', 'high', 'low', 'close' are often expected.
                # This mapping needs to be accurate.
                
                # Example: if your generator outputs H, L, O, C, C-O, H-L
                # and your eval_feature_names from preprocessor are ['High', 'Low', 'Open', 'Close', 'Close_Open', 'High_Low', 'RSI_14', ...]
                base_feature_names_for_df = eval_feature_names[:num_base_features]
                
                generated_df = pd.DataFrame(generated_base_sequence_np, columns=base_feature_names_for_df)
                
                # Standardize column names for pandas_ta if necessary (e.g., to lowercase 'high', 'low', 'open', 'close')
                # This is a common requirement for pandas_ta.
                generated_df_for_ta = generated_df.copy()
                rename_map = {col: col.lower() for col in generated_df_for_ta.columns if col.lower() in ['high', 'low', 'open', 'close', 'volume']}
                generated_df_for_ta.rename(columns=rename_map, inplace=True)

                synthetic_ti_features_list = []
                synthetic_ti_names = []

                if PANDAS_TA_AVAILABLE:
                    print("Calculating technical indicators on synthetic data using pandas_ta...")
                    # Example TIs (customize as per your 11 indicators and their params)
                    # Ensure generated_df_for_ta has 'high', 'low', 'open', 'close' columns if these TIs need them.
                    if all(col in generated_df_for_ta.columns for col in ['high', 'low', 'close']):
                        generated_df_for_ta.ta.rsi(length=14, append=True) # Appends 'RSI_14'
                        generated_df_for_ta.ta.macd(append=True) # Appends 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
                        # Add other pandas_ta indicators here, e.g., EMA, Bollinger Bands
                        # generated_df_for_ta.ta.ema(length=20, append=True)
                        # generated_df_for_ta.ta.bbands(length=20, append=True) # Appends BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
                        
                        # Collect TI features that were appended by pandas_ta
                        # The names are typically like 'RSI_14', 'MACD_12_26_9', etc.
                        # Identify newly added columns by pandas-ta
                        original_ta_cols = set(rename_map.values()) # cols fed to TA
                        current_ta_cols = set(generated_df_for_ta.columns)
                        newly_added_ti_cols = list(current_ta_cols - original_ta_cols)
                        
                        for ti_col_name in newly_added_ti_cols:
                            synthetic_ti_features_list.append(generated_df_for_ta[ti_col_name].values.reshape(-1,1))
                            synthetic_ti_names.append(ti_col_name.upper()) # Store in uppercase to match typical eval_feature_names
                        print(f"Calculated TIs: {synthetic_ti_names}")
                    else:
                        print("WARNING: Not all required columns (high, low, close) found in generated data for pandas_ta. Skipping TI calculation.")
                else:
                    print("Skipping technical indicator calculation as pandas_ta is not available.")

                # Combine generated base sequence with calculated TIs
                if synthetic_ti_features_list:
                    synthetic_ti_features_np = np.concatenate(synthetic_ti_features_list, axis=1)
                    X_syn_processed = np.concatenate((generated_base_sequence_np, synthetic_ti_features_np), axis=1)
                    # Update eval_feature_names to include these new TIs
                    # This assumes the original eval_feature_names from preprocessor already contains
                    # placeholders or actual values for these TIs in the correct order.
                    # For simplicity, we'll construct a new list of feature names for the synthetic data.
                    current_synthetic_feature_names = base_feature_names_for_df + synthetic_ti_names
                else:
                    X_syn_processed = generated_base_sequence_np
                    current_synthetic_feature_names = base_feature_names_for_df
                
                print(f"Processed synthetic data X_syn_processed shape: {X_syn_processed.shape}")

                # --- Placeholder for High-Frequency Window Extraction (Section 5.4) ---
                # If HF data is derived from the hourly generated sequence, it would happen here.
                # This would involve taking the `generated_base_sequence_np` (specifically the close prices),
                # and applying logic to "hallucinate" or model 15-min/30-min movements.
                # This is a complex step and might require another model or sophisticated interpolation.
                # If done, X_syn_processed and current_synthetic_feature_names would be further augmented.
                print("Placeholder: High-frequency window extraction would occur here if implemented.")


                # --- Prepare Real Data Segment for Evaluation ---
                # Ensure the real data segment matches the structure of X_syn_processed (base + TIs + HF)
                # The `X_real_eval_source_np` should ideally already contain all these features
                # as processed by `PreprocessorPlugin`.
                
                # We need to select the portion of X_real_eval_source_np that aligns with sequence_length_T
                X_real_eval_segment_np = X_real_eval_source_np[:sequence_length_T, :]
                
                # Ensure feature names for evaluation align with both synthetic and real data structures
                # `eval_feature_names` (from preprocessor) should be the superset of all features.
                # We need to ensure `current_synthetic_feature_names` maps correctly to a subset of `eval_feature_names`
                # and that `X_syn_processed` and `X_real_eval_segment_np` are aligned column-wise for evaluation.

                # For robust evaluation, ensure both arrays have the same number of features
                # and that `eval_feature_names` corresponds to these features.
                # If TIs were calculated for synthetic, the real data must also have them in the same order.
                
                # Let's assume `eval_feature_names` from preprocessor is the ground truth order.
                # We need to make sure `X_syn_processed` columns are reordered/selected to match `eval_feature_names`
                # if `current_synthetic_feature_names` differs in order or content.
                
                # This is a simplified alignment:
                # It assumes `current_synthetic_feature_names` are a subset of `eval_feature_names`
                # and that the real data `X_real_eval_segment_np` already has all `eval_feature_names` columns.
                
                final_synthetic_data_for_eval = pd.DataFrame(X_syn_processed, columns=current_synthetic_feature_names)
                final_real_data_for_eval_df = pd.DataFrame(X_real_eval_segment_np, columns=eval_feature_names) # From preprocessor

                # Align columns of synthetic data to match the order and selection of eval_feature_names from real data
                # Only keep columns present in both and order synthetic like real
                common_features = [feat_name for feat_name in eval_feature_names if feat_name in final_synthetic_data_for_eval.columns]
                
                if not common_features:
                    raise ValueError("No common features found between generated synthetic data and real evaluation feature names. Check feature naming and TI calculation.")

                final_synthetic_data_for_eval_aligned = final_synthetic_data_for_eval[common_features]
                final_real_data_for_eval_aligned = final_real_data_for_eval_df[common_features]
                aligned_eval_feature_names = common_features

                print(f"Shape of final synthetic data for eval: {final_synthetic_data_for_eval_aligned.shape}")
                print(f"Shape of final real data for eval: {final_real_data_for_eval_aligned.shape}")
                print(f"Number of features for evaluation: {len(aligned_eval_feature_names)}. Names: {aligned_eval_feature_names[:10]}...")


                # 3. Evaluate synthetic data using the EvaluatorPlugin
                print("Evaluating synthetic data via EvaluatorPlugin...")
                metrics = evaluator_plugin.evaluate(
                    synthetic_data_sequence=final_synthetic_data_for_eval_aligned.values, # Pass as numpy array
                    real_data_sequence=final_real_data_for_eval_aligned.values,           # Pass as numpy array
                    feature_names=aligned_eval_feature_names, # Use the aligned feature names
                    config=config 
                )
                print("✔︎ Evaluation completed.")
                # print(f"Evaluation metrics: {json.dumps(metrics, indent=4)}") # Can be very verbose

                # Save metrics to a file
                metrics_output_file = config.get("metrics_output_file", "examples/results/evaluation_metrics.json")
                os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)
                with open(metrics_output_file, "w") as f:
                    # Custom serializer for numpy types if metrics contain them
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.integer): return int(obj)
                            if isinstance(obj, np.floating): return float(obj)
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
                            return super(NumpyEncoder, self).default(obj)
                    json.dump(metrics, f, indent=4, cls=NumpyEncoder)
                print(f"✔︎ Evaluation metrics saved to {metrics_output_file}")

            except Exception as e:
                print(f"❌ Synthetic data generation or evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
