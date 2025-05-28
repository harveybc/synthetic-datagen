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
from typing import Any, Dict, List
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
import traceback # MOVED HERE - NOW AT GLOBAL SCOPE

# --- MONKEY PATCH for numpy.NaN ---
# Applied because pandas_ta 0.3.14b0 (or a dependency) seems to use
# the deprecated np.NaN with NumPy 2.x.
if hasattr(np, '__version__') and int(np.__version__.split('.')[0]) >= 2: # Check if NumPy is version 2.x or higher
    if not hasattr(np, 'NaN'):
        print("INFO: Monkey patching numpy: Assigning np.NaN = np.nan for compatibility.")
        np.NaN = np.nan
    elif np.NaN is not np.nan: # If NaN exists but is not the same object as nan (less likely for np 2.x)
        print("INFO: Monkey patching numpy: np.NaN exists but is not np.nan. Re-assigning.")
        np.NaN = np.nan
# --- END MONKEY PATCH ---

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.config import DEFAULT_VALUES # Ensure this provides 'full_feature_names_ordered'
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

# Debugging information
print("--- Python sys.path from main.py ---")
for p in sys.path:
    print(p)
print("--- End Python sys.path ---")

print("--- Python Environment Variables from main.py ---")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")
print(f"PATH: {os.environ.get('PATH')}")
print("--- End Python Environment Variables ---")

print("--- Checking NumPy version before pandas_ta import ---")
try:
    import numpy
    print(f"Successfully imported numpy. Version: {numpy.__version__}")
    print(f"Numpy location: {numpy.__file__}")
    # Try accessing np.NaN to see if this specific part fails early
    try:
        _ = numpy.NaN
        print("numpy.NaN is accessible.")
    except AttributeError as e_nan:
        print(f"Error accessing numpy.NaN: {e_nan}")
except ImportError as e_np:
    print(f"Failed to import numpy: {e_np}")
except Exception as e_np_other:
    print(f"An unexpected error occurred while importing numpy: {e_np_other}")
print("--- End NumPy version check ---")

print("--- Attempting to import pandas_ta directly in main.py ---")
try:
    import pandas_ta
    print("✔︎ pandas_ta imported successfully.")
except ImportError as e_pta:
    print(f"❌ Failed to import pandas_ta: {e_pta}")
except Exception as e_pta_other:
    print(f"An unexpected error occurred while importing pandas_ta: {e_pta_other}")
print("--- End pandas_ta import attempt ---")

# Assume these are defined in your config or constants file and loaded into `config`
# For example:
# TARGET_COLUMN_ORDER = config.get("target_column_order", ['DATETIME', 'High', 'Low', 'Open', 'Close', 'Close_Open', 'High_Low', 'RSI_14', ...])
# DATE_TIME_COLUMN_NAME = config.get("datetime_col_name", "DATETIME")
# NUM_BASE_FEATURES_GENERATED = config.get("num_base_features_generated", 6) # e.g., H,L,O,C, C-O, H-L
# BASE_FEATURE_NAMES = TARGET_COLUMN_ORDER[1:NUM_BASE_FEATURES_GENERATED+1] # e.g., ['High', ..., 'High_Low']

# --- REMOVE UNUSED FUNCTION ---
# The function 'generate_synthetic_datetimes_before_real' was defined but not found to be used.
# If it is used by other modules importing main.py (unlikely for a main script), it should be kept.
# For now, assuming it's unused within this script's execution flow.
# def generate_synthetic_datetimes_before_real(...): ...

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
        decoder_model = getattr(generator_plugin, "sequential_model", None) 
        if decoder_model is None:
            decoder_model = getattr(generator_plugin, "model", None)
            if decoder_model is None:
                raise RuntimeError("GeneratorPlugin must expose attribute 'sequential_model' or 'model'.")
            else:
                print("DEBUG main.py: Found decoder model under attribute 'model'.")
        else:
            print("DEBUG main.py: Found decoder model under attribute 'sequential_model'.")
        
        if not hasattr(decoder_model, 'inputs') or not decoder_model.inputs:
            raise RuntimeError(f"Decoder model '{type(decoder_model).__name__}' does not have inspectable 'inputs' attribute or it's empty.")

        decoder_inputs = decoder_model.inputs 
        decoder_input_names = [inp.name.split(':')[0] for inp in decoder_inputs] 
        
        latent_input_name_from_config = generator_plugin.params.get("decoder_input_name_latent")
        if not latent_input_name_from_config:
            raise ValueError("GeneratorPlugin config missing 'decoder_input_name_latent'.")

        inferred_latent_shape = None # Will be a tuple (seq_len, features)
        latent_input_found = False
        for i, input_tensor in enumerate(decoder_inputs):
            input_layer_name = input_tensor.name.split(':')[0]
            if input_layer_name == latent_input_name_from_config:
                shape_list = list(input_tensor.shape) # e.g. [None, 18, 32]
                if len(shape_list) == 3 and shape_list[1] is not None and shape_list[2] is not None:
                    inferred_latent_shape = (shape_list[1], shape_list[2]) # (seq_len, features)
                    latent_input_found = True
                    print(f"DEBUG main.py: Found latent input '{latent_input_name_from_config}' with original shape {input_tensor.shape}. Inferred latent_shape: {inferred_latent_shape}")
                    break
                else:
                    print(f"DEBUG main.py: Latent input '{latent_input_name_from_config}' found, but original shape {input_tensor.shape} (list: {shape_list}) is not 3D with defined sequence length and features for shape inference.")
        
        if not latent_input_found:
            raise RuntimeError(f"Could not find input layer named '{latent_input_name_from_config}' in decoder model. Available inputs: {decoder_input_names}")
        if inferred_latent_shape is None:
            raise RuntimeError(f"Could not determine a valid inferred latent shape for '{latent_input_name_from_config}'. Expected 3D shape like (None, seq_len, features).")

        print(f"DEBUG main.py: Setting FeederPlugin latent_shape to: {inferred_latent_shape}")
        # Pass the tuple (seq_len, features)
        feeder_plugin.set_params(latent_shape=list(inferred_latent_shape)) # Pass as list as per FeederPlugin's type hints/usage
        # Update the main config as well, so OptimizerPlugin sees the correct base if it doesn't tune it
        config['latent_shape'] = list(inferred_latent_shape)
        # If optimizer tunes 'latent_dim' (feature part), FeederPlugin's set_params will handle it.
    except Exception as e:
        print(f"Failed to load or initialize Generator Plugin '{plugin_name}': {e}")
        traceback.print_exc() # This will now work correctly
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


    # --- REMOVE REDUNDANT CONFIG MERGE ---
    # The following print and merge_config call appear to be duplicates of the initial merge
    # and might incorrectly re-apply or nullify parts of the config (e.g., file_config={}).
    # print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    # unknown_args_dict = process_unknown_args(unknown_args) # unknown_args_dict is already defined
    # config = merge_config(config, {}, {}, {}, cli_args, unknown_args_dict)


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
            # Add a check here: if pd.to_datetime returns None without an error
            if current_dt is None:
                print(f"WARNING generate_datetime_column: pd.to_datetime('{start_datetime_str}') returned None. Defaulting to current system time.")
                current_dt = pd.to_datetime(datetime.now().replace(microsecond=0))
        except Exception as e:
            print(f"ERROR generate_datetime_column: Error parsing 'start_datetime_str' ('{start_datetime_str}'): {e}. Defaulting to current system time.")
            current_dt = pd.to_datetime(datetime.now().replace(microsecond=0))
        
        # Ensure current_dt is a valid datetime object before proceeding
        if not isinstance(current_dt, (datetime, pd.Timestamp)):
            print(f"CRITICAL generate_datetime_column: current_dt is not a valid datetime object after parsing/defaulting (type: {type(current_dt)}). Forcing to now.")
            current_dt = pd.to_datetime(datetime.now().replace(microsecond=0))
            if not isinstance(current_dt, (datetime, pd.Timestamp)): # Final check if even system time fails
                 raise SystemError(f"Failed to obtain a valid datetime object. Last attempt with system time resulted in type: {type(current_dt)}")

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
    
    is_gan_training_mode = config.get("gan_training_mode", False) # Assuming you might add this
    is_hyperparam_opt_mode = config.get("hyperparameter_optimization_mode", False) and config.get("run_hyperparameter_optimization", True) # MODIFIED

    # Determine operation mode
    # if is_gan_training_mode: # Placeholder for GAN training mode
    #     print("▶ Running GAN training with Optimizer Plugin…")
    #     # ... (GAN training logic using optimizer_plugin.train_gan(...) or similar)
    #     sys.exit(0)

    # elif is_hyperparam_opt_mode: # MODIFIED
    if is_hyperparam_opt_mode: # Check if hyperparameter optimization should run
        print("▶ Running hyperparameter optimization with Optimizer Plugin…")
        try:
            optimal_params = optimizer_plugin.optimize(
                feeder_plugin,
                generator_plugin,
                evaluator_plugin,
                config
            )
            print("✔︎ Hyperparameter optimization completed.")
            # print(f"Optimal parameters: {optimal_params}") # Optionally print
            # Update config with optimal_params before proceeding to generation, or exit
            # For now, exiting as per original logic for hyperparam opt.
            sys.exit(0) 
        except Exception as e:
            print(f"❌ Hyperparameter optimization failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # If not in hyperparameter optimization mode (or if it was skipped), proceed to generation.
    # Ensure FeederPlugin is ready for generation (it should be from initialization)
    print("▶ Generating synthetic data with Generator Plugin…")

    # This block will now execute if:
    # 1. No optimizer was specified.
    # 2. GAN training was completed (it falls through).
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
            real_datetimes_for_eval_key = "x_val_dates" # CORRECTED KEY
        elif "x_test" in datasets and datasets["x_test"] is not None:
            real_data_for_eval_key = "x_test"
            real_datetimes_for_eval_key = "x_test_dates" # CORRECTED KEY
        elif "x_train" in datasets and datasets["x_train"] is not None: # Fallback to train if others not present
            real_data_for_eval_key = "x_train"
            real_datetimes_for_eval_key = "x_train_dates" # CORRECTED KEY
        
        if not real_data_for_eval_key:
            raise ValueError("No suitable real data (x_validation, x_test, or x_train) found in preprocessor output for evaluation.")

        X_real_eval_source = datasets[real_data_for_eval_key]
        # Ensure datetimes are pandas Series of Timestamps
        if real_datetimes_for_eval_key and real_datetimes_for_eval_key in datasets and datasets[real_datetimes_for_eval_key] is not None:
            datetimes_eval_source = pd.Series(pd.to_datetime(datasets[real_datetimes_for_eval_key]))
        elif isinstance(X_real_eval_source, pd.DataFrame) and X_real_eval_source.index.inferred_type == 'datetime64':
             datetimes_eval_source = pd.Series(X_real_eval_source.index)
        else:
            # Add more debug info to the error message
            available_datetime_keys = [k for k in datasets.keys() if "date" in k.lower()]
            raise ValueError(
                f"Could not find or infer datetimes for the evaluation data source '{real_data_for_eval_key}'. "
                f"Attempted key: '{real_datetimes_for_eval_key}'. "
                f"Available dataset keys possibly related to datetimes: {available_datetime_keys}. "
                f"Is '{real_datetimes_for_eval_key}' present and not None in datasets? "
                f"Is X_real_eval_source a DataFrame with a datetime index?"
            )

        if isinstance(X_real_eval_source, pd.DataFrame):
            X_real_eval_source_np = X_real_eval_source.values
        elif isinstance(X_real_eval_source, np.ndarray):
            X_real_eval_source_np = X_real_eval_source
        else:
            raise TypeError(f"Unsupported data type for '{real_data_for_eval_key}': {type(X_real_eval_source)}")

        print(f"Using '{real_data_for_eval_key}' (shape: {X_real_eval_source_np.shape}) and its datetimes for evaluation reference.")

    except Exception as e:
        print(f"❌ Preprocessing or evaluation setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- SEQUENTIAL SYNTHETIC DATA GENERATION ---
    print("▶ Starting conditional sequential synthetic data generation...")
    try:
        # Determine the number of ticks to generate
        n_config = config.get('num_synthetic_samples_to_generate') # This will be 0 by default from app/config.py

        if n_config is not None and n_config > 0:
            num_synthetic_samples_to_generate = n_config
            print(f"INFO: Using 'num_synthetic_samples_to_generate' from configuration: {num_synthetic_samples_to_generate}")
        else:
            # If n_config is 0, None (not possible with DEFAULT_VALUES), or negative, default to length of evaluation data
            default_length_source_name = "datetimes_eval_source"
            default_length = 0
            if hasattr(datetimes_eval_source, '__len__') and datetimes_eval_source is not None:
                default_length = len(datetimes_eval_source)
            else:
                # This case should ideally be caught earlier when datetimes_eval_source is defined.
                print(f"WARNING: {default_length_source_name} is not available or not a sequence. Defaulting generated samples to 0, which will likely error.")
            
            if n_config == 0:
                print(f"INFO: 'num_synthetic_samples_to_generate' is 0 in config. Defaulting to length of {default_length_source_name} ({default_length}).")
            elif n_config is None: # Should not happen given DEFAULT_VALUES
                 print(f"INFO: 'num_synthetic_samples_to_generate' not found in config. Defaulting to length of {default_length_source_name} ({default_length}).")
            else: # n_config < 0
                 print(f"INFO: 'num_synthetic_samples_to_generate' is negative ({n_config}) in config. Defaulting to length of {default_length_source_name} ({default_length}).")
            
            num_synthetic_samples_to_generate = default_length

        if num_synthetic_samples_to_generate <= 0:
            len_eval_source_str = str(len(datetimes_eval_source)) if hasattr(datetimes_eval_source, '__len__') and datetimes_eval_source is not None else 'N/A or empty'
            error_msg = (
                f"num_synthetic_samples_to_generate must be positive. "
                f"Resulted in {num_synthetic_samples_to_generate}. "
                f"Config value for 'num_synthetic_samples_to_generate' was: {n_config}. "
                f"Length of datetimes_eval_source: {len_eval_source_str}. "
                f"Ensure 'num_synthetic_samples_to_generate' in your config is positive, "
                f"or if it's 0 or not set, ensure evaluation data (e.g., x_test_dates from preprocessor) is available and non-empty."
            )
            raise ValueError(error_msg)
        
        # Generate target datetimes for the synthetic sequence
        start_datetime_str = config.get('start_datetime', datetimes_eval_source.iloc[0].strftime('%Y-%m-%d %H:%M:%S') if not datetimes_eval_source.empty else datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        periodicity_str = config.get('dataset_periodicity', '1h') # Default to hourly if not specified
        
        # Use the robust generate_datetime_column to get string datetimes first
        target_datetimes_str_list = generate_datetime_column(
            start_datetime_str=start_datetime_str,
            num_samples=num_synthetic_samples_to_generate,
            periodicity_str=periodicity_str
        )
        # Convert to pandas Series of Timestamps for the FeederPlugin
        target_datetimes_for_generation = pd.Series(pd.to_datetime(target_datetimes_str_list))
        
        sequence_length_T = len(target_datetimes_for_generation)
        if sequence_length_T == 0:
            raise ValueError("Cannot generate sequence of length 0. Check datetime generation parameters.")
        
        print(f"Targeting {sequence_length_T} synthetic samples, starting from {target_datetimes_for_generation.iloc[0]} with periodicity {periodicity_str}.")

        # Prepare initial full feature window for the GeneratorPlugin
        decoder_input_window_size = generator_plugin.params.get("decoder_input_window_size")
        generator_req_feature_names = generator_plugin.params.get("full_feature_names_ordered", [])
        if not generator_req_feature_names:
            raise ValueError("GeneratorPlugin's 'full_feature_names_ordered' is empty in config.")
        num_all_features_gen = len(generator_req_feature_names)

        initial_full_feature_window = np.zeros((decoder_input_window_size, num_all_features_gen), dtype=np.float32) # Default
        
        if X_real_eval_source_np is not None and X_real_eval_source_np.shape[0] >= decoder_input_window_size:
            print(f"Attempting to populate initial_full_feature_window from preprocessed real data ('{real_data_for_eval_key}').")
            # Create a DataFrame from the real data source to easily select and reorder columns
            # Ensure eval_feature_names matches the columns of X_real_eval_source_np
            if len(eval_feature_names) != X_real_eval_source_np.shape[1]:
                print(f"GeneratorPlugin Warning: Mismatch between length of eval_feature_names ({len(eval_feature_names)}) "
                      f"and number of columns in X_real_eval_source_np ({X_real_eval_source_np.shape[1]}). "
                      f"Cannot reliably create DataFrame for initial window. Using zeros.")
            else:
                real_data_df_for_init = pd.DataFrame(X_real_eval_source_np, columns=eval_feature_names)
                
                missing_features_in_source = [name for name in generator_req_feature_names if name not in real_data_df_for_init.columns]
                if missing_features_in_source:
                    print(f"GeneratorPlugin Warning: The following features required for 'initial_full_feature_window' "
                          f"are missing in the preprocessed data source ('{real_data_for_eval_key}'): {missing_features_in_source}. "
                          f"These features will be zero-filled in the initial window.")
                    for mf in missing_features_in_source:
                        real_data_df_for_init[mf] = 0.0 # Add missing columns with zeros
                
                try:
                    # Select and reorder columns according to generator_req_feature_names
                    # Ensure 'DATE_TIME' is handled correctly if it's numeric or needs to be dropped/converted for the window
                    # The generator_req_feature_names should list all features including a numeric DATE_TIME if used by the model window
                    
                    # If 'DATE_TIME' is in generator_req_feature_names, it's assumed to be a numeric feature here.
                    # If it's not meant to be part of the numeric window for the Keras model, it should be excluded from
                    # generator_req_feature_names or handled appropriately.
                    
                    aligned_df_for_init = real_data_df_for_init[generator_req_feature_names]
                    
                    # Take the last 'decoder_input_window_size' rows
                    initial_segment_np = aligned_df_for_init.iloc[-decoder_input_window_size:].values.astype(np.float32)
                    
                    if initial_segment_np.shape == initial_full_feature_window.shape:
                        initial_full_feature_window = initial_segment_np
                        print(f"Successfully populated initial_full_feature_window from real data. Shape: {initial_full_feature_window.shape}")
                    else:
                        print(f"GeneratorPlugin Warning: Shape mismatch after preparing initial segment from real data. "
                              f"Expected {initial_full_feature_window.shape}, got {initial_segment_np.shape}. Using zeros.")
                except KeyError as e:
                    print(f"GeneratorPlugin Warning: KeyError when aligning features for initial_full_feature_window: {e}. "
                          f"This likely means some features in 'generator_full_feature_names_ordered' were not found in "
                          f"preprocessed data source columns ('{eval_feature_names}'). Using zeros for initial window.")
                except Exception as e:
                    print(f"GeneratorPlugin Warning: Unexpected error preparing initial_full_feature_window from real data: {e}. Using zeros.")
        else:
            if X_real_eval_source_np is None:
                print("GeneratorPlugin Warning: Preprocessed real data (X_real_eval_source_np) is None. Using zeros for initial_full_feature_window.")
            else: # X_real_eval_source_np.shape[0] < decoder_input_window_size
                print(f"GeneratorPlugin Warning: Not enough rows in preprocessed real data "
                      f"({X_real_eval_source_np.shape[0]}) to fill initial_full_feature_window "
                      f"of size {decoder_input_window_size}. Using zeros.")
        
        # Final check on DATE_TIME column if it's part of the feature set for the generator's window
        if 'DATE_TIME' in generator_req_feature_names:
            dt_idx = generator_req_feature_names.index('DATE_TIME')
            # Ensure this column is numeric. If it was populated from real data, PreprocessorPlugin should have handled it.
            # If it's still zeros, that's fine. If it came from real data and isn't numeric, it's an issue.
            if not np.issubdtype(initial_full_feature_window[:, dt_idx].dtype, np.number):
                print(f"GeneratorPlugin Warning: 'DATE_TIME' column (index {dt_idx}) in initial_full_feature_window is not numeric (dtype: {initial_full_feature_window[:, dt_idx].dtype}). Replacing with zeros.")
                initial_full_feature_window[:, dt_idx] = 0.0


        print(f"Using initial_full_feature_window with shape: {initial_full_feature_window.shape}. Sum of first row: {np.sum(initial_full_feature_window[0]) if initial_full_feature_window.size > 0 else 'N/A'}")

        # Call FeederPlugin to get Z, conditional_data, context_h for each step
        print(f"Generating feeder outputs for {sequence_length_T} steps...")
        feeder_outputs_sequence = feeder_plugin.generate(
            n_ticks_to_generate=sequence_length_T,
            target_datetimes=target_datetimes_for_generation # Pass the pandas Series of Timestamps
        )
        
        if not feeder_outputs_sequence or len(feeder_outputs_sequence) != sequence_length_T:
            raise RuntimeError(f"FeederPlugin did not return the expected number of outputs. Expected {sequence_length_T}, got {len(feeder_outputs_sequence) if feeder_outputs_sequence else 0}.")

        print("Generating full synthetic sequence via GeneratorPlugin...")
        # GeneratorPlugin's generate method now handles all feature generation including TIs
        generated_full_sequence_batch = generator_plugin.generate(
            feeder_outputs_sequence=feeder_outputs_sequence,
            sequence_length_T=sequence_length_T,
            initial_full_feature_window=initial_full_feature_window
        ) # Expected output shape: (1, sequence_length_T, num_all_features)

        if not (isinstance(generated_full_sequence_batch, np.ndarray) and generated_full_sequence_batch.ndim == 3):
            raise ValueError(f"GeneratorPlugin output has unexpected shape or type: {generated_full_sequence_batch.shape if isinstance(generated_full_sequence_batch, np.ndarray) else type(generated_full_sequence_batch)}")

        X_syn_processed_np = generated_full_sequence_batch[0] # Shape: (sequence_length_T, num_all_features)
        print(f"Generated full synthetic sequence with shape: {X_syn_processed_np.shape}")

        # --- Technical Indicator calculation is now INSIDE GeneratorPlugin ---
        # The X_syn_processed_np already contains all features, incluyendo TIs.

        # --- Prepare Real Data Segment for Evaluation ---
        # Ensure the real data segment matches the structure of X_syn_processed
        # The `X_real_eval_source_np` should ideally already contain all these features
        # as processed by `PreprocessorPlugin`.
        
        # We need to select the portion of X_real_eval_source_np que se alinea con sequence_length_T
        # If generating more synthetic samples than available real evaluation data, clamp real data length.
        num_real_eval_samples_available = X_real_eval_source_np.shape[0]
        eval_segment_length = min(sequence_length_T, num_real_eval_samples_available)
        
        if sequence_length_T > num_real_eval_samples_available:
            print(f"WARNING: Number of generated synthetic samples ({sequence_length_T}) exceeds available real evaluation samples ({num_real_eval_samples_available}). Evaluation will use the first {eval_segment_length} samples of both.")
        
        X_real_eval_segment_np = X_real_eval_source_np[:eval_segment_length, :]
        # Also slice the synthetic data if it was longer than available real data for eval
        X_syn_processed_for_eval_np = X_syn_processed_np[:eval_segment_length, :]


        # Feature names for synthetic data come from GeneratorPlugin's config
        synthetic_feature_names = generator_plugin.params.get("full_feature_names_ordered")
        if not synthetic_feature_names or len(synthetic_feature_names) != X_syn_processed_for_eval_np.shape[1]:
            raise ValueError("Mismatch between 'full_feature_names_ordered' in GeneratorPlugin and generated data columns.")

        # `eval_feature_names` (from preprocessor) is the reference for real data columns
        # and for selecting/ordering columns for evaluation.
        
        final_synthetic_data_for_eval_df = pd.DataFrame(X_syn_processed_for_eval_np, columns=synthetic_feature_names)
        final_real_data_for_eval_df = pd.DataFrame(X_real_eval_segment_np, columns=eval_feature_names)

        # Align columns of synthetic data to match the order and selection of eval_feature_names from real data
        common_features = [feat_name for feat_name in eval_feature_names if feat_name in final_synthetic_data_for_eval_df.columns]
        
        if not common_features:
            raise ValueError("No common features found between generated synthetic data and real evaluation feature names. Check feature naming and TI calculation.")

        final_synthetic_data_for_eval_aligned_df = final_synthetic_data_for_eval_df[common_features]
        final_real_data_for_eval_aligned_df = final_real_data_for_eval_df[common_features]
        aligned_eval_feature_names = common_features

        print(f"Shape of final synthetic data for eval: {final_synthetic_data_for_eval_aligned_df.shape}")
        print(f"Shape of final real data for eval: {final_real_data_for_eval_aligned_df.shape}")
        print(f"Number of features for evaluation: {len(aligned_eval_feature_names)}. Names (first 10): {aligned_eval_feature_names[:10]}...")

        # --- Save the generated (aligned) synthetic data ---
        synthetic_data_output_file = config.get("synthetic_data_output_file", "examples/results/generated_synthetic_data.csv")
        os.makedirs(os.path.dirname(synthetic_data_output_file), exist_ok=True)
        
        # Create a DataFrame to save, including the DATE_TIME column
        # Use the target_datetimes_for_generation, sliced to match eval_segment_length
        datetimes_for_output_df = target_datetimes_for_generation.iloc[:eval_segment_length].reset_index(drop=True)
        
        # Ensure the DataFrame for saving has the DATE_TIME column first
        output_df_with_datetime = final_synthetic_data_for_eval_aligned_df.copy()
        output_df_with_datetime.insert(0, config.get("datetime_col_name", "DATE_TIME"), datetimes_for_output_df)
        
        output_df_with_datetime.to_csv(synthetic_data_output_file, index=False)
        print(f"✔︎ Generated (aligned) synthetic data saved to {synthetic_data_output_file}")


        # 3. Evaluate synthetic data using the EvaluatorPlugin
        print("Evaluating synthetic data via EvaluatorPlugin...")
        metrics = evaluator_plugin.evaluate(
            synthetic_data=final_synthetic_data_for_eval_aligned_df.values, # Ensure keyword matches method
            real_data_processed=final_real_data_for_eval_aligned_df.values, # Ensure keyword matches method
            feature_names=aligned_eval_feature_names,
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
    # --- END OF MAIN FUNCTION ---

# --- ADD SCRIPT EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nINFO: Execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e_global:
        print(f"❌ CRITICAL GLOBAL ERROR: An unhandled exception occurred outside main function execution: {e_global}")
        traceback.print_exc() # This will now work correctly
        sys.exit(1)
