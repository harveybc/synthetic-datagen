#!/usr/bin/env python3
"""
main.py

Punto de entrada de la aplicación de predicción de EUR/USD. Este script orquesta:
    - La carga y fusión de configuraciones (CLI, archivos locales y remotos).
    - La inicialización de los plugins: Predictor, Optimizer, Pipeline y Preprocessor.
    - La selección entre ejecutar la optimización de hiperparámetros o entrenar y evaluar directamente.
    - El guardado de la configuración resultante de forma local y/o remota.
"""

import os # Ensure os is imported
import sys # Ensure sys is imported
import traceback 
import json # Ensure json is imported
import pandas as pd # Ensure pandas is imported
import numpy as np # Ensure numpy is imported
import tempfile # ADD THIS IMPORT

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

# Define the target CSV column order based on normalized_d2.csv
# This should be the exact header string from your file, split into a list.
TARGET_CSV_COLUMNS = [
    "DATE_TIME","RSI","MACD","MACD_Histogram","MACD_Signal","EMA",
    "Stochastic_%K","Stochastic_%D","ADX","DI+","DI-","ATR","CCI",
    "WilliamsR","Momentum","ROC","OPEN","HIGH","LOW","CLOSE",
    "BC-BO","BH-BL","BH-BO","BO-BL","S&P500_Close","vix_close",
    "CLOSE_15m_tick_1","CLOSE_15m_tick_2","CLOSE_15m_tick_3","CLOSE_15m_tick_4",
    "CLOSE_15m_tick_5","CLOSE_15m_tick_6","CLOSE_15m_tick_7","CLOSE_15m_tick_8",
    "CLOSE_30m_tick_1","CLOSE_30m_tick_2","CLOSE_30m_tick_3","CLOSE_30m_tick_4",
    "CLOSE_30m_tick_5","CLOSE_30m_tick_6","CLOSE_30m_tick_7","CLOSE_30m_tick_8",
    "day_of_month","hour_of_day","day_of_week"
]

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
    # GAN TRAINING OR VAE‐GENERATE + EVALUATE / PREPEND
    # -------------------------------------------------------------------------
    
    is_gan_training_mode = config.get("gan_training_mode", False) 
    is_hyperparam_opt_mode = config.get("hyperparameter_optimization_mode", False) and config.get("run_hyperparameter_optimization", True)

    if is_hyperparam_opt_mode:
        print("▶ Running hyperparameter optimization with Optimizer Plugin…")
        try:
            optimal_params = optimizer_plugin.optimize(
                feeder_plugin,
                generator_plugin,
                evaluator_plugin,
                config
            )
            print("✔︎ Hyperparameter optimization completed.")
            sys.exit(0) 
        except Exception as e:
            print(f"❌ Hyperparameter optimization failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # If not in hyperparameter optimization mode, proceed to generation/prepending.
    print("▶ Starting data generation and prepending process...")

    try:
        # Configuration for paths and sizes
        x_train_file_path = config["x_train_file"]
        if not x_train_file_path or not os.path.exists(x_train_file_path): # Added check for existence
            raise FileNotFoundError(f"x_train_file not found or not configured: {x_train_file_path}")
            
        max_steps_train_real = config["max_steps_train"] # How much of x_train_file to use
        n_samples_synthetic = config.get("n_samples", config.get("num_synthetic_samples_to_generate", 0)) # Use n_samples, fallback
        datetime_col_name = config.get("datetime_col_name", "DATE_TIME")
        dataset_periodicity = config.get("dataset_periodicity", "1h")
        decoder_input_window_size = generator_plugin.params.get("decoder_input_window_size")
        generator_full_feature_names = generator_plugin.params.get("full_feature_names_ordered", [])
        if not generator_full_feature_names:
            raise ValueError("GeneratorPlugin 'full_feature_names_ordered' is not configured.")

        # 1. Load and Preprocess the full x_train_file to get features for the initial window and real segment
        print(f"Preprocessing full '{x_train_file_path}' to extract initial window for generator and real segment...")
        
        # --- Corrected Preprocessor Call for x_train_file ---
        # Pass a copy of the main config. The preprocessor should be robust enough
        # to handle its configured sources or ignore irrelevant ones for this task.
        # We are primarily interested in its 'x_train' related outputs.
        config_for_full_train_preprocessing = config.copy()
        
        # Apply workarounds for preprocessor if necessary (e.g., STL window)
        # This part is from your existing code, ensure preprocessor_plugin has plugin_params
        if hasattr(preprocessor_plugin, 'plugin_params'): # Check if preprocessor_plugin has plugin_params
            if not config_for_full_train_preprocessing.get('use_stl', False):
                if 'stl_window' in preprocessor_plugin.plugin_params or 'stl_window' in config_for_full_train_preprocessing:
                    config_for_full_train_preprocessing['stl_window'] = 0 
        
        print(f"DEBUG main.py: Calling preprocessor_plugin.run_preprocessing with full config for x_train_file: {x_train_file_path}")
        all_processed_datasets = preprocessor_plugin.run_preprocessing(config=config_for_full_train_preprocessing)
        
        X_train_processed_full_anytype = all_processed_datasets.get("x_train")
        datetimes_train_processed_full_anytype = all_processed_datasets.get("x_train_dates")
        processed_train_feature_names = all_processed_datasets.get("feature_names", []) 

        if X_train_processed_full_anytype is None:
            raise ValueError(f"Preprocessor did not return 'x_train' data after processing '{x_train_file_path}'. Check preprocessor logic and output keys.")
        if datetimes_train_processed_full_anytype is None:
            raise ValueError(f"Preprocessor did not return 'x_train_dates' after processing '{x_train_file_path}'.")

        datetimes_train_processed_full_series = pd.Series(pd.to_datetime(datetimes_train_processed_full_anytype))

        if isinstance(X_train_processed_full_anytype, pd.DataFrame):
            if not processed_train_feature_names: 
                processed_train_feature_names = list(X_train_processed_full_anytype.columns)
            X_train_processed_full_np = X_train_processed_full_anytype.values.astype(np.float32)
        elif isinstance(X_train_processed_full_anytype, np.ndarray):
            X_train_processed_full_np = X_train_processed_full_anytype.astype(np.float32)
        else:
            raise TypeError(f"Preprocessor 'x_train' output is of unexpected type: {type(X_train_processed_full_anytype)}")

        # --- ADDED: Handle 3D preprocessor output for x_train ---
        if X_train_processed_full_np.ndim == 3:
            print(f"DEBUG main.py: X_train_processed_full_np is 3D {X_train_processed_full_np.shape}. Taking last element of sequence dimension.")
            # Assume (num_samples, sequence_length, num_features) -> (num_samples, num_features)
            # This takes the features from the last time step of each sequence.
            X_train_processed_full_np = X_train_processed_full_np[:, -1, :]
            print(f"DEBUG main.py: X_train_processed_full_np reshaped to 2D: {X_train_processed_full_np.shape}")
        # --- END ADDED ---

        if not processed_train_feature_names and X_train_processed_full_np.ndim > 1 and X_train_processed_full_np.shape[0] > 0:
            num_feats = X_train_processed_full_np.shape[-1]
            processed_train_feature_names = [f"feature_{i}" for i in range(num_feats)]
            print(f"Warning: Using generic feature names for processed x_train data as 'feature_names' not in preprocessor output or derivable from DataFrame.")
        
        print(f"DEBUG main.py: Processed x_train data shape: {X_train_processed_full_np.shape}, Num features: {len(processed_train_feature_names)}, First few features: {processed_train_feature_names[:5]}")

        if X_train_processed_full_np.shape[0] < decoder_input_window_size:
            raise ValueError(f"Processed x_train data (length {X_train_processed_full_np.shape[0]}) is shorter than decoder window size ({decoder_input_window_size}). Cannot extract initial window.")

        # 2. Prepare initial_full_feature_window from the END of processed x_train_file
        df_temp_processed_train = pd.DataFrame(X_train_processed_full_np, columns=processed_train_feature_names)
        initial_window_raw_df_from_processed_train = df_temp_processed_train.iloc[-decoder_input_window_size:]
        
        initial_full_feature_window_for_gen_df = pd.DataFrame(
            np.nan, 
            index=range(decoder_input_window_size), 
            columns=generator_full_feature_names
        ).astype(np.float32)

        for col_name_gen in generator_full_feature_names:
            if col_name_gen in initial_window_raw_df_from_processed_train.columns:
                initial_full_feature_window_for_gen_df[col_name_gen] = initial_window_raw_df_from_processed_train[col_name_gen].values
        
        initial_full_feature_window_for_gen = initial_full_feature_window_for_gen_df.fillna(0.0).values 
        print(f"Prepared initial_full_feature_window for generator with shape: {initial_full_feature_window_for_gen.shape} from end of processed {x_train_file_path}")
        
        # 3. Prepare Real Data Segment for Final Output (from START of processed x_train_file)
        num_real_rows_for_output = min(max_steps_train_real, X_train_processed_full_np.shape[0])
        X_real_segment_for_output_np = X_train_processed_full_np[:num_real_rows_for_output]
        datetimes_real_segment_for_output = datetimes_train_processed_full_series.iloc[:num_real_rows_for_output].reset_index(drop=True)
        
        first_dt_real_segment = pd.Timestamp.now(tz='UTC') # Default if no real rows, make it timezone-aware
        if not datetimes_real_segment_for_output.empty:
            first_dt_real_segment = datetimes_real_segment_for_output.iloc[0]
        elif num_real_rows_for_output > 0 : 
             raise ValueError("Real data segment for output has no datetimes, but rows were expected.")
        
        print(f"Real data segment for output: {num_real_rows_for_output} rows, starting {first_dt_real_segment if num_real_rows_for_output > 0 else 'N/A (no real rows)'}")

        # 4. Generate Datetimes for Synthetic Data (to be Prepended)
        time_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(dataset_periodicity) or pd.Timedelta(hours=1))
        
        final_synthetic_target_datetimes_objs = []
        if n_samples_synthetic > 0:
            final_synthetic_target_datetimes_objs = generate_synthetic_datetimes_before_real( 
                real_start_dt=first_dt_real_segment, 
                num_synthetic_samples=n_samples_synthetic,
                time_delta_val=time_delta,
                periodicity_str_val=dataset_periodicity
            )
            if len(final_synthetic_target_datetimes_objs) != n_samples_synthetic:
                raise ValueError(f"Generated {len(final_synthetic_target_datetimes_objs)} synthetic datetimes, but expected {n_samples_synthetic}.")
        
        final_synthetic_target_datetimes_series = pd.Series(final_synthetic_target_datetimes_objs, dtype='datetime64[ns]')
        if n_samples_synthetic > 0 and not final_synthetic_target_datetimes_series.empty:
            print(f"Generated {len(final_synthetic_target_datetimes_series)} datetimes for synthetic data, from {final_synthetic_target_datetimes_series.iloc[0]} to {final_synthetic_target_datetimes_series.iloc[-1]}.")
        elif n_samples_synthetic > 0 and final_synthetic_target_datetimes_series.empty:
             print("Warning: Requested synthetic samples but generated empty datetime series.")
        else:
            print("No synthetic samples requested (n_samples_synthetic is 0).")

        # 5. Generate Synthetic Values (only if n_samples_synthetic > 0)
        X_syn_generated_values_np = np.array([]).reshape(0, len(generator_full_feature_names)) 
        if n_samples_synthetic > 0 and not final_synthetic_target_datetimes_series.empty:
            print(f"Generating feeder outputs for {n_samples_synthetic} synthetic steps...")
            feeder_outputs_sequence_synthetic = feeder_plugin.generate(
                n_ticks_to_generate=n_samples_synthetic,
                target_datetimes=final_synthetic_target_datetimes_series
            )

            print("Generating synthetic feature values via GeneratorPlugin...")
            generated_output_from_plugin = generator_plugin.generate(
                feeder_outputs_sequence=feeder_outputs_sequence_synthetic,
                sequence_length_T=n_samples_synthetic,
                initial_full_feature_window=initial_full_feature_window_for_gen
            )
            if isinstance(generated_output_from_plugin, list) and len(generated_output_from_plugin) == 1 and isinstance(generated_output_from_plugin[0], np.ndarray):
                 X_syn_generated_values_np = generated_output_from_plugin[0]
            elif isinstance(generated_output_from_plugin, np.ndarray) and generated_output_from_plugin.ndim == 3 and generated_output_from_plugin.shape[0] == 1 : 
                 X_syn_generated_values_np = generated_output_from_plugin[0]
            elif isinstance(generated_output_from_plugin, np.ndarray) and generated_output_from_plugin.ndim == 2 and generated_output_from_plugin.shape[0] == n_samples_synthetic :
                 X_syn_generated_values_np = generated_output_from_plugin
            else:
                raise TypeError(f"Unexpected output type or shape from generator_plugin.generate: {type(generated_output_from_plugin)}, shape if np.array: {getattr(generated_output_from_plugin, 'shape', 'N/A')}")
        elif n_samples_synthetic > 0:
            print("Skipping synthetic value generation as synthetic datetime series is empty.")


        # 6. Create and Save Synthetic-Only DataFrame (using TARGET_CSV_COLUMNS)
        df_synthetic_generated_full_features = pd.DataFrame(X_syn_generated_values_np, columns=generator_full_feature_names)
        if n_samples_synthetic > 0 and not final_synthetic_target_datetimes_series.empty:
            # Ensure datetimes match the number of rows in X_syn_generated_values_np
            if len(final_synthetic_target_datetimes_series) == df_synthetic_generated_full_features.shape[0]:
                # If DATE_TIME is already a column from generator_full_feature_names (as a placeholder), overwrite it.
                # This is expected because config.py defines "DATE_TIME" in generator_full_feature_names_ordered.
                if datetime_col_name in df_synthetic_generated_full_features.columns:
                    df_synthetic_generated_full_features[datetime_col_name] = final_synthetic_target_datetimes_series.values
                else:
                    # Fallback: if DATE_TIME was somehow not in generator_full_feature_names, then insert.
                    df_synthetic_generated_full_features.insert(0, datetime_col_name, final_synthetic_target_datetimes_series.values)
            else:
                print(f"Warning: Mismatch between synthetic datetimes ({len(final_synthetic_target_datetimes_series)}) and generated values ({df_synthetic_generated_full_features.shape[0]}). Datetime column might be misaligned or omitted for synthetic data.")
        
        output_df_synthetic_final = pd.DataFrame(columns=TARGET_CSV_COLUMNS)
        if not df_synthetic_generated_full_features.empty:
            for col_target_name in TARGET_CSV_COLUMNS:
                if col_target_name in df_synthetic_generated_full_features.columns:
                    output_df_synthetic_final[col_target_name] = df_synthetic_generated_full_features[col_target_name]
                else: 
                    output_df_synthetic_final[col_target_name] = np.nan
        
        synthetic_data_output_path = config.get("synthetic_data_output_file", "examples/results/generated_synthetic_data.csv")
        os.makedirs(os.path.dirname(synthetic_data_output_path), exist_ok=True)
        output_df_synthetic_final.to_csv(synthetic_data_output_path, index=False, na_rep='NaN')
        print(f"✔︎ Synthetic-only data saved to {synthetic_data_output_path} (Shape: {output_df_synthetic_final.shape})")

        # 7. Create Real Data DataFrame for Output (using TARGET_CSV_COLUMNS)
        df_real_segment_processed_full_features = pd.DataFrame(X_real_segment_for_output_np, columns=processed_train_feature_names)
        if num_real_rows_for_output > 0 and not datetimes_real_segment_for_output.empty:
            if len(datetimes_real_segment_for_output) == df_real_segment_processed_full_features.shape[0]:
                # For the real data segment, 'processed_train_feature_names' (from preprocessor)
                # typically does NOT include 'DATE_TIME', so 'insert' is correct here.
                if datetime_col_name not in df_real_segment_processed_full_features.columns:
                    df_real_segment_processed_full_features.insert(0, datetime_col_name, datetimes_real_segment_for_output.values)
                else:
                    # This case would be unusual if preprocessor doesn't output DATE_TIME as a feature
                    df_real_segment_processed_full_features[datetime_col_name] = datetimes_real_segment_for_output.values
            else:
                print(f"Warning: Mismatch between real datetimes ({len(datetimes_real_segment_for_output)}) and real segment values ({df_real_segment_processed_full_features.shape[0]}). Datetime column might be misaligned or omitted for real data segment.")


        output_df_real_final_segment = pd.DataFrame(columns=TARGET_CSV_COLUMNS)
        if not df_real_segment_processed_full_features.empty:
            for col_target_name in TARGET_CSV_COLUMNS:
                if col_target_name in df_real_segment_processed_full_features.columns:
                    output_df_real_final_segment[col_target_name] = df_real_segment_processed_full_features[col_target_name]
                else: 
                    output_df_real_final_segment[col_target_name] = np.nan
        
        print(f"DEBUG main.py: Real data segment for output DF shape: {output_df_real_final_segment.shape}")

        # 8. Combine and Save Final Output (Prepended)
        if output_df_synthetic_final.empty and output_df_real_final_segment.empty:
            print("WARNING: Both synthetic and real data segments are empty. Output file will be empty or header-only.")
            df_combined_prepended = pd.DataFrame(columns=TARGET_CSV_COLUMNS) 
        elif output_df_synthetic_final.empty:
            df_combined_prepended = output_df_real_final_segment.reindex(columns=TARGET_CSV_COLUMNS)
        elif output_df_real_final_segment.empty:
            df_combined_prepended = output_df_synthetic_final.reindex(columns=TARGET_CSV_COLUMNS)
        else:
            df_combined_prepended = pd.concat(
                [output_df_synthetic_final.reindex(columns=TARGET_CSV_COLUMNS), 
                 output_df_real_final_segment.reindex(columns=TARGET_CSV_COLUMNS)], 
                ignore_index=True
            )
        
        final_output_path = config.get("output_file", "examples/results/prepended_data.csv")
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        df_combined_prepended.to_csv(final_output_path, index=False, na_rep='NaN')
        print(f"✔︎ Combined data (synthetic prepended to real) saved to {final_output_path} (Shape: {df_combined_prepended.shape})")

        # 9. Proceed with Evaluation (using x_test or x_validation as per original logic)
        print("\n▶ Starting independent evaluation process using evaluation dataset (e.g., x_test)...")
        
        eval_data_source_path = config.get('x_test_file', config.get('x_validation_file'))
        if not eval_data_source_path or not os.path.exists(eval_data_source_path):
            print(f"WARNING: Evaluation data source (x_test_file or x_validation_file: '{eval_data_source_path}') not found or not configured. Skipping evaluation.")
        else:
            config_for_eval_preprocessing = config.copy()
            if hasattr(preprocessor_plugin, 'plugin_params'): # Check if preprocessor_plugin has plugin_params
                if not config_for_eval_preprocessing.get('use_stl', False):
                     if 'stl_window' in preprocessor_plugin.plugin_params or 'stl_window' in config_for_eval_preprocessing:
                        config_for_eval_preprocessing['stl_window'] = 0
            
            print(f"DEBUG main.py: Calling preprocessor_plugin.run_preprocessing for evaluation data: {eval_data_source_path}")
            eval_datasets = preprocessor_plugin.run_preprocessing(config=config_for_eval_preprocessing)
            
            X_real_eval_source_anytype = eval_datasets.get("x_test", eval_datasets.get("x_validation"))
            datetimes_eval_source_anytype = eval_datasets.get("x_test_dates", eval_datasets.get("x_val_dates")) 
            eval_feature_names = eval_datasets.get("feature_names", []) 

            if X_real_eval_source_anytype is None or datetimes_eval_source_anytype is None:
                print(f"WARNING: No suitable x_test or x_validation data found from preprocessor output for path '{eval_data_source_path}'. Skipping evaluation.")
            else:
                datetimes_eval_source = pd.Series(pd.to_datetime(datetimes_eval_source_anytype))
                if isinstance(X_real_eval_source_anytype, pd.DataFrame):
                    if not eval_feature_names: eval_feature_names = list(X_real_eval_source_anytype.columns)
                    X_real_eval_source_np = X_real_eval_source_anytype.values.astype(np.float32)
                elif isinstance(X_real_eval_source_anytype, np.ndarray):
                    X_real_eval_source_np = X_real_eval_source_anytype.astype(np.float32)
                else:
                    raise TypeError(f"Preprocessor eval output ('x_test' or 'x_validation') is of unexpected type: {type(X_real_eval_source_anytype)}")

                # --- ADDED: Handle 3D preprocessor output for eval data ---
                if X_real_eval_source_np.ndim == 3:
                    print(f"DEBUG main.py: X_real_eval_source_np is 3D {X_real_eval_source_np.shape}. Taking last element of sequence dimension.")
                    # Assume (num_samples, sequence_length, num_features) -> (num_samples, num_features)
                    X_real_eval_source_np = X_real_eval_source_np[:, -1, :]
                    print(f"DEBUG main.py: X_real_eval_source_np reshaped to 2D: {X_real_eval_source_np.shape}")
                # --- END ADDED ---

                if not eval_feature_names and X_real_eval_source_np.ndim > 1 and X_real_eval_source_np.shape[0] > 0: 
                    eval_feature_names = [f"feature_{i}" for i in range(X_real_eval_source_np.shape[-1])]
                
                num_eval_samples = X_real_eval_source_np.shape[0]
                if num_eval_samples == 0:
                    print("WARNING: Evaluation data source is empty. Skipping evaluation generation.")
                else:
                    print(f"Generating {num_eval_samples} synthetic samples for evaluation against processed eval data...")
                    
                    feeder_outputs_for_eval = feeder_plugin.generate(
                        n_ticks_to_generate=num_eval_samples,
                        target_datetimes=datetimes_eval_source 
                    )
                    
                    initial_window_for_eval_gen_df = pd.DataFrame(
                        np.nan, 
                        index=range(decoder_input_window_size), 
                        columns=generator_full_feature_names
                    ).astype(np.float32)

                    if X_real_eval_source_np.shape[0] >= decoder_input_window_size:
                        df_temp_processed_eval = pd.DataFrame(X_real_eval_source_np, columns=eval_feature_names)
                        initial_window_raw_df_from_eval = df_temp_processed_eval.iloc[:decoder_input_window_size] 
                        for col_name_gen_eval in generator_full_feature_names:
                            if col_name_gen_eval in initial_window_raw_df_from_eval.columns:
                                initial_window_for_eval_gen_df[col_name_gen_eval] = initial_window_raw_df_from_eval[col_name_gen_eval].values
                    
                    initial_window_for_eval_gen = initial_window_for_eval_gen_df.fillna(0.0).values
                    
                    generated_eval_output = generator_plugin.generate(
                        feeder_outputs_sequence=feeder_outputs_for_eval,
                        sequence_length_T=num_eval_samples,
                        initial_full_feature_window=initial_window_for_eval_gen
                    )
                    if isinstance(generated_eval_output, list) and len(generated_eval_output) == 1 and isinstance(generated_eval_output[0], np.ndarray):
                        X_syn_for_eval_np = generated_eval_output[0]
                    elif isinstance(generated_eval_output, np.ndarray) and generated_eval_output.ndim == 3 and generated_eval_output.shape[0] == 1 : 
                        X_syn_for_eval_np = generated_eval_output[0]
                    elif isinstance(generated_eval_output, np.ndarray) and generated_eval_output.ndim == 2 and generated_eval_output.shape[0] == num_eval_samples :
                        X_syn_for_eval_np = generated_eval_output
                    else:
                        raise TypeError(f"Unexpected output type or shape from generator_plugin.generate for eval: {type(generated_eval_output)}")

                    df_syn_for_eval_full_features = pd.DataFrame(X_syn_for_eval_np, columns=generator_full_feature_names)
                    df_real_for_eval_processed_features = pd.DataFrame(X_real_eval_source_np, columns=eval_feature_names)

                    common_features_eval = [f_name for f_name in eval_feature_names if f_name in df_syn_for_eval_full_features.columns]
                    if not common_features_eval:
                        raise ValueError("No common features between synthetic data for eval (generator output) and real eval data (preprocessor output).")

                    df_syn_for_eval_aligned = df_syn_for_eval_full_features[common_features_eval]
                    df_real_for_eval_aligned = df_real_for_eval_processed_features[common_features_eval]
                    
                    print(f"Shape of synthetic data for evaluation (aligned): {df_syn_for_eval_aligned.shape}")
                    print(f"Shape of real data for evaluation (aligned): {df_real_for_eval_aligned.shape}")

                    print("Evaluating synthetic data (independent batch) via EvaluatorPlugin...")
                    metrics = evaluator_plugin.evaluate(
                        synthetic_data=df_syn_for_eval_aligned.values,
                        real_data_processed=df_real_for_eval_aligned.values,
                        feature_names=common_features_eval,
                        config=config 
                    )
                    print("✔︎ Evaluation completed.")
                    metrics_output_file = config.get("metrics_file", "examples/results/evaluation_metrics.json") 
                    os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)
                    with open(metrics_output_file, "w") as f:
                        json.dump(metrics, f, indent=4)
                    print(f"✔︎ Evaluation metrics saved to {metrics_output_file}")

    except Exception as e:
        print(f"❌ Synthetic data generation, prepending, or evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Save final configuration
    if args.save_config:
        try:
            config_save_path = args.save_config if isinstance(args.save_config, str) else "examples/results/final_config.json"
            save_config(config_save_path, config)
            print(f"✔︎ Configuration saved to {config_save_path}")
        except Exception as e:
            print(f"WARNING: Failed to save configuration: {e}. Proceeding without saving config.")
    
    print("Fin de la ejecución del script main.py.")

# --- ADD SCRIPT EXECUTION BLOCK ---
if __name__ == "__main__":
    # Ensure necessary imports for the script are at the top level of main.py
    # import pandas as pd # Already imported globally
    # import numpy as np # Already imported globally
    # import os # Already imported globally
    # import sys # Already imported globally
    # import traceback # Already imported globally
    # import json # Already imported globally
    from datetime import datetime, timedelta
    # ... other necessary global imports for main ...

    try:
        main()
    except KeyboardInterrupt:
        print("\nINFO: Execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e_global:
        print(f"❌ CRITICAL GLOBAL ERROR: An unhandled exception occurred outside main function execution: {e_global}")
        traceback.print_exc() # This will now work correctly
        sys.exit(1)
