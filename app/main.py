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
        max_steps_train_real = config["max_steps_train"] # How much of x_train_file to use
        n_samples_synthetic = config["n_samples"]       # How many synthetic samples to generate
        datetime_col_name = config.get("datetime_col_name", "DATE_TIME")
        dataset_periodicity = config.get("dataset_periodicity", "1h")
        decoder_input_window_size = generator_plugin.params.get("decoder_input_window_size")
        generator_full_feature_names = generator_plugin.params.get("full_feature_names_ordered", [])
        if not generator_full_feature_names:
            raise ValueError("GeneratorPlugin 'full_feature_names_ordered' is not configured.")

        # 1. Load and Preprocess the full x_train_file to get features for the initial window
        print(f"Preprocessing full '{x_train_file_path}' to extract initial window for generator...")
        config_for_train_preprocessing = config.copy()
        # Ensure preprocessor targets x_train_file for its primary data source
        # This might involve setting 'real_data_file' or specific keys the preprocessor uses
        # For stl_preprocessor, it uses 'real_data_file', 'x_train_file', etc. from config.
        # We want it to process the content of `x_train_file_path`.
        # A simple way is to ensure the relevant keys in `config_for_train_preprocessing` point to `x_train_file_path`.
        # Note: The preprocessor might also use x_validation_file, x_test_file for its internal logic
        # if it's designed to output multiple datasets. We are interested in its processing of x_train_file.

        # --- Create a temporary config for preprocessor to focus on x_train_file ---
        # This ensures that when preprocessor_plugin.run_preprocessing is called,
        # it primarily processes the x_train_file_path for 'x_train' output.
        temp_config_for_preprocessing_x_train = config.copy()
        temp_config_for_preprocessing_x_train['real_data_file'] = x_train_file_path # stl_preprocessor uses this
        temp_config_for_preprocessing_x_train['x_train_file'] = x_train_file_path
        # Nullify other data file paths for this specific call if they might interfere,
        # or ensure your preprocessor correctly prioritizes/isolates x_train processing.
        temp_config_for_preprocessing_x_train['x_validation_file'] = None 
        temp_config_for_preprocessing_x_train['x_test_file'] = None
        temp_config_for_preprocessing_x_train['y_train_file'] = None # If y_train is not needed for x_train feature processing
        temp_config_for_preprocessing_x_train['y_validation_file'] = None
        temp_config_for_preprocessing_x_train['y_test_file'] = None
        
        # Apply workarounds for preprocessor if necessary (similar to eval data preprocessing)
        # WORKAROUND 1 (STL Window)
        if not temp_config_for_preprocessing_x_train.get('use_stl', False):
            if 'stl_window' in preprocessor_plugin.plugin_params or 'stl_window' in temp_config_for_preprocessing_x_train:
                temp_config_for_preprocessing_x_train['stl_window'] = 0 # Disable if use_stl is false
        # WORKAROUND 2 (Even Rows) - This is complex, ensure your x_train_file is robust or handle carefully
        # For simplicity, this example assumes x_train_file_path is suitable or preprocessor handles it.

        train_datasets_processed = preprocessor_plugin.run_preprocessing(config=temp_config_for_preprocessing_x_train)
        
        X_train_processed_full_np = train_datasets_processed.get("x_train")
        datetimes_train_processed_full_series = pd.Series(pd.to_datetime(train_datasets_processed.get("x_train_dates")))
        processed_train_feature_names = train_datasets_processed.get("feature_names", [])

        if X_train_processed_full_np is None or datetimes_train_processed_full_series.empty:
            raise ValueError(f"Failed to preprocess '{x_train_file_path}' or extract 'x_train' data and dates.")
        if not processed_train_feature_names:
             # Attempt to get from DataFrame columns if X_train_processed_full_np is a DataFrame
            if isinstance(X_train_processed_full_np, pd.DataFrame):
                processed_train_feature_names = list(X_train_processed_full_np.columns)
            else: # Fallback if it's an ndarray and feature_names wasn't in datasets
                num_feats = X_train_processed_full_np.shape[1]
                processed_train_feature_names = [f"feature_{i}" for i in range(num_feats)]
                print(f"Warning: Using generic feature names for processed x_train data as 'feature_names' not in preprocessor output.")

        if isinstance(X_train_processed_full_np, pd.DataFrame):
            X_train_processed_full_np = X_train_processed_full_np.values
        
        if X_train_processed_full_np.shape[0] < decoder_input_window_size:
            raise ValueError(f"Processed x_train data (length {X_train_processed_full_np.shape[0]}) is shorter than decoder window size ({decoder_input_window_size}). Cannot extract initial window.")

        # 2. Prepare initial_full_feature_window from the END of processed x_train_file
        # Ensure features align with generator_full_feature_names
        df_temp_processed_train = pd.DataFrame(X_train_processed_full_np, columns=processed_train_feature_names)
        initial_window_raw_df = df_temp_processed_train.iloc[-decoder_input_window_size:]
        
        initial_full_feature_window_for_gen = np.full((decoder_input_window_size, len(generator_full_feature_names)), np.nan, dtype=np.float32)
        initial_window_df_aligned = pd.DataFrame(initial_full_feature_window_for_gen, columns=generator_full_feature_names)

        for col_name in generator_full_feature_names:
            if col_name in initial_window_raw_df.columns:
                initial_window_df_aligned[col_name] = initial_window_raw_df[col_name].values
            elif col_name == datetime_col_name: # Handle DATE_TIME if it's a feature for generator
                 # This assumes generator expects numerical representation or it's handled internally
                 # For now, filling with index if needed, or NaN. Generator usually gets datetimes via feeder.
                 pass # Typically DATE_TIME as a feature in the window is just its float index or similar
        
        initial_full_feature_window_for_gen = initial_window_df_aligned.fillna(0.0).values # Fill NaNs, e.g. with 0
        print(f"Prepared initial_full_feature_window for generator with shape: {initial_full_feature_window_for_gen.shape} from end of processed {x_train_file_path}")
        # GeneratorPlugin's initial_denormalized_close_anchor is already loaded from x_train_file's end.

        # 3. Prepare Real Data Segment for Final Output (from START of processed x_train_file)
        num_real_rows_for_output = min(max_steps_train_real, X_train_processed_full_np.shape[0])
        X_real_segment_for_output_np = X_train_processed_full_np[:num_real_rows_for_output]
        datetimes_real_segment_for_output = datetimes_train_processed_full_series.iloc[:num_real_rows_for_output].reset_index(drop=True)
        
        if datetimes_real_segment_for_output.empty:
            raise ValueError("Real data segment for output is empty or has no datetimes.")
        first_dt_real_segment = datetimes_real_segment_for_output.iloc[0]
        
        print(f"Real data segment for output: {num_real_rows_for_output} rows, starting {first_dt_real_segment}")

        # 4. Generate Datetimes for Synthetic Data (to be Prepended)
        time_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(dataset_periodicity) or pd.Timedelta(hours=1))
        
        final_synthetic_target_datetimes_objs = generate_synthetic_datetimes_before_real(
            real_start_dt=first_dt_real_segment,
            num_synthetic_samples=n_samples_synthetic,
            time_delta_val=time_delta,
            periodicity_str_val=dataset_periodicity
        )
        if len(final_synthetic_target_datetimes_objs) != n_samples_synthetic:
            raise ValueError(f"Generated {len(final_synthetic_target_datetimes_objs)} synthetic datetimes, but expected {n_samples_synthetic}.")
        
        final_synthetic_target_datetimes_series = pd.Series(final_synthetic_target_datetimes_objs)
        print(f"Generated {len(final_synthetic_target_datetimes_series)} datetimes for synthetic data, from {final_synthetic_target_datetimes_series.iloc[0]} to {final_synthetic_target_datetimes_series.iloc[-1]}.")

        # 5. Generate Synthetic Values
        print(f"Generating feeder outputs for {n_samples_synthetic} synthetic steps...")
        feeder_outputs_sequence_synthetic = feeder_plugin.generate(
            n_ticks_to_generate=n_samples_synthetic,
            target_datetimes=final_synthetic_target_datetimes_series
        )

        print("Generating synthetic feature values via GeneratorPlugin...")
        generated_full_sequence_batch = generator_plugin.generate(
            feeder_outputs_sequence=feeder_outputs_sequence_synthetic,
            sequence_length_T=n_samples_synthetic,
            initial_full_feature_window=initial_full_feature_window_for_gen
        )
        X_syn_generated_values_np = generated_full_sequence_batch[0] # Shape: (n_samples_synthetic, num_all_features_gen)

        # 6. Create and Save Synthetic-Only DataFrame
        df_synthetic_generated = pd.DataFrame(X_syn_generated_values_np, columns=generator_full_feature_names)
        df_synthetic_generated.insert(0, datetime_col_name, final_synthetic_target_datetimes_series)
        
        output_df_synthetic_final = pd.DataFrame()
        for col_target_name in TARGET_CSV_COLUMNS:
            if col_target_name in df_synthetic_generated.columns:
                output_df_synthetic_final[col_target_name] = df_synthetic_generated[col_target_name]
            else:
                output_df_synthetic_final[col_target_name] = 0.0 # Or np.nan
        
        synthetic_data_output_path = config.get("synthetic_data_output_file", "examples/results/generated_synthetic_data.csv")
        os.makedirs(os.path.dirname(synthetic_data_output_path), exist_ok=True)
        output_df_synthetic_final.to_csv(synthetic_data_output_path, index=False, na_rep='NaN')
        print(f"✔︎ Synthetic-only data saved to {synthetic_data_output_path}")

        # 7. Create Real Data DataFrame for Output
        # Features for real segment should match generator_full_feature_names if possible, or processed_train_feature_names
        # For consistency, try to use generator_full_feature_names
        df_real_segment_processed = pd.DataFrame(X_real_segment_for_output_np, columns=processed_train_feature_names)
        df_real_segment_processed.insert(0, datetime_col_name, datetimes_real_segment_for_output)

        output_df_real_final_segment = pd.DataFrame()
        for col_target_name in TARGET_CSV_COLUMNS:
            if col_target_name in df_real_segment_processed.columns:
                output_df_real_final_segment[col_target_name] = df_real_segment_processed[col_target_name]
            else:
                # If a column in TARGET_CSV_COLUMNS is not in processed real data (e.g. a purely synthetic one)
                output_df_real_final_segment[col_target_name] = 0.0 # Or np.nan

        # 8. Combine and Save Final Output (Prepended)
        df_combined_prepended = pd.concat([output_df_synthetic_final, output_df_real_final_segment], ignore_index=True)
        
        final_output_path = config.get("output_file", "examples/results/prepended_data.csv")
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        df_combined_prepended.to_csv(final_output_path, index=False, na_rep='NaN')
        print(f"✔︎ Combined data (synthetic prepended to real) saved to {final_output_path}")

        # 9. Proceed with Evaluation (using x_test or x_validation as per original logic)
        print("\n▶ Starting independent evaluation process using evaluation dataset (e.g., x_test)...")
        # This part reuses some logic from the original main script for evaluation setup
        # Preprocess real data for evaluation (e.g., x_test)
        config_for_eval_preprocessing = config.copy() 
        # Apply workarounds if needed (similar to above)
        # ... (Workaround logic for stl_window, even rows for eval files)
        # Ensure preprocessor uses x_test_file or x_validation_file
        config_for_eval_preprocessing['real_data_file'] = config.get('x_test_file') # Or x_validation_file
        config_for_eval_preprocessing['x_train_file'] = None # Avoid reprocessing x_train here
        
        eval_datasets = preprocessor_plugin.run_preprocessing(config=config_for_eval_preprocessing)
        
        X_real_eval_source = eval_datasets.get("x_test", eval_datasets.get("x_validation"))
        datetimes_eval_source = pd.Series(pd.to_datetime(eval_datasets.get("x_test_dates", eval_datasets.get("x_val_dates"))))
        eval_feature_names = eval_datasets.get("feature_names", [])

        if X_real_eval_source is None or datetimes_eval_source.empty:
            print("WARNING: No suitable x_test or x_validation data found from preprocessor for evaluation. Skipping evaluation.")
        else:
            if isinstance(X_real_eval_source, pd.DataFrame): X_real_eval_source = X_real_eval_source.values
            if not eval_feature_names: 
                eval_feature_names = [f"feature_{i}" for i in range(X_real_eval_source.shape[1])]

            # For evaluation, we need synthetic data of the same length and features as X_real_eval_source
            # We can generate a new batch of synthetic data for this purpose, or use a portion of
            # X_syn_generated_values_np if its style is considered representative.
            # For a fair evaluation of the generator's quality, generate fresh data matching eval data length.
            
            num_eval_samples = X_real_eval_source.shape[0]
            print(f"Generating {num_eval_samples} synthetic samples for evaluation against processed eval data...")
            
            # Use datetimes from the evaluation set for this generation
            feeder_outputs_for_eval = feeder_plugin.generate(
                n_ticks_to_generate=num_eval_samples,
                target_datetimes=datetimes_eval_source
            )
            # Use a generic initial window for this separate evaluation generation
            # or the same one used for prepending if style consistency is desired.
            # For simplicity, using a new zero window or the previously prepared one.
            initial_window_for_eval_gen = np.zeros((decoder_input_window_size, len(generator_full_feature_names)), dtype=np.float32)
            # A better approach might be to use a segment from the start of X_real_eval_source if available and long enough.
            if X_real_eval_source.shape[0] >= decoder_input_window_size and X_real_eval_source.shape[1] == len(generator_full_feature_names):
                 # This assumes X_real_eval_source has the same full features as generator expects.
                 # This might require X_real_eval_source to be the output of the preprocessor that aligns with generator_full_feature_names
                 # For now, let's assume eval_feature_names from preprocessor output of x_test matches generator_full_feature_names
                 # This is a strong assumption.
                 if X_real_eval_source.shape[1] == len(generator_full_feature_names):
                    initial_window_for_eval_gen = X_real_eval_source[:decoder_input_window_size, :] # Example: use start of eval data
                 else: # Fallback if feature counts don't match
                    print(f"Warning: Eval data feature count ({X_real_eval_source.shape[1]}) "
                          f"differs from generator's expected ({len(generator_full_feature_names)}). Using zero window for eval generation.")
            
            generated_eval_batch = generator_plugin.generate(
                feeder_outputs_sequence=feeder_outputs_for_eval,
                sequence_length_T=num_eval_samples,
                initial_full_feature_window=initial_window_for_eval_gen # Or the one from x_train end
            )
            X_syn_for_eval_np = generated_eval_batch[0]

            df_syn_for_eval = pd.DataFrame(X_syn_for_eval_np, columns=generator_full_feature_names)
            df_real_for_eval = pd.DataFrame(X_real_eval_source, columns=eval_feature_names)

            # Align columns for evaluation
            common_features_eval = [f_name for f_name in eval_feature_names if f_name in df_syn_for_eval.columns]
            if not common_features_eval:
                raise ValueError("No common features between synthetic data for eval and real eval data.")

            df_syn_for_eval_aligned = df_syn_for_eval[common_features_eval]
            df_real_for_eval_aligned = df_real_for_eval[common_features_eval]
            
            print(f"Shape of synthetic data for evaluation: {df_syn_for_eval_aligned.shape}")
            print(f"Shape of real data for evaluation: {df_real_for_eval_aligned.shape}")

            print("Evaluating synthetic data (independent batch) via EvaluatorPlugin...")
            metrics = evaluator_plugin.evaluate(
                synthetic_data=df_syn_for_eval_aligned.values,
                real_data_processed=df_real_for_eval_aligned.values,
                feature_names=common_features_eval,
                config=config 
            )
            print("✔︎ Evaluation completed.")
            metrics_output_file = config.get("metrics_file", "examples/results/evaluation_metrics.json") # Use a different name if needed
            os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)
            with open(metrics_output_file, "w") as f:
                json.dump(metrics, f, indent=4, cls=NpEncoder)
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
