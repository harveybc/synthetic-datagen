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
        periodicity_str_val: str # ADDED periodicity_str_val
    ) -> list[pd.Timestamp]:
        """
        Generates a list of 'num_synthetic_samples' datetime objects in chronological order,
        ending such that the next tick after the last synthetic datetime would lead into 'real_start_dt'.
        Skips weekends if periodicity_str_val is not 'daily'.
        """
        if num_synthetic_samples == 0:
            return []

        generated_datetimes_reversed = [] # Store in reverse chronological order first
        current_reference_dt = real_start_dt

        daily_target_time = None
        if periodicity_str_val == "daily":
            daily_target_time = current_reference_dt.time()

        count = 0
        max_iterations = num_synthetic_samples * 7 + 10 # Generous buffer for weekend skips
        iterations = 0

        while count < num_synthetic_samples and iterations < max_iterations:
            iterations += 1
            current_candidate_dt = current_reference_dt - time_delta_val
            is_weekend = current_candidate_dt.dayofweek >= 5 # Saturday (5) or Sunday (6)

            if periodicity_str_val == "daily":
                if not is_weekend:
                    current_candidate_dt = current_candidate_dt.replace(
                        hour=daily_target_time.hour,
                        minute=daily_target_time.minute,
                        second=daily_target_time.second,
                        microsecond=daily_target_time.microsecond
                    )
                    generated_datetimes_reversed.append(current_candidate_dt)
                    count += 1
            else: # For non-daily (e.g., hourly, minutely)
                if not is_weekend:
                    generated_datetimes_reversed.append(current_candidate_dt)
                    count += 1
            
            current_reference_dt = current_candidate_dt

        if count < num_synthetic_samples:
            print(f"WARN main.py: generate_synthetic_datetimes_before_real only generated {count}/{num_synthetic_samples} datetimes. Check data range and periodicity.")
        
        return list(reversed(generated_datetimes_reversed))

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
        if not x_train_file_path or not os.path.exists(x_train_file_path):            
            print(f"ERROR main.py: x_train_file '{x_train_file_path}' not found or not specified. Exiting.")
            sys.exit(1)
            
        max_steps_train_real = config["max_steps_train"] 
        n_samples_synthetic = config.get("n_samples", config.get("num_synthetic_samples_to_generate", 0))
        datetime_col_name = config.get("datetime_col_name", "DATE_TIME")
        dataset_periodicity = config.get("dataset_periodicity", "1h")
        decoder_input_window_size = generator_plugin.params.get("decoder_input_window_size")
        generator_full_feature_names_for_df_creation = generator_plugin.params.get("full_feature_names_ordered", [])
        if not generator_full_feature_names_for_df_creation or "DATE_TIME" not in generator_full_feature_names_for_df_creation:
            print(f"ERROR main.py: 'generator_full_feature_names_ordered' must be set in config and include 'DATE_TIME'. Current: {generator_full_feature_names_for_df_creation}")
            sys.exit(1)

        # --- Step A: Read the raw real data segment that will be appended and determine its start datetime ---
        df_real_raw_segment_for_output = pd.DataFrame(columns=TARGET_CSV_COLUMNS) 
        first_dt_of_real_segment_to_append: Optional[pd.Timestamp] = None

        if max_steps_train_real > 0:
            try:
                temp_df_real_raw = pd.read_csv(
                    x_train_file_path, 
                    nrows=max_steps_train_real, 
                    header=0 if config.get("feeder_real_data_file_has_header", True) else None
                )

                if not temp_df_real_raw.empty:
                    aligned_real_segment = pd.DataFrame(columns=TARGET_CSV_COLUMNS)
                    for col_target_name in TARGET_CSV_COLUMNS:
                        if col_target_name in temp_df_real_raw.columns:
                            aligned_real_segment[col_target_name] = temp_df_real_raw[col_target_name]
                        elif isinstance(col_target_name, int) and col_target_name < len(temp_df_real_raw.columns):
                             aligned_real_segment[TARGET_CSV_COLUMNS[col_target_name]] = temp_df_real_raw.iloc[:, col_target_name]
                        else:
                            aligned_real_segment[col_target_name] = np.nan 
                    df_real_raw_segment_for_output = aligned_real_segment

                    if datetime_col_name in df_real_raw_segment_for_output.columns and not df_real_raw_segment_for_output[datetime_col_name].empty:
                        first_dt_val = df_real_raw_segment_for_output[datetime_col_name].iloc[0]
                        if pd.notnull(first_dt_val):
                            first_dt_of_real_segment_to_append = pd.to_datetime(first_dt_val)
                        else:
                            print(f"WARN main.py: First datetime in raw real segment is null.")
                    else:
                        print(f"WARN main.py: Datetime column '{datetime_col_name}' not found or empty in the read raw real segment.")
                else:
                    print("DEBUG main.py: Raw real data segment read from x_train_file is empty (max_steps_train_real might be > file length or file empty).")
            except Exception as e_raw_read:
                print(f"WARN main.py: Error reading raw real data segment from '{x_train_file_path}': {e_raw_read}. Real segment will be empty.")
        else:
            print("DEBUG main.py: No real rows requested for output (max_steps_train_real is 0). Real segment will be empty.")
        
        # --- Step B: Generate target datetimes for synthetic data based on the real segment's start ---
        target_datetimes_for_generation = pd.Series([], dtype='datetime64[ns]')
        if n_samples_synthetic > 0:
            if first_dt_of_real_segment_to_append is not None:
                print(f"DEBUG main.py: First datetime of real segment to append (for synthetic generation reference): {first_dt_of_real_segment_to_append}")
                time_delta_map = {
                    "1h": timedelta(hours=1), "1H": timedelta(hours=1),
                    "15min": timedelta(minutes=15), "15T": timedelta(minutes=15), "15m": timedelta(minutes=15),
                    "1min": timedelta(minutes=1), "1T": timedelta(minutes=1), "1m": timedelta(minutes=1),
                    "daily": timedelta(days=1), "1D": timedelta(days=1)
                }
                generation_time_delta = time_delta_map.get(dataset_periodicity)
                if not generation_time_delta:
                    print(f"ERROR main.py: Invalid dataset_periodicity '{dataset_periodicity}' for prepending. Exiting.")
                    sys.exit(1)

                synthetic_target_datetimes_objs = generate_synthetic_datetimes_before_real(
                    real_start_dt=first_dt_of_real_segment_to_append,
                    num_synthetic_samples=n_samples_synthetic,
                    time_delta_val=generation_time_delta,
                    periodicity_str_val=dataset_periodicity
                )
                target_datetimes_for_generation = pd.Series(synthetic_target_datetimes_objs)
                if not target_datetimes_for_generation.empty:
                     print(f"DEBUG main.py: Generated {len(target_datetimes_for_generation)} synthetic datetimes for prepending. First: {target_datetimes_for_generation.iloc[0]}, Last: {target_datetimes_for_generation.iloc[-1]}")
                elif n_samples_synthetic > 0: # Should have been caught by empty synthetic_target_datetimes_objs
                     print(f"WARN main.py: No synthetic datetimes were generated despite valid real_start_dt. Check 'n_samples' and date logic.")
            else:
                print("ERROR main.py: Cannot determine first datetime of real segment for prepending, but synthetic samples are requested. Synthetic data generation will use fallback start time.")
                synthetic_start_dt_str = config.get("start_datetime") or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                target_datetimes_for_generation_str_list = generate_datetime_column( # Ensure this function is defined or imported
                    start_datetime_str=synthetic_start_dt_str,
                    num_samples=n_samples_synthetic,
                    periodicity_str=dataset_periodicity
                )
                target_datetimes_for_generation = pd.Series(pd.to_datetime(target_datetimes_for_generation_str_list))
        else:
            print("DEBUG main.py: No synthetic samples requested (n_samples_synthetic is 0).")


        # --- Step C: Preprocess full x_train_file for initial generator window ---
        # This part remains largely the same, as it's for the generator's initial state,
        # which needs processed features from the *end* of the training data.
        print(f"Preprocessing full '{x_train_file_path}' to extract initial window for generator...")
        config_for_full_train_preprocessing = config.copy()
        
        if hasattr(preprocessor_plugin, 'plugin_params'):
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
        # datetimes_train_processed_full_anytype can be None if preprocessor doesn't return it, handle gracefully for initial_datetimes_for_gen_window_series

        X_train_processed_full_np: np.ndarray
        if isinstance(X_train_processed_full_anytype, pd.DataFrame):
            if not processed_train_feature_names: 
                processed_train_feature_names = list(X_train_processed_full_anytype.columns)
            X_train_processed_full_np = X_train_processed_full_anytype.values.astype(np.float32)
        elif isinstance(X_train_processed_full_anytype, np.ndarray):
            X_train_processed_full_np = X_train_processed_full_anytype.astype(np.float32)
        else:
            raise TypeError(f"Preprocessor 'x_train' output is of unexpected type: {type(X_train_processed_full_anytype)}")

        if X_train_processed_full_np.ndim == 3:
            print(f"DEBUG main.py: X_train_processed_full_np is 3D {X_train_processed_full_np.shape}. Taking last element of sequence dimension.")
            X_train_processed_full_np = X_train_processed_full_np[:, -1, :]
            print(f"DEBUG main.py: X_train_processed_full_np reshaped to 2D: {X_train_processed_full_np.shape}")

        if not processed_train_feature_names and X_train_processed_full_np.ndim > 1 and X_train_processed_full_np.shape[0] > 0:
            num_feats = X_train_processed_full_np.shape[-1]
            processed_train_feature_names = [f"feature_{i}" for i in range(num_feats)]
            print(f"Warning: Using generic feature names for processed x_train data as 'feature_names' not in preprocessor output or derivable from DataFrame.")
        
        print(f"DEBUG main.py: Processed x_train data shape for initial window: {X_train_processed_full_np.shape}, Num features: {len(processed_train_feature_names)}, First few features: {processed_train_feature_names[:5]}")

        if X_train_processed_full_np.shape[0] < decoder_input_window_size:
            raise ValueError(f"Processed x_train data (length {X_train_processed_full_np.shape[0]}) is shorter than decoder window size ({decoder_input_window_size}). Cannot extract initial window.")

        df_temp_processed_train = pd.DataFrame(X_train_processed_full_np, columns=processed_train_feature_names)
        initial_window_raw_df_from_processed_train = df_temp_processed_train.iloc[-decoder_input_window_size:]
        
        initial_full_feature_window_for_gen_df = pd.DataFrame(
            np.nan, 
            index=range(decoder_input_window_size), 
            columns=generator_full_feature_names_for_df_creation
        ).astype(np.float32)

        for col_name_gen in generator_full_feature_names_for_df_creation:
            if col_name_gen in initial_window_raw_df_from_processed_train.columns:
                initial_full_feature_window_for_gen_df[col_name_gen] = initial_window_raw_df_from_processed_train[col_name_gen].values
        
        initial_full_feature_window_for_gen = initial_full_feature_window_for_gen_df.fillna(0.0).values 
        print(f"Prepared initial_full_feature_window for generator with shape: {initial_full_feature_window_for_gen.shape} from end of processed {x_train_file_path}")
        
        true_prev_close_for_initial_window_log_return: Optional[float] = None
        if 'CLOSE' in processed_train_feature_names:
            close_col_idx_in_processed = processed_train_feature_names.index('CLOSE')
            if X_train_processed_full_np.shape[0] > decoder_input_window_size:
                idx_before_window_starts = X_train_processed_full_np.shape[0] - decoder_input_window_size - 1
                true_prev_close_for_initial_window_log_return = float(X_train_processed_full_np[idx_before_window_starts, close_col_idx_in_processed])
                print(f"DEBUG main.py: Extracted true_prev_close_for_initial_window_log_return: {true_prev_close_for_initial_window_log_return}")
            else:
                print(f"DEBUG main.py: Not enough rows in X_train_processed_full_np ({X_train_processed_full_np.shape[0]}) to get a true_prev_close before the window of size {decoder_input_window_size}.")
        else:
            print("DEBUG main.py: 'CLOSE' not in processed_train_feature_names. Cannot extract true_prev_close_for_initial_window_log_return.")
        
        initial_datetimes_for_gen_window_series: Optional[pd.Series] = None
        if datetimes_train_processed_full_anytype is not None:
            datetimes_train_processed_full_series_for_window = pd.Series(pd.to_datetime(datetimes_train_processed_full_anytype))
            if initial_full_feature_window_for_gen is not None and initial_full_feature_window_for_gen.shape[0] > 0:
                num_rows_in_initial_window = initial_full_feature_window_for_gen.shape[0]
                if len(datetimes_train_processed_full_series_for_window) >= num_rows_in_initial_window:
                    initial_datetimes_for_gen_window_series = datetimes_train_processed_full_series_for_window.iloc[-num_rows_in_initial_window:].reset_index(drop=True)
                else:
                    print(f"WARN main.py: Length of processed datetimes ({len(datetimes_train_processed_full_series_for_window)}) is less than initial window size ({num_rows_in_initial_window}). Cannot set initial_datetimes_for_gen_window_series.")
        else:
            print("WARN main.py: datetimes_train_processed_full_anytype is None from preprocessor. Cannot set initial_datetimes_for_gen_window_series.")


        # --- Step D: Generate Synthetic Values ---
        X_syn_generated_values_np = np.array([]).reshape(0, len(generator_full_feature_names_for_df_creation)) 
        final_synthetic_target_datetimes_series = target_datetimes_for_generation # Use the correctly generated series

        if n_samples_synthetic > 0 and not final_synthetic_target_datetimes_series.empty:
            print(f"Generating feeder outputs for {n_samples_synthetic} synthetic steps...")
            feeder_outputs_sequence_synthetic = feeder_plugin.generate(
                n_ticks_to_generate=n_samples_synthetic,
                target_datetimes=final_synthetic_target_datetimes_series # This is already pd.Series of Timestamps
            )

            print("Generating synthetic feature values via GeneratorPlugin...")
            generated_output_from_plugin = generator_plugin.generate(
                feeder_outputs_sequence=feeder_outputs_sequence_synthetic,
                sequence_length_T=n_samples_synthetic,
                initial_full_feature_window=initial_full_feature_window_for_gen,
                initial_datetimes_for_window=initial_datetimes_for_gen_window_series,
                true_prev_close_for_initial_window_log_return=true_prev_close_for_initial_window_log_return
            )
            if isinstance(generated_output_from_plugin, list) and len(generated_output_from_plugin) == 1 and isinstance(generated_output_from_plugin[0], np.ndarray):
                 X_syn_generated_values_np = generated_output_from_plugin[0]
            elif isinstance(generated_output_from_plugin, np.ndarray) and generated_output_from_plugin.ndim == 3 and generated_output_from_plugin.shape[0] == 1 : 
                 X_syn_generated_values_np = generated_output_from_plugin[0]
            elif isinstance(generated_output_from_plugin, np.ndarray) and generated_output_from_plugin.ndim == 2 and generated_output_from_plugin.shape[0] == n_samples_synthetic :
                 X_syn_generated_values_np = generated_output_from_plugin
            else:
                raise TypeError(f"Unexpected output type or shape from generator_plugin.generate: {type(generated_output_from_plugin)}, shape if np.array: {getattr(generated_output_from_plugin, 'shape', 'N/A')}")
        elif n_samples_synthetic > 0: # but final_synthetic_target_datetimes_series is empty
            print("Skipping synthetic value generation as synthetic datetime series is empty (likely due to issues with real_start_dt).")


        # --- Step E: Create Synthetic DataFrame ---
        df_synthetic_generated_full_features = pd.DataFrame(X_syn_generated_values_np, columns=generator_full_feature_names_for_df_creation)
        if n_samples_synthetic > 0 and not final_synthetic_target_datetimes_series.empty:
            if len(final_synthetic_target_datetimes_series) == df_synthetic_generated_full_features.shape[0]:
                # Ensure datetime_col_name is a column before assigning
                if datetime_col_name not in df_synthetic_generated_full_features.columns:
                     df_synthetic_generated_full_features.insert(0, datetime_col_name, np.nan) # Add if missing
                df_synthetic_generated_full_features[datetime_col_name] = final_synthetic_target_datetimes_series.dt.strftime('%Y-%m-%d %H:%M:%S').values
            else:
                print(f"Warning: Mismatch between synthetic datetimes ({len(final_synthetic_target_datetimes_series)}) and generated values ({df_synthetic_generated_full_features.shape[0]}). Datetime column might be misaligned or omitted for synthetic data.")
        
        output_df_synthetic_aligned = pd.DataFrame(columns=TARGET_CSV_COLUMNS)
        if not df_synthetic_generated_full_features.empty:
            for col_target_name in TARGET_CSV_COLUMNS:
                if col_target_name in df_synthetic_generated_full_features.columns:
                    output_df_synthetic_aligned[col_target_name] = df_synthetic_generated_full_features[col_target_name]
                else:
                    output_df_synthetic_aligned[col_target_name] = np.nan

        # --- Step F: Real DataFrame for Output is df_real_raw_segment_for_output (already prepared and aligned) ---
        print(f"DEBUG main.py: Real data segment for final output DF shape: {df_real_raw_segment_for_output.shape}")


        # --- Step G: Combine and Save Final Output (Prepended) ---
        if output_df_synthetic_aligned.empty and df_real_raw_segment_for_output.empty:
            print("WARNING: Both synthetic and real data segments are empty. Output file will be empty or header-only.")
            df_combined_prepended = pd.DataFrame(columns=TARGET_CSV_COLUMNS) 
        elif output_df_synthetic_aligned.empty:
            df_combined_prepended = df_real_raw_segment_for_output
        elif df_real_raw_segment_for_output.empty:
            df_combined_prepended = output_df_synthetic_aligned
        else:
            # Ensure both dataframes have the DATE_TIME column correctly formatted before concat if necessary
            # df_synthetic_generated_full_features should have it from above.
            # df_real_raw_segment_for_output should have it from the raw CSV read.
            df_combined_prepended = pd.concat(
                [output_df_synthetic_aligned, 
                 df_real_raw_segment_for_output], 
                ignore_index=True
            )
        
        final_output_path = config.get("output_file")
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        df_combined_prepended.to_csv(final_output_path, index=False, na_rep='NaN')
        print(f"✔︎ Combined data (synthetic prepended to real) saved to {final_output_path} (Shape: {df_combined_prepended.shape})")

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
    from datetime import datetime, timedelta # Ensure this is imported at the top

    try:
        main()
    except KeyboardInterrupt:
        print("\nINFO: Execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e_global:
        print(f"❌ CRITICAL GLOBAL ERROR: An unhandled exception occurred outside main function execution: {e_global}")
        traceback.print_exc() # This will now work correctly
        sys.exit(1)
