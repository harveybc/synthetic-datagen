"""
GeneratorPlugin: Transforma vectores latentes en datos sintéticos por tick (primer elemento de la ventana).

Interfaz:
- plugin_params: configuración por defecto.
- plugin_debug_vars: variables expuestas en debug.
- __init__(config): carga y valida parámetros, carga el modelo decoder.
- set_params(**kwargs): actualiza parámetros.
- get_debug_info(): devuelve info de debug.
- add_debug_info(info): añade debug info a un diccionario.
- generate(Z, config): genera datos sintéticos unidimensionales por tick.

"""

import numpy as np # Make sure numpy is imported
import pandas as pd # Make sure pandas is imported
import tensorflow as tf # Ensure tensorflow is imported for tf.keras
from tensorflow.keras.models import load_model, Model
import pandas_ta as ta # Add pandas-ta
import os # For os.path.exists
import zipfile # For checking .keras file format (zip)
from tqdm.auto import tqdm # ADDED for the overall progress bar
import json # For loading normalization params
from typing import Dict, Any, List, Optional # ADD THIS LINE


class GeneratorPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "sequential_model_file": None,
        "decoder_input_window_size": 144,
        "full_feature_names_ordered": [],
        "decoder_output_feature_names": [],
        "ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
        "ti_feature_names": [],
        "date_conditional_feature_names": [],
        "feeder_conditional_feature_names": [],
        "ti_calculation_min_lookback": 200,
        "ti_params": {},
        "decoder_input_name_latent": "decoder_input_z_seq",
        "decoder_input_name_window": "input_x_window",
        "decoder_input_name_conditions": "decoder_input_conditions",
        "decoder_input_name_context": "decoder_input_h_context",
        "generator_normalization_params_file": None # Keep this for general normalization
        # "real_data_file_for_initial_close" is REMOVED from plugin_params
    }
    plugin_debug_vars = [
        "sequential_model_file", "decoder_input_window_size", "batch_size_inference",
        "full_feature_names_ordered", "decoder_output_feature_names",
        "ti_calculation_min_lookback"
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el GeneratorPlugin y carga el modelo generador secuencial.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        
        self.params = self.plugin_params.copy()
        self.sequential_model: Optional[Model] = None
        self.model: Optional[Model] = None # Alias for sequential_model
        self.normalization_params: Optional[Dict[str, Dict[str, float]]] = None
        self.initial_denormalized_close_anchor: Optional[float] = None
        self.previous_normalized_close: Optional[float] = None # ADDED for log_return calculation

        # Determine the file path for initial CLOSE anchor from the main config
        initial_close_file_path = config.get("x_train_file", config.get("real_data_file"))

        self.set_params(**config) # This will call _load_model_from_path

        model_path = self.params.get("sequential_model_file")
        if not model_path:
             raise ValueError("El parámetro 'sequential_model_file' debe especificar la ruta al modelo generador secuencial y no puede estar vacío después de la configuración.")
        if self.sequential_model is None: # Check after set_params has run _load_model_from_path
            raise IOError(f"Modelo secuencial no se pudo cargar desde {model_path} durante la inicialización, o la ruta no se proporcionó correctamente.")

        if not self.params.get("full_feature_names_ordered"):
            raise ValueError("El parámetro 'full_feature_names_ordered' es obligatorio.")
        if not self.params.get("decoder_output_feature_names"):
            raise ValueError("El parámetro 'decoder_output_feature_names' es obligatorio.")
        
        self.feature_to_idx = {name: i for i, name in enumerate(self.params["full_feature_names_ordered"])}
        self.num_all_features = len(self.params["full_feature_names_ordered"])
        self._validate_feature_name_consistency()
        
        # Load normalization params if path is provided in config
        norm_params_file = self.params.get("generator_normalization_params_file") # Use the specific key
        if self.normalization_params is None and norm_params_file:
            self.normalization_params = self._load_normalization_params(norm_params_file)
        
        # Load initial CLOSE anchor using the determined path
        if self.initial_denormalized_close_anchor is None:
            self._load_initial_close_anchor(initial_close_file_path)


    def _load_model_from_path(self, model_path: str):
        """
        Carga el modelo Keras desde la ruta especificada y actualiza self.sequential_model y self.model.
        Intenta usar Keras 3 set_safe_mode y Keras 2 enable/disable_unsafe_deserialization.
        """
        if not model_path:
            # This case should ideally be handled before calling, or raise an error if a path is always expected.
            # For now, if path is empty, we assume no model is to be loaded or it's an error state.
            self.sequential_model = None
            self.model = None
            print("GeneratorPlugin: Advertencia - Se intentó cargar el modelo con una ruta vacía.")
            # Depending on requirements, you might want to raise ValueError here.
            # For now, to match previous logic where set_params could be called to clear model,
            # we allow setting to None if path is None.
            # However, __init__ will fail if sequential_model is None after set_params.
            return

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GeneratorPlugin: Archivo de modelo Keras no encontrado en la ruta: {model_path}")

        print(f"GeneratorPlugin: Intentando cargar modelo Keras desde: {model_path}")

        # Verificar si el archivo es un ZIP válido (formato .keras)
        if model_path.endswith(".keras"):
            try:
                with zipfile.ZipFile(model_path, 'r') as zf:
                    # Podrías añadir más verificaciones aquí, ej. zf.testzip() o buscar archivos específicos
                    pass # Simple check: if it opens, it's likely a zip
                print(f"GeneratorPlugin: El archivo '{model_path}' parece ser un archivo ZIP válido (esperado para formato .keras).")
            except zipfile.BadZipFile:
                print(f"GeneratorPlugin: ERROR CRÍTICO - El archivo '{model_path}' NO es un archivo ZIP válido. "
                      "Esto es inesperado para un archivo .keras y probablemente causará un fallo en la carga.")
                # No levantar error aquí todavía, dejar que load_model intente y falle con su propio error.
            except Exception as e_zip:
                print(f"GeneratorPlugin: Advertencia - Error al verificar el formato ZIP para '{model_path}': {e_zip}")
        
        original_safe_mode_setting = None # For Keras 3 style
        deserialization_method_used = None # To track which method was used

        try:
            # Attempt Keras 3 style safe_mode setting
            if hasattr(tf.keras.config, 'set_safe_mode') and hasattr(tf.keras.config, 'safe_mode'):
                print("GeneratorPlugin: Attempting to use Keras 3 style safe_mode for deserialization.")
                original_safe_mode_setting = tf.keras.config.safe_mode()
                tf.keras.config.set_safe_mode(False)
                deserialization_method_used = "keras3_safe_mode"
                print(f"GeneratorPlugin: Keras 3 safe_mode set to False. Original was: {original_safe_mode_setting}")
            # Fallback to Keras 2 style if Keras 3 methods are not present
            elif hasattr(tf.keras.config, 'enable_unsafe_deserialization'):
                print("GeneratorPlugin: Attempting to use Keras 2 style enable_unsafe_deserialization.")
                tf.keras.config.enable_unsafe_deserialization()
                deserialization_method_used = "keras2_unsafe_deserialization"
                print("GeneratorPlugin: Keras 2 unsafe deserialization enabled.")
            else:
                print("GeneratorPlugin: Warning - Could not find a known method to control Keras unsafe deserialization.")
                # Proceed without changing settings, load_model might fail if unsafe operations are needed.

            loaded_model: Model = load_model(model_path, compile=False)
            
            # Revert deserialization settings
            if deserialization_method_used == "keras3_safe_mode":
                if original_safe_mode_setting is not None:
                    tf.keras.config.set_safe_mode(original_safe_mode_setting)
                    print(f"GeneratorPlugin: Keras 3 safe_mode restored to: {original_safe_mode_setting}")
            elif deserialization_method_used == "keras2_unsafe_deserialization":
                if hasattr(tf.keras.config, 'disable_unsafe_deserialization'):
                    tf.keras.config.disable_unsafe_deserialization()
                    print("GeneratorPlugin: Keras 2 unsafe deserialization disabled.")
                else:
                    # This case should ideally not be hit if enable_unsafe_deserialization existed
                    print("GeneratorPlugin: Warning - Keras 2 disable_unsafe_deserialization method not found after enabling.")


            self.sequential_model = loaded_model
            self.model = loaded_model # Mantener el alias
            print(f"GeneratorPlugin: Modelo Keras cargado exitosamente desde {model_path}")
            # print(self.sequential_model.summary()) # Descomentar para depurar la estructura del modelo
        except Exception as e:
            # Attempt to revert deserialization settings even in case of an error
            try:
                if deserialization_method_used == "keras3_safe_mode":
                    if original_safe_mode_setting is not None:
                        tf.keras.config.set_safe_mode(original_safe_mode_setting)
                        print(f"GeneratorPlugin: Keras 3 safe_mode restored to {original_safe_mode_setting} during error handling.")
                elif deserialization_method_used == "keras2_unsafe_deserialization":
                    if hasattr(tf.keras.config, 'disable_unsafe_deserialization'):
                        tf.keras.config.disable_unsafe_deserialization()
                        print("GeneratorPlugin: Keras 2 unsafe deserialization disabled during error handling.")
            except Exception as e_config_restore:
                print(f"GeneratorPlugin: Warning - Failed to revert deserialization settings during error handling: {e_config_restore}")

            error_message = f"Error al cargar el modelo Keras desde '{model_path}'. Tipo de error: {type(e).__name__}, Mensaje: {e}"
            print(f"GeneratorPlugin: {error_message}")
            if "HDF5" in str(e) and model_path.endswith(".keras"):
                print("GeneratorPlugin: DETALLE CRÍTICO - Se encontró un error HDF5 al cargar un archivo .keras. "
                      "Esto sugiere un problema de versión de Keras/TensorFlow, un archivo corrupto, "
                      "o que el archivo fue guardado como .h5 pero renombrado a .keras.")
            self.sequential_model = None # Asegurar que el modelo es None si la carga falla
            self.model = None
            raise IOError(error_message) # Re-lanzar como IOError para consistencia


    def set_params(self, **kwargs):
        """
        Actualiza parámetros del plugin.
        """
        print(f"GeneratorPlugin.set_params called with kwargs: {list(kwargs.keys())}")
        
        old_model_file = self.params.get("sequential_model_file")
        old_norm_file = self.params.get("generator_normalization_params_file")
        # old_initial_close_file is no longer a plugin-specific param
        old_full_feature_names = self.params.get("full_feature_names_ordered")

        # Determine the file path for initial CLOSE anchor from the main config (kwargs)
        # or fall back to existing main config values if not in kwargs.
        # This path is NOT stored in self.params directly.
        current_config_for_initial_close = kwargs # Prioritize kwargs
        initial_close_file_path_candidate = current_config_for_initial_close.get(
            "x_train_file", current_config_for_initial_close.get("real_data_file")
        )

        for param_key_short in self.plugin_params.keys():
            prefixed_key = f"generator_{param_key_short}"
            
            if prefixed_key in kwargs:
                self.params[param_key_short] = kwargs[prefixed_key]
            elif param_key_short in kwargs: # Check non-prefixed as well
                self.params[param_key_short] = kwargs[param_key_short]
        
        # Explicitly handle generator_normalization_params_file if it's passed with prefix or not
        if "generator_normalization_params_file" in kwargs:
            self.params["generator_normalization_params_file"] = kwargs["generator_normalization_params_file"]
        elif "normalization_params_file" in kwargs and "generator_normalization_params_file" not in self.plugin_params:
             # If a generic "normalization_params_file" is passed and the specific one isn't a plugin param
             pass # self.params["generator_normalization_params_file"] = kwargs["normalization_params_file"]


        new_model_file = self.params.get("sequential_model_file")
        new_norm_file = self.params.get("generator_normalization_params_file")
        # new_initial_close_file is now initial_close_file_path_candidate

        if new_model_file != old_model_file or (new_model_file and self.sequential_model is None):
            self._load_model_from_path(new_model_file)
        elif not new_model_file and old_model_file:
            print("GeneratorPlugin: La ruta del modelo se ha borrado. Limpiando el modelo cargado.")
            self.sequential_model = None
            self.model = None

        if new_norm_file != old_norm_file or (new_norm_file and self.normalization_params is None):
            self.normalization_params = self._load_normalization_params(new_norm_file)
        elif not new_norm_file and old_norm_file:
            print("GeneratorPlugin: Normalization params file path cleared. Resetting normalization_params.")
            self.normalization_params = None
            
        # Reload initial close anchor if the source file path from main config might have changed
        # or if it hasn't been loaded yet.
        # We don't store the path in self.params, so we check if the candidate path is valid
        # and if the anchor is None.
        if initial_close_file_path_candidate and self.initial_denormalized_close_anchor is None:
            self._load_initial_close_anchor(initial_close_file_path_candidate)
        elif not initial_close_file_path_candidate and self.initial_denormalized_close_anchor is not None:
            # This case is tricky: if the path in main config becomes None, should we reset?
            # For now, if it was loaded, keep it unless explicitly told to reload with a new path.
            # If the intent is to clear it if the path is removed from main config,
            # we'd need to track the path used to load it.
            # Simpler: it's loaded once at init or if path changes and anchor is None.
            pass


        if self.params.get("full_feature_names_ordered") != old_full_feature_names or \
           any(key in kwargs for key in ["decoder_output_feature_names", "ohlc_feature_names", 
                                         "ti_feature_names", "date_conditional_feature_names", 
                                         "feeder_conditional_feature_names",
                                         "generator_decoder_output_feature_names", 
                                         "generator_ohlc_feature_names",
                                         "generator_ti_feature_names",
                                         "generator_date_conditional_feature_names",
                                         "generator_feeder_conditional_feature_names"
                                         ]):
            if self.params.get("full_feature_names_ordered"):
                self.feature_to_idx = {name: i for i, name in enumerate(self.params["full_feature_names_ordered"])}
                self.num_all_features = len(self.params["full_feature_names_ordered"])
                self._validate_feature_name_consistency()
            elif old_full_feature_names: 
                 raise ValueError("'full_feature_names_ordered' cannot be empty after update.")
        

    def _load_initial_close_anchor(self, file_path: Optional[str]):
        """
        Loads the last CLOSE price from the specified data file (e.g., x_train_file).
        The file_path comes from the main application config.
        """
        if not file_path:
            print("GeneratorPlugin: Warning - No data file path (e.g., 'x_train_file') provided in main config for initial CLOSE. Initial CLOSE anchor will default to 1.0.")
            self.initial_denormalized_close_anchor = 1.0
            return

        try:
            df_real = pd.read_csv(file_path)
            if 'CLOSE' in df_real.columns and not df_real['CLOSE'].empty:
                # IMPORTANT ASSUMPTION: The 'CLOSE' in this file might be NORMALIZED if it's 'normalized_d4.csv'.
                # If it's normalized, we need to denormalize it here.
                # For now, let's assume we need to check normalization_params.
                last_close_val_from_file = float(df_real['CLOSE'].iloc[-1])

                if self.normalization_params and "CLOSE" in self.normalization_params:
                    # If normalization params exist for CLOSE, assume value in file is normalized
                    self.initial_denormalized_close_anchor = self._denormalize_value(last_close_val_from_file, "CLOSE")
                    print(f"GeneratorPlugin: Initial CLOSE anchor (denormalized from file) loaded from '{file_path}': {self.initial_denormalized_close_anchor}")
                else:
                    # If no normalization params for CLOSE, assume value in file is already denormalized
                    self.initial_denormalized_close_anchor = last_close_val_from_file
                    print(f"GeneratorPlugin: Initial CLOSE anchor (assumed denormalized) loaded from '{file_path}': {self.initial_denormalized_close_anchor}")
            else:
                print(f"GeneratorPlugin: Warning - 'CLOSE' column not found or empty in '{file_path}'. Initial CLOSE anchor defaulting to 1.0.")
                self.initial_denormalized_close_anchor = 1.0
        except FileNotFoundError:
            print(f"GeneratorPlugin: ERROR - Data file for initial CLOSE not found: {file_path}. Defaulting to 1.0.")
            self.initial_denormalized_close_anchor = 1.0
        except Exception as e:
            print(f"GeneratorPlugin: ERROR - Could not load initial CLOSE from '{file_path}': {e}. Defaulting to 1.0.")
            self.initial_denormalized_close_anchor = 1.0
        
        if self.initial_denormalized_close_anchor is None or pd.isna(self.initial_denormalized_close_anchor): # Final fallback
            self.initial_denormalized_close_anchor = 1.0
            print("GeneratorPlugin: Critical - initial_denormalized_close_anchor was None/NaN after attempting load. Defaulted to 1.0.")


    def _validate_feature_name_consistency(self):
        """
        Validates that all configured feature name lists are subsets of 
        'full_feature_names_ordered' and that critical feature lists are not empty.
        """
        print("GeneratorPlugin: Validating feature name consistency...")
        full_set = set(self.params.get("full_feature_names_ordered", []))
        if not full_set:
            raise ValueError("'full_feature_names_ordered' cannot be empty and must be configured.")

        def check_subset(list_name, critical=False):
            feature_list = self.params.get(list_name, [])
            if critical and not feature_list:
                raise ValueError(f"'{list_name}' is a critical parameter and cannot be empty.")
            
            current_set = set(feature_list)
            if not current_set.issubset(full_set):
                missing = current_set - full_set
                raise ValueError(
                    f"Features in '{list_name}' are not all present in 'full_feature_names_ordered'. "
                    f"Missing: {missing}. Ensure '{list_name}' only contains features from "
                    f"'full_feature_names_ordered'."
                )
            print(f"GeneratorPlugin: Feature list '{list_name}' validated successfully against 'full_feature_names_ordered'.")

        check_subset("decoder_output_feature_names", critical=True)
        check_subset("ohlc_feature_names", critical=True)
        check_subset("ti_feature_names", critical=False) # Can be empty if no TIs are calculated
        
        # For conditional features, we also need to check their sin/cos transformed versions if applicable
        # date_conditional_feature_names are the original names (e.g., "day_of_month")
        # Their transformed versions (e.g., "day_of_month_sin") must be in full_feature_names_ordered
        date_cond_original_names = self.params.get("date_conditional_feature_names", [])
        transformed_date_cond_names = []
        for name in date_cond_original_names:
            transformed_date_cond_names.append(f"{name}_sin")
            transformed_date_cond_names.append(f"{name}_cos")
        
        date_cond_set = set(transformed_date_cond_names)
        if transformed_date_cond_names and not date_cond_set.issubset(full_set):
            missing_transformed = date_cond_set - full_set
            raise ValueError(
                f"Transformed date conditional features (from 'date_conditional_feature_names') are not all present "
                f"in 'full_feature_names_ordered'. Missing: {missing_transformed}. "
                f"Ensure sin/cos versions of date features are in 'full_feature_names_ordered'."
            )
        if transformed_date_cond_names:
             print(f"GeneratorPlugin: Transformed date conditional features validated successfully.")
        
        check_subset("feeder_conditional_feature_names", critical=False) # Can be empty

        # Validate that decoder input names are set
        for input_name_key in ["decoder_input_name_latent", "decoder_input_name_window", 
                               "decoder_input_name_conditions", "decoder_input_name_context"]:
            if not self.params.get(input_name_key):
                raise ValueError(f"Decoder input name parameter '{input_name_key}' is not configured in GeneratorPlugin.params.")
        print("GeneratorPlugin: All decoder input name parameters are configured.")
        print("GeneratorPlugin: Feature name consistency validation complete.")

    # Placeholder for _normalize_value and _denormalize_value
    # You need to implement these based on your normalization strategy
    # For example, using a loaded JSON file with min/max values per feature.

    def _load_normalization_params(self, file_path: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
        if not file_path:
            print("GeneratorPlugin: Warning - No normalization_params_file provided. Denormalization/normalization will be identity operations.")
            return None
        try:
            import json
            with open(file_path, 'r') as f:
                params = json.load(f)
            # Basic validation: check if it's a dict and contains 'min'/'max' for entries
            if not isinstance(params, dict):
                raise ValueError("Normalization params file should contain a JSON object (dictionary).")
            for feature, values in params.items():
                if not (isinstance(values, dict) and "min" in values and "max" in values):
                    raise ValueError(f"Normalization params for feature '{feature}' must be a dict with 'min' and 'max' keys.")
            print(f"GeneratorPlugin: Successfully loaded normalization parameters from {file_path}")
            return params
        except FileNotFoundError:
            print(f"GeneratorPlugin: ERROR - Normalization parameters file not found: {file_path}. Denormalization/normalization will be identity operations.")
            return None
        except json.JSONDecodeError:
            print(f"GeneratorPlugin: ERROR - Could not decode JSON from normalization parameters file: {file_path}. Denormalization/normalization will be identity operations.")
            return None
        except ValueError as ve:
            print(f"GeneratorPlugin: ERROR - Invalid format in normalization parameters file '{file_path}': {ve}. Denormalization/normalization will be identity operations.")
            return None


    def _normalize_value(self, value: float, feature_name: str) -> float:
        if self.normalization_params and feature_name in self.normalization_params:
            params = self.normalization_params[feature_name]
            min_val, max_val = params['min'], params['max']
            if max_val == min_val: # Avoid division by zero if min and max are the same
                return 0.0 if value == min_val else (value - min_val) # Or handle as per your logic, e.g., 0.5 or raise error
            return (value - min_val) / (max_val - min_val)
        # print(f"GeneratorPlugin: Warning - No normalization params for '{feature_name}' or params not loaded. Returning original value for normalization.")
        return value # Identity if no params

    def _denormalize_value(self, norm_value: float, feature_name: str) -> float:
        if self.normalization_params and feature_name in self.normalization_params:
            params = self.normalization_params[feature_name]
            min_val, max_val = params['min'], params['max']
            return norm_value * (max_val - min_val) + min_val
        # print(f"GeneratorPlugin: Warning - No normalization params for '{feature_name}' or params not loaded. Returning original value for denormalization.")
        return norm_value # Identity if no params


    def get_debug_info(self) -> Dict[str, Any]:
        """
        Devuelve diccionario con valores de debug.

        :return: {var: valor} para cada var en plugin_debug_vars.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        """
        Inserta info de debug en el diccionario proporcionado.

        :param debug_info: diccionario destino.
        """
        debug_info.update(self.get_debug_info())

    def _calculate_technical_indicators(self, ohlc_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators based on the provided OHLC history.
        Returns a DataFrame containing only the calculated TI columns for the last row.
        Assumes ohlc_history_df has columns named as in self.params["ohlc_feature_names"] (e.g., OPEN, HIGH, LOW, CLOSE).
        """
        if ohlc_history_df.empty or len(ohlc_history_df) < 2:
            nan_df = pd.DataFrame(columns=self.params["ti_feature_names"], index=[0]).astype(np.float32)
            nan_df = nan_df.fillna(np.nan)
            return nan_df

        ohlc_map = {
            self.params["ohlc_feature_names"][0]: 'open',
            self.params["ohlc_feature_names"][1]: 'high',
            self.params["ohlc_feature_names"][2]: 'low',
            self.params["ohlc_feature_names"][3]: 'close'
        }
        df = ohlc_history_df.rename(columns=ohlc_map)

        ti_df = pd.DataFrame(index=df.index)
        p = self.params["ti_params"]
        ti_names_to_calculate = self.params["ti_feature_names"]

        # Calculate TIs using pandas_ta, checking if they are requested
        if 'RSI' in ti_names_to_calculate:
            ti_df['RSI'] = ta.rsi(df['close'], length=p.get("rsi_length", 14))
        if any(x in ti_names_to_calculate for x in ['MACD', 'MACD_Histogram', 'MACD_Signal']):
            macd_output = ta.macd(df['close'], fast=p.get("macd_fast", 12), slow=p.get("macd_slow", 26), signal=p.get("macd_signal", 9))
            if macd_output is not None and not macd_output.empty:
                if 'MACD' in ti_names_to_calculate: ti_df['MACD'] = macd_output.iloc[:,0]
                if 'MACD_Histogram' in ti_names_to_calculate: ti_df['MACD_Histogram'] = macd_output.iloc[:,1]
                if 'MACD_Signal' in ti_names_to_calculate: ti_df['MACD_Signal'] = macd_output.iloc[:,2]
        if 'EMA' in ti_names_to_calculate:
            ti_df['EMA'] = ta.ema(df['close'], length=p.get("ema_length", 14))
        if any(x in ti_names_to_calculate for x in ['Stochastic_%K', 'Stochastic_%D']):
            stoch_output = ta.stoch(df['high'], df['low'], df['close'], k=p.get("stoch_k", 14), d=p.get("stoch_d", 3), smooth_k=p.get("stoch_smooth_k", 3))
            if stoch_output is not None and not stoch_output.empty:
                if 'Stochastic_%K' in ti_names_to_calculate: ti_df['Stochastic_%K'] = stoch_output.iloc[:,0]
                if 'Stochastic_%D' in ti_names_to_calculate: ti_df['Stochastic_%D'] = stoch_output.iloc[:,1]
        if any(x in ti_names_to_calculate for x in ['ADX', 'DI+', 'DI-']):
            adx_output = ta.adx(df['high'], df['low'], df['close'], length=p.get("adx_length", 14))
            if adx_output is not None and not adx_output.empty:
                if 'ADX' in ti_names_to_calculate: ti_df['ADX'] = adx_output.iloc[:,0]
                if 'DI+' in ti_names_to_calculate: ti_df['DI+'] = adx_output.iloc[:,1]
                if 'DI-' in ti_names_to_calculate: ti_df['DI-'] = adx_output.iloc[:,2]
        if 'ATR' in ti_names_to_calculate:
            ti_df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=p.get("atr_length", 14))
        if 'CCI' in ti_names_to_calculate:
            ti_df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=p.get("cci_length", 14))
        if 'WilliamsR' in ti_names_to_calculate:
            ti_df['WilliamsR'] = ta.willr(df['high'], df['low'], df['close'], length=p.get("willr_length", 14))
        if 'Momentum' in ti_names_to_calculate:
            ti_df['Momentum'] = ta.mom(df['close'], length=p.get("mom_length", 14))
        if 'ROC' in ti_names_to_calculate:
            ti_df['ROC'] = ta.roc(df['close'], length=p.get("roc_length", 14))

        for ti_name in self.params["ti_feature_names"]:
            if ti_name not in ti_df.columns:
                ti_df[ti_name] = np.nan
        
        return ti_df[self.params["ti_feature_names"]].iloc[[-1]].reset_index(drop=True).astype(np.float32)

    def generate(self,
                 feeder_outputs_sequence: List[Dict[str, np.ndarray]],
                 sequence_length_T: int,
                 initial_full_feature_window: Optional[np.ndarray] = None
                ) -> np.ndarray:
        """
        Genera una secuencia de variables base de manera autorregresiva usando el modelo secuencial cargado.
        """
        decoder_input_window_size = self.params["decoder_input_window_size"]
        min_ohlc_hist_len = self.params["ti_calculation_min_lookback"]
        
        ohlc_names = self.params["ohlc_feature_names"] # Should be ["OPEN", "HIGH", "LOW", "CLOSE"]

        # Initialize self.previous_normalized_close for this generation sequence
        self.previous_normalized_close = None # Reset for each call to generate
        if initial_full_feature_window is not None and 'CLOSE' in self.feature_to_idx:
            last_norm_close_in_window = initial_full_feature_window[-1, self.feature_to_idx['CLOSE']]
            if pd.notnull(last_norm_close_in_window):
                self.previous_normalized_close = float(last_norm_close_in_window)
                print(f"GeneratorPlugin: Initialized previous_normalized_close from initial_full_feature_window: {self.previous_normalized_close}")

        ohlc_history_for_ti_list = [] 

        current_input_feature_window = np.zeros((decoder_input_window_size, self.num_all_features), dtype=np.float32)
        if initial_full_feature_window is not None:
            if initial_full_feature_window.shape == current_input_feature_window.shape:
                current_input_feature_window = initial_full_feature_window.astype(np.float32).copy()
                start_idx_for_ohlc_hist = max(0, decoder_input_window_size - min_ohlc_hist_len)
                for i in range(start_idx_for_ohlc_hist, decoder_input_window_size):
                    # OHLC in initial_full_feature_window are assumed to be NORMALIZED
                    row_ohlc_norm_values = {
                        name: current_input_feature_window[i, self.feature_to_idx[name]]
                        for name in self.params["ohlc_feature_names"] if name in self.feature_to_idx
                    }
                    # Denormalize here for TI history
                    ohlc_dict_denorm = {
                        name: self._denormalize_value(row_ohlc_norm_values.get(name, np.nan), name)
                        for name in self.params["ohlc_feature_names"]
                    }
                    # Only add if all OHLC values are valid after denormalization
                    if all(pd.notnull(v) for v in ohlc_dict_denorm.values()) and len(ohlc_dict_denorm) == len(self.params["ohlc_feature_names"]):
                        ohlc_history_for_ti_list.append(ohlc_dict_denorm)

            else:
                raise ValueError(f"Shape mismatch for initial_full_feature_window. Expected {current_input_feature_window.shape}, got {initial_full_feature_window.shape}")
        else:
            print("GeneratorPlugin Warning: No initial_full_feature_window provided. Using zeros.")

        generated_sequence_all_features_list = []

        for t in tqdm(range(sequence_length_T), desc="Generating synthetic sequence", unit="step", dynamic_ncols=True):
            current_tick_assembled_features = np.full(self.num_all_features, np.nan, dtype=np.float32)

            feeder_step_output = feeder_outputs_sequence[t]
            zt_original = feeder_step_output["Z"] 
            
            if zt_original.ndim == 2:
                zt = np.expand_dims(zt_original, axis=0)
            elif zt_original.ndim == 3 and zt_original.shape[0] == 1: 
                zt = zt_original
            else:
                raise ValueError(f"Unexpected shape for zt from Feeder: {zt_original.shape}.")

            conditional_data_t = feeder_step_output["conditional_data"] 
            if conditional_data_t.ndim == 1: conditional_data_t = np.expand_dims(conditional_data_t, axis=0)

            context_h_t = feeder_step_output.get("context_h", np.zeros((1,1))) 
            if context_h_t.ndim == 1: context_h_t = np.expand_dims(context_h_t, axis=0)

            decoder_inputs = {
                self.params["decoder_input_name_latent"]: zt,
                self.params["decoder_input_name_conditions"]: conditional_data_t,
                self.params["decoder_input_name_context"]: context_h_t
            }
            
            generated_decoder_output_step_t = self.sequential_model.predict(decoder_inputs, verbose=0)
            
            if generated_decoder_output_step_t.ndim == 3 and generated_decoder_output_step_t.shape[1] == 1:
                decoded_features_for_current_tick = generated_decoder_output_step_t[0, 0, :]
            elif generated_decoder_output_step_t.ndim == 2 and generated_decoder_output_step_t.shape[0] == 1:
                decoded_features_for_current_tick = generated_decoder_output_step_t[0, :]
            else:
                raise ValueError(f"Unexpected decoder output shape: {generated_decoder_output_step_t.shape}.")

            # 1. Fill features from decoder output
            # This now includes OPEN, HIGH, LOW, and all high-frequency ticks.
            # CLOSE and log_return are NOT expected from the decoder with the current model.
            for i, name in enumerate(self.params["decoder_output_feature_names"]): # Length is now 23
                if name in self.feature_to_idx:
                    current_tick_assembled_features[self.feature_to_idx[name]] = decoded_features_for_current_tick[i] # i will go up to 22
            
            norm_open = current_tick_assembled_features[self.feature_to_idx[ohlc_names[0]]] if ohlc_names[0] in self.feature_to_idx else np.nan
            norm_high = current_tick_assembled_features[self.feature_to_idx[ohlc_names[1]]] if ohlc_names[1] in self.feature_to_idx else np.nan
            norm_low = current_tick_assembled_features[self.feature_to_idx[ohlc_names[2]]] if ohlc_names[2] in self.feature_to_idx else np.nan
            
            # log_return is NOT from decoder, so normalized_log_return_from_decoder will be effectively unused.
            # normalized_log_return_from_decoder = np.nan # Explicitly set, or rely on it not being filled.
            # if "log_return" in self.feature_to_idx and "log_return" in self.params["decoder_output_feature_names"]: # This condition will now be false
            #    normalized_log_return_from_decoder = current_tick_assembled_features[self.feature_to_idx["log_return"]]

            # REMOVED STEP 1.A: Derive CLOSE price using log_return from decoder, as log_return is not from decoder.

            # NEW STRATEGY FOR norm_close:
            norm_close = np.nan
            if "CLOSE" in self.feature_to_idx:
                if t == 0 and self.previous_normalized_close is None and self.initial_denormalized_close_anchor is not None:
                    # Use anchor for the very first CLOSE if no history window was provided
                    norm_close = self._normalize_value(self.initial_denormalized_close_anchor, "CLOSE")
                    print(f"GeneratorPlugin: Step {t}, using initial_denormalized_close_anchor for CLOSE: {self.initial_denormalized_close_anchor} -> {norm_close}")
                elif pd.notnull(norm_open):
                    norm_close = norm_open # Placeholder: CLOSE = OPEN
                    # print(f"GeneratorPlugin: Step {t}, setting norm_close = norm_open = {norm_close}")
                else:
                    # If norm_open is also NaN, norm_close remains NaN, to be filled by Step 8
                    print(f"GeneratorPlugin: Step {t}, norm_open is NaN. norm_close will be NaN before fallback.")
                
                if pd.notnull(norm_close):
                    current_tick_assembled_features[self.feature_to_idx["CLOSE"]] = norm_close
            else:
                norm_close = np.nan # CLOSE feature not in full list

            # 2. Fill conditional features (sin/cos dates, fundamentals)
            cond_input_idx = 0
            for original_date_feat_name in self.params["date_conditional_feature_names"]:
                for suffix in ["_sin", "_cos"]:
                    feat_name = f"{original_date_feat_name}{suffix}"
                    if feat_name in self.feature_to_idx:
                        current_tick_assembled_features[self.feature_to_idx[feat_name]] = conditional_data_t[0, cond_input_idx]
                    cond_input_idx += 1
            
            for name in self.params["feeder_conditional_feature_names"]:
                if name in self.feature_to_idx:
                    current_tick_assembled_features[self.feature_to_idx[name]] = conditional_data_t[0, cond_input_idx]
                cond_input_idx += 1

            # 3. Fill raw date features (normalized)
            dt_obj = feeder_step_output["datetimes"] # pd.Timestamp object
            raw_date_map = {
                "day_of_month": dt_obj.day,
                "hour_of_day": dt_obj.hour,
                "day_of_week": dt_obj.dayofweek # Monday=0 to Sunday=6
                # "day_of_year": dt_obj.dayofyear # If needed
            }
            for raw_feat_name, raw_val in raw_date_map.items():
                if raw_feat_name in self.feature_to_idx:
                    # These must be in normalization_params to be correctly normalized
                    current_tick_assembled_features[self.feature_to_idx[raw_feat_name]] = self._normalize_value(float(raw_val), raw_feat_name)

            # 4. Calculate and fill Technical Indicators
            # This step now uses the derived norm_close
            current_ohlc_for_ti_calc_normalized = {
                ohlc_names[0]: norm_open,
                ohlc_names[1]: norm_high,
                ohlc_names[2]: norm_low,
                ohlc_names[3]: norm_close # Use the derived norm_close from new strategy
            }
            
            current_ohlc_values_for_ti_dict = {
                name: self._denormalize_value(current_ohlc_for_ti_calc_normalized.get(name, np.nan), name)
                for name in self.params["ohlc_feature_names"]
            }
            
            if all(pd.notnull(v) for v in current_ohlc_values_for_ti_dict.values()) and \
               len(current_ohlc_values_for_ti_dict) == len(self.params["ohlc_feature_names"]):
                ohlc_history_for_ti_list.append(current_ohlc_values_for_ti_dict)
                if len(ohlc_history_for_ti_list) >= 1: 
                    ohlc_df_for_ti_calc = pd.DataFrame(ohlc_history_for_ti_list)
                    calculated_tis_series_denormalized = self._calculate_technical_indicators(ohlc_df_for_ti_calc).iloc[0]
                    
                    for ti_name in self.params["ti_feature_names"]:
                        if ti_name in self.feature_to_idx:
                            denormalized_ti_val = calculated_tis_series_denormalized.get(ti_name, np.nan)
                            if pd.notnull(denormalized_ti_val):
                                normalized_ti_val = self._normalize_value(denormalized_ti_val, ti_name)
                                current_tick_assembled_features[self.feature_to_idx[ti_name]] = normalized_ti_val
                            # else: TI is NaN, will be handled by Step 8 if not filled by carry-forward logic below
                # else: Not enough history for TIs, will be handled by Step 8
            # else: Not all OHLC values were available (e.g., derived CLOSE was NaN), TIs handled by Step 8

            # 5. Calculate and fill derived OHLC features (e.g., BC-BO)
            # This step now uses the derived norm_close
            if not np.isnan(norm_close) and not np.isnan(norm_open):
                if "BC-BO" in self.feature_to_idx: 
                    bc_bo_val = norm_close - norm_open
                    current_tick_assembled_features[self.feature_to_idx["BC-BO"]] = self._normalize_value(bc_bo_val, "BC-BO")
            if not np.isnan(norm_high) and not np.isnan(norm_low):
                if "BH-BL" in self.feature_to_idx:
                    bh_bl_val = norm_high - norm_low
                    current_tick_assembled_features[self.feature_to_idx["BH-BL"]] = self._normalize_value(bh_bl_val, "BH-BL")
            if not np.isnan(norm_high) and not np.isnan(norm_open):
                if "BH-BO" in self.feature_to_idx:
                    bh_bo_val = norm_high - norm_open
                    current_tick_assembled_features[self.feature_to_idx["BH-BO"]] = self._normalize_value(bh_bo_val, "BH-BO")
            if not np.isnan(norm_open) and not np.isnan(norm_low):
                if "BO-BL" in self.feature_to_idx:
                    bo_bl_val = norm_open - norm_low
                    current_tick_assembled_features[self.feature_to_idx["BO-BL"]] = self._normalize_value(bo_bl_val, "BO-BL")

            # 6. Calculate and fill log_return (REINSTATED AND MODIFIED)
            # log_return is now calculated from the sequence of norm_close values.
            if "log_return" in self.feature_to_idx:
                log_return_val_to_normalize = 0.0  # Default to 0 log return (no change)
                if (self.previous_normalized_close is not None and
                        self.previous_normalized_close > 1e-9 and  # Avoid division by zero or tiny numbers
                        pd.notnull(norm_close) and
                        norm_close > 1e-9): # Ensure current close is also valid
                    
                    # Denormalize previous and current close to calculate log_return on actual prices
                    # This is more robust than calculating on normalized values if normalization is non-linear
                    # or if min/max are very different, though simple ratio of normalized is often used.
                    # For simplicity with current _normalize/_denormalize, let's try with normalized first.
                    # If issues arise, switch to denormalized for ratio.
                    ratio = norm_close / self.previous_normalized_close
                    if ratio > 1e-9: # Ensure ratio is positive
                        log_return_val_to_normalize = np.log(ratio)
                    # else: ratio is zero or negative, log_return remains 0.0
                
                # Store the raw log_return; it will be normalized if normalization_params for "log_return" exist.
                # The _normalize_value function handles this.
                current_tick_assembled_features[self.feature_to_idx["log_return"]] = self._normalize_value(log_return_val_to_normalize, "log_return")

            if pd.notnull(norm_close): # Update previous_normalized_close for the next step
               self.previous_normalized_close = norm_close


            # 7. Fill historical tick data (e.g., CLOSE_15m_tick_X)
            # This section is now mostly skipped if tick features are in decoder_output_feature_names.
            decoder_outputs_set = set(self.params.get("decoder_output_feature_names", []))
            
            needs_tick_derivation = False # Initialize
            
            # Determine if any tick-like features need derivation.
            # These are features in full_feature_names_ordered that look like ticks 
            # but are not present in decoder_outputs_set.
            # Example prefixes, adjust if your tick naming patterns are different.
            potential_tick_feature_prefixes = ["CLOSE_15m_tick_", "CLOSE_30m_tick_"]
            
            current_full_feature_names = self.params.get("full_feature_names_ordered")
            if current_full_feature_names: # Ensure the list exists
                for feat_name_full in current_full_feature_names:
                    is_potential_tick_feature = False
                    for prefix in potential_tick_feature_prefixes:
                        if feat_name_full.startswith(prefix):
                            is_potential_tick_feature = True
                            break # Found a matching prefix for this feature
                    
                    if is_potential_tick_feature:
                        # Check if this potential tick feature is in feature_to_idx (i.e., expected in the output)
                        # AND not provided by the decoder.
                        if feat_name_full in self.feature_to_idx and feat_name_full not in decoder_outputs_set:
                            needs_tick_derivation = True
                            print(f"GeneratorPlugin: Tick feature '{feat_name_full}' identified for derivation (not in decoder outputs).")
                            break # Found one feature needing derivation, so set flag and stop checking.
            
            if needs_tick_derivation:
                # Define tick_configs here, as it's needed by the subsequent logic
                # (which is assumed to be present in your executed file, causing the error).
                tick_configs = [
                    ("CLOSE_15m_tick_{}", 15, 8),  # (pattern, interval_minutes, num_sub_ticks)
                    ("CLOSE_30m_tick_{}", 30, 8),
                    # Add other tick configurations if necessary based on your feature names
                ]
                print(f"GeneratorPlugin: 'tick_configs' defined for deriving tick features: {tick_configs}")

                # The user's executed file likely has the historical tick derivation logic here,
                # which includes the loop: for pattern, _, num_sub_ticks in tick_configs:
                # This is where the NameError for 'tick_configs' occurs if it's not defined above.
                # Since the attached file has 'pass', I will keep 'pass' here.
                # The critical fix is defining 'tick_configs' just above this comment block.
                
                # Example of how the user's code might look (leading to the error if tick_configs is undefined):
                # num_generated_steps_for_ticks = len(generated_sequence_all_features_list)
                # dataset_periodicity_str_for_ticks = self.params.get('dataset_periodicity', '1h').lower()
                # main_interval_minutes = 0
                # # ... logic to set main_interval_minutes based on dataset_periodicity_str_for_ticks ...

                # for pattern, interval_minutes, num_sub_ticks in tick_configs: # THIS IS THE LINE CAUSING THE ERROR
                #     if main_interval_minutes == 0 or main_interval_minutes % interval_minutes != 0:
                #         continue
                #     ticks_per_main_interval = main_interval_minutes // interval_minutes
                #     for i_tick in range(1, num_sub_ticks + 1):
                #         tick_feat_name = pattern.format(i_tick)
                #         if tick_feat_name in self.feature_to_idx and tick_feat_name not in decoder_outputs_set:
                #             # ... actual derivation logic for the tick feature ...
                #             # val_to_fill = ...
                #             # current_tick_assembled_features[self.feature_to_idx[tick_feat_name]] = val_to_fill
                #             pass
                pass # Placeholder for the actual tick derivation logic the user has in their executed file.
            
            # 7b. Fill DATE_TIME (placeholder float index) 
            if 'DATE_TIME' in self.feature_to_idx:
                current_tick_assembled_features[self.feature_to_idx['DATE_TIME']] = np.float32(t)

            # 8. Placeholder for any remaining unfilled (NaN) features
            # These are features in full_feature_names_ordered not covered above (e.g., STL, Wavelet, MTM, historical ticks if not from window)
            for i, feat_name in enumerate(self.params["full_feature_names_ordered"]):
                if np.isnan(current_tick_assembled_features[i]):
                    # Try to use the value from the previous step in the input window
                    prev_window_val = current_input_feature_window[-1, i]
                    if pd.notnull(prev_window_val) and not np.isnan(prev_window_val): # Check if it's a valid number
                        current_tick_assembled_features[i] = prev_window_val
                    else:
                        # Fallback: small random normalized value (0-1) or 0.5 as neutral
                        # Avoid exact 0.0 if possible, as per user's request about zeros.
                        current_tick_assembled_features[i] = np.random.uniform(0.01, 0.1) # Small, non-zero
                        # Alternatively, if the feature has normalization params, use its mid-point
                        # if self.normalization_params and feat_name in self.normalization_params:
                        #    current_tick_assembled_features[i] = 0.5
                        # else: # If no norm params, it's harder to pick a good default
                        #    current_tick_assembled_features[i] = np.random.uniform(0.01, 0.1) 
                    # print(f"GeneratorPlugin: Warning - Feature '{feat_name}' was not generated. Filled with placeholder: {current_tick_assembled_features[i]:.4f}")

            generated_sequence_all_features_list.append(current_tick_assembled_features)

            current_input_feature_window = np.roll(current_input_feature_window, -1, axis=0)
            current_input_feature_window[-1, :] = current_tick_assembled_features

            if len(ohlc_history_for_ti_list) > self.params["ti_calculation_min_lookback"] + 50: # Buffer
                ohlc_history_for_ti_list.pop(0)
        
        final_generated_sequence = np.array(generated_sequence_all_features_list, dtype=np.float32)
        if np.isnan(final_generated_sequence).any():
            nan_counts = np.sum(np.isnan(final_generated_sequence), axis=0)
            nan_features = [self.params["full_feature_names_ordered"][i] for i, count in enumerate(nan_counts) if count > 0]
            print(f"GeneratorPlugin: Warning - NaNs found in final_generated_sequence. Features with NaNs: {nan_features}. Replacing with 0.01.")
            final_generated_sequence = np.nan_to_num(final_generated_sequence, nan=0.01)

        return np.expand_dims(final_generated_sequence, axis=0)

    def update_model(self, new_model: Model):
        """
        Actualiza el modelo del generador. Usado por GANTrainerPlugin después del entrenamiento de GAN.
        """
        if not isinstance(new_model, Model):
            raise TypeError(f"new_model debe ser un modelo Keras, se recibió {type(new_model)}")
        print("GeneratorPlugin: Actualizando sequential_model con una nueva instancia de modelo.")
        self.sequential_model = new_model
        self.model = new_model # Mantener la alias de self.model consistente
