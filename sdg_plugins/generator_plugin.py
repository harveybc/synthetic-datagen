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
        old_full_feature_names = self.params.get("full_feature_names_ordered")

        # --- MODIFICATION: Update self.main_config first ---
        # This ensures that if kwargs contains updates to main config keys (like x_train_file),
        # they are reflected in self.main_config before deriving paths from it.
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs) 
        else: # Should not happen if __init__ ran correctly and self.main_config was set
            print("GeneratorPlugin: Warning - self.main_config not found or is None during set_params. Initializing from kwargs.")
            self.main_config = kwargs.copy()

        # Determine the file path for initial CLOSE anchor from the (potentially updated) main config
        initial_close_file_path_candidate = self.main_config.get(
            "x_train_file", self.main_config.get("real_data_file") # Uses updated self.main_config
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
        # We track the path used to load it with self._last_initial_close_file_path
        
        # --- Refined logic for initial_close_anchor reload ---
        last_loaded_path_attr = '_last_initial_close_file_path'
        current_last_loaded_path = getattr(self, last_loaded_path_attr, None)

        if initial_close_file_path_candidate:
            if self.initial_denormalized_close_anchor is None or \
               current_last_loaded_path != initial_close_file_path_candidate:
                print(f"GeneratorPlugin: Reloading initial close anchor. Reason: Anchor is None ({self.initial_denormalized_close_anchor is None}) or path changed (current: '{initial_close_file_path_candidate}', last: '{current_last_loaded_path}').")
                self._load_initial_close_anchor(initial_close_file_path_candidate)
                setattr(self, last_loaded_path_attr, initial_close_file_path_candidate)
        elif not initial_close_file_path_candidate and self.initial_denormalized_close_anchor is not None:
            print("GeneratorPlugin: Warning - x_train_file path for initial close anchor became None, but anchor was already loaded. Keeping existing anchor. Clearing last loaded path.")
            setattr(self, last_loaded_path_attr, None) 
            # Optionally, consider clearing self.initial_denormalized_close_anchor here if the policy is to nullify it
            # when the path is removed. For now, it's kept if already loaded.

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
        if ohlc_history_df.empty:
            nan_placeholder_df = pd.DataFrame(columns=self.params["ti_feature_names"], index=[0]).astype(np.float32)
            nan_placeholder_df = nan_placeholder_df.fillna(np.nan)
            return nan_placeholder_df

        ohlc_map = {
            self.params["ohlc_feature_names"][0]: 'open',
            self.params["ohlc_feature_names"][1]: 'high',
            self.params["ohlc_feature_names"][2]: 'low',
            self.params["ohlc_feature_names"][3]: 'close'
        }
        missing_ohlc_cols = [col for col in self.params["ohlc_feature_names"] if col not in ohlc_history_df.columns]
        if missing_ohlc_cols:
            print(f"GeneratorPlugin: Warning (_calculate_technical_indicators) - Missing OHLC columns in input: {missing_ohlc_cols}. TIs will be NaN.")
            nan_placeholder_df = pd.DataFrame(columns=self.params["ti_feature_names"], index=[0]).astype(np.float32)
            nan_placeholder_df = nan_placeholder_df.fillna(np.nan)
            return nan_placeholder_df
            
        df = ohlc_history_df.rename(columns=ohlc_map)
        ti_df_results = pd.DataFrame(index=df.index) # Results will have same index as input df
        p = self.params["ti_params"]
        ti_names_to_calculate = self.params["ti_feature_names"]

        # RSI
        rsi_len = p.get("rsi_length", 14)
        if 'RSI' in ti_names_to_calculate:
            if len(df['close']) >= rsi_len:
                rsi_series = ta.rsi(df['close'], length=rsi_len)
                ti_df_results['RSI'] = rsi_series if isinstance(rsi_series, pd.Series) else np.nan
            else:
                ti_df_results['RSI'] = np.nan

        # MACD
        macd_fast = p.get("macd_fast", 12)
        macd_slow = p.get("macd_slow", 26)
        macd_signal = p.get("macd_signal", 9)
        min_len_macd = macd_slow + macd_signal - 1 
        if any(x in ti_names_to_calculate for x in ['MACD', 'MACD_Histogram', 'MACD_Signal']):
            if len(df['close']) >= min_len_macd:
                macd_output_df = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                if isinstance(macd_output_df, pd.DataFrame) and not macd_output_df.empty:
                    macd_col_name = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
                    hist_col_name = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
                    signal_col_name = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"

                    if 'MACD' in ti_names_to_calculate:
                        ti_df_results['MACD'] = macd_output_df[macd_col_name] if macd_col_name in macd_output_df.columns else np.nan
                    if 'MACD_Histogram' in ti_names_to_calculate:
                        ti_df_results['MACD_Histogram'] = macd_output_df[hist_col_name] if hist_col_name in macd_output_df.columns else np.nan
                    if 'MACD_Signal' in ti_names_to_calculate:
                        ti_df_results['MACD_Signal'] = macd_output_df[signal_col_name] if signal_col_name in macd_output_df.columns else np.nan
                else:
                    if 'MACD' in ti_names_to_calculate: ti_df_results['MACD'] = np.nan
                    if 'MACD_Histogram' in ti_names_to_calculate: ti_df_results['MACD_Histogram'] = np.nan
                    if 'MACD_Signal' in ti_names_to_calculate: ti_df_results['MACD_Signal'] = np.nan
            else:
                if 'MACD' in ti_names_to_calculate: ti_df_results['MACD'] = np.nan
                if 'MACD_Histogram' in ti_names_to_calculate: ti_df_results['MACD_Histogram'] = np.nan
                if 'MACD_Signal' in ti_names_to_calculate: ti_df_results['MACD_Signal'] = np.nan

        # Stochastic
        stoch_k_period = p.get("stoch_k", 14)
        stoch_d_period = p.get("stoch_d", 3) # This is the 'd' passed to ta.stoch
        stoch_smooth_k_period = p.get("stoch_smooth_k", 3)

        # Min length for a non-NaN smoothed %K value
        min_len_for_k_smooth = stoch_k_period + stoch_smooth_k_period - 1
        # Min length for a non-NaN %D value (based on smoothed %K and stoch_d_period)
        min_len_for_d_final = min_len_for_k_smooth + stoch_d_period - 1

        stoch_k_wanted = 'Stochastic_%K' in ti_names_to_calculate
        stoch_d_wanted = 'Stochastic_%D' in ti_names_to_calculate

        # Initialize to NaN
        if stoch_k_wanted: ti_df_results['Stochastic_%K'] = np.nan
        if stoch_d_wanted: ti_df_results['Stochastic_%D'] = np.nan

        # If stoch_d_period (from config, typically > 0) is used, 
        # ta.stoch will attempt to calculate %D.
        # This is only safe from internal pandas-ta errors if len(df) >= min_len_for_d_final.
        if stoch_d_period > 0: # Assuming d_period from config is the one to use
            if len(df['high']) >= min_len_for_d_final:
                if stoch_k_wanted or stoch_d_wanted: # Only call if at least one is actually needed
                    stoch_output_df = ta.stoch(df['high'], df['low'], df['close'], 
                                               k=stoch_k_period, 
                                               d=stoch_d_period, # Pass the configured d period
                                               smooth_k=stoch_smooth_k_period)
                    
                    if isinstance(stoch_output_df, pd.DataFrame) and not stoch_output_df.empty:
                        # Standard column names from pandas-ta for stoch
                        k_col_name = f"STOCHk_{stoch_k_period}_{stoch_d_period}_{stoch_smooth_k_period}"
                        d_col_name = f"STOCHd_{stoch_k_period}_{stoch_d_period}_{stoch_smooth_k_period}"

                        if stoch_k_wanted and k_col_name in stoch_output_df.columns:
                            ti_df_results['Stochastic_%K'] = stoch_output_df[k_col_name]
                        
                        if stoch_d_wanted and d_col_name in stoch_output_df.columns:
                            ti_df_results['Stochastic_%D'] = stoch_output_df[d_col_name]
            # Else (len < min_len_for_d_final): %K and %D remain NaN, ta.stoch is not called.
        
        elif stoch_d_period == 0: # User explicitly configured d=0 (wants %K only, no %D)
            if stoch_k_wanted:
                if len(df['high']) >= min_len_for_k_smooth:
                    # Call ta.stoch with d=0. This might still be problematic in some pandas-ta versions
                    # if it expects stoch_d.name later. For safety, one might need to use a version
                    # of stoch that only computes K, or handle this case very carefully.
                    # Assuming for now d=0 is intended to skip %D calculation part that causes error.
                    # However, the error `stoch_d.name` implies `stoch_d` is None even if `d=0`.
                    # This path (d=0) is risky with the current pandas-ta error.
                    # Safest is to treat d=0 as "no D line", and if K is wanted, it's subject to K's own length.
                    # But the call to ta.stoch itself might be the issue if d=0 leads to stoch_d=None.
                    # Given the error, it's safer to avoid calling ta.stoch if d=0 might lead to stoch_d=None.
                    # For now, if d=0, we assume only K is wanted and it needs min_len_for_k_smooth.
                    # This part is tricky. The previous logic (stricter guard) is safer.
                    # Reverting to the stricter guard: if d_period from config is >0, then min_len_for_d_final is the guard.
                    # If d_period from config is 0, then this block is effectively "don't calculate D".
                    # The code above (if stoch_d_period > 0) handles the common case.
                    # If stoch_d_period is truly 0 from config, and only K is wanted:
                    try:
                        # Attempt to get K only, by passing d=0. This is speculative.
                        stoch_output_df_k_only = ta.stoch(df['high'], df['low'], df['close'],
                                                          k=stoch_k_period,
                                                          d=0, # Explicitly pass d=0
                                                          smooth_k=stoch_smooth_k_period)
                        if isinstance(stoch_output_df_k_only, pd.DataFrame) and not stoch_output_df_k_only.empty:
                             k_col_name_d0 = f"STOCHk_{stoch_k_period}_0_{stoch_smooth_k_period}" # Name with d=0
                             if k_col_name_d0 in stoch_output_df_k_only.columns:
                                ti_df_results['Stochastic_%K'] = stoch_output_df_k_only[k_col_name_d0]
                             # Stochastic_%D remains NaN as d=0
                    except AttributeError as e_stoch_d0:
                        print(f"GeneratorPlugin: Warning - ta.stoch with d=0 failed for K-only: {e_stoch_d0}. Stochastic_%K will be NaN.")
                        # Stochastic_%K remains NaN
                # Else (len < min_len_for_k_smooth): Stochastic_%K remains NaN
        # If stoch_d_period from config is > 0, the first 'if stoch_d_period > 0:' block handles it.
        # The 'elif stoch_d_period == 0:' is for the specific case where user configures d=0.
        
        # ADX
        adx_len_param = p.get("adx_length", 14)
        # Min length for DMI lines (+DI, -DI)
        min_len_dmi_lines = adx_len_param 
        # Min length for ADX value (DMI period + ADX smoothing period - 1)
        min_len_adx_value = adx_len_param * 2 -1 
        
        if any(x in ti_names_to_calculate for x in ['ADX', 'DI+', 'DI-']):
            # Check if data is long enough for at least DMI lines
            if len(df['high']) >= min_len_dmi_lines:
                adx_output_df = ta.adx(df['high'], df['low'], df['close'], length=adx_len_param)
                if isinstance(adx_output_df, pd.DataFrame) and not adx_output_df.empty:
                    # Standard column names from pandas-ta for adx
                    adx_col_name = f"ADX_{adx_len_param}"
                    dip_col_name = f"DMP_{adx_len_param}" # DI+
                    dim_col_name = f"DMN_{adx_len_param}" # DI-

                    if 'DI+' in ti_names_to_calculate:
                        ti_df_results['DI+'] = adx_output_df[dip_col_name] if dip_col_name in adx_output_df.columns else np.nan
                    if 'DI-' in ti_names_to_calculate:
                        ti_df_results['DI-'] = adx_output_df[dim_col_name] if dim_col_name in adx_output_df.columns else np.nan
                    
                    if 'ADX' in ti_names_to_calculate:
                        # Check if data is long enough for ADX value specifically
                        if len(df['high']) >= min_len_adx_value:
                            ti_df_results['ADX'] = adx_output_df[adx_col_name] if adx_col_name in adx_output_df.columns else np.nan
                        else:
                            ti_df_results['ADX'] = np.nan # Not enough data for ADX
                else: # adx_output_df is None, not a DataFrame, or empty
                    if 'ADX' in ti_names_to_calculate: ti_df_results['ADX'] = np.nan
                    if 'DI+' in ti_names_to_calculate: ti_df_results['DI+'] = np.nan
                    if 'DI-' in ti_names_to_calculate: ti_df_results['DI-'] = np.nan
            else: # len(df['high']) < min_len_dmi_lines
                if 'ADX' in ti_names_to_calculate: ti_df_results['ADX'] = np.nan
                if 'DI+' in ti_names_to_calculate: ti_df_results['DI+'] = np.nan
                if 'DI-' in ti_names_to_calculate: ti_df_results['DI-'] = np.nan

        # ATR
        atr_len = p.get("atr_length", 14)
        if 'ATR' in ti_names_to_calculate:
            if len(df['high']) >= atr_len: # ATR needs at least 'length' periods for first value
                 atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
                 ti_df_results['ATR'] = atr_series if isinstance(atr_series, pd.Series) else np.nan
            else:
                ti_df_results['ATR'] = np.nan

        # CCI
        cci_len = p.get("cci_length", 14)
        if 'CCI' in ti_names_to_calculate:
            if len(df['high']) >= cci_len: # CCI needs at least 'length' periods
                cci_series = ta.cci(df['high'], df['low'], df['close'], length=cci_len)
                ti_df_results['CCI'] = cci_series if isinstance(cci_series, pd.Series) else np.nan
            else:
                ti_df_results['CCI'] = np.nan
        
        # Williams %R
        willr_len = p.get("willr_length", 14)
        if 'WilliamsR' in ti_names_to_calculate:
            if len(df['high']) >= willr_len: # Williams %R needs at least 'length' periods
                willr_series = ta.willr(df['high'], df['low'], df['close'], length=willr_len)
                ti_df_results['WilliamsR'] = willr_series if isinstance(willr_series, pd.Series) else np.nan
            else:
                ti_df_results['WilliamsR'] = np.nan

        # Momentum
        mom_len = p.get("mom_length", 14)
        if 'Momentum' in ti_names_to_calculate:
            if len(df['close']) >= mom_len + 1: # Momentum needs length + 1 samples
                mom_series = ta.mom(df['close'], length=mom_len)
                ti_df_results['Momentum'] = mom_series if isinstance(mom_series, pd.Series) else np.nan
            else:
                ti_df_results['Momentum'] = np.nan
        
        # ROC
        roc_len = p.get("roc_length", 14)
        if 'ROC' in ti_names_to_calculate:
            if len(df['close']) >= roc_len + 1: # ROC needs length + 1 samples
                roc_series = ta.roc(df['close'], length=roc_len)
                ti_df_results['ROC'] = roc_series if isinstance(roc_series, pd.Series) else np.nan
            else:
                ti_df_results['ROC'] = np.nan

        # EMA
        ema_len = p.get("ema_length", 14)
        if 'EMA' in ti_names_to_calculate:
            if len(df['close']) >= ema_len: # EMA needs at least 'length' periods for first value
                ema_series = ta.ema(df['close'], length=ema_len)
                ti_df_results['EMA'] = ema_series if isinstance(ema_series, pd.Series) else np.nan
            else:
                ti_df_results['EMA'] = np.nan

        # Ensure all requested TI columns exist in ti_df_results, filling with NaN if not calculated
        for ti_name in self.params["ti_feature_names"]:
            if ti_name not in ti_df_results.columns:
                ti_df_results[ti_name] = np.nan
        
        # Select only the requested TIs and return the last row.
        if ti_df_results.empty and not df.empty: # Should not happen if df was not empty and ti_df_results was indexed to it
            # This case implies no TIs were calculated or assigned.
            # Create a DataFrame with NaNs based on df's index to ensure tail(1) works.
            for ti_name_fill in self.params["ti_feature_names"]: # Iterate to create columns
                 ti_df_results[ti_name_fill] = np.nan
        elif ti_df_results.empty and df.empty: # Input df was empty, already handled at the start
             nan_placeholder_df = pd.DataFrame(columns=self.params["ti_feature_names"], index=[0]).astype(np.float32)
             nan_placeholder_df = nan_placeholder_df.fillna(np.nan)
             return nan_placeholder_df

        # Ensure all target columns are present before selecting, to avoid KeyError
        final_ti_columns_present = [col for col in self.params["ti_feature_names"] if col in ti_df_results.columns]
        
        if not final_ti_columns_present: # No TIs could be put into ti_df_results
             nan_placeholder_df = pd.DataFrame(np.nan, index=[0], columns=self.params["ti_feature_names"]).astype(np.float32)
             return nan_placeholder_df

        # Return the last row of the calculated TIs
        last_row_tis = ti_df_results[final_ti_columns_present].tail(1).reset_index(drop=True)
        
        # Reindex to ensure all originally requested TIs are present, filling missing ones with NaN
        last_row_tis_reindexed = last_row_tis.reindex(columns=self.params["ti_feature_names"], fill_value=np.nan)
        
        return last_row_tis_reindexed.astype(np.float32)

    # ADD THIS HELPER METHOD (adapted from FeederPlugin)
    def _get_scaled_date_features_for_plugin(self, datetime_obj: pd.Timestamp) -> np.ndarray:
        """Generates scaled (sin/cos) date features for a given datetime. Uses Generator's params and main_config."""
        date_features = []
        # Ensure self.main_config is available, it's set in __init__ from the passed config
        main_cfg = getattr(self, 'main_config', {}) # Fallback to empty dict if not found

        # Use self.params for feature names, self.main_config for max values (as Feeder does)
        date_conditional_names = self.params.get("date_conditional_feature_names", [])

        if "day_of_month" in date_conditional_names:
            dom = datetime_obj.day
            max_dom = main_cfg.get("feeder_max_day_of_month", 31)
            date_features.extend([np.sin(2 * np.pi * dom / max_dom), np.cos(2 * np.pi * dom / max_dom)])
        if "hour_of_day" in date_conditional_names:
            hod = datetime_obj.hour
            max_hod = main_cfg.get("feeder_max_hour_of_day", 23)
            date_features.extend([np.sin(2 * np.pi * hod / (max_hod + 1)), np.cos(2 * np.pi * hod / (max_hod + 1))])
        if "day_of_week" in date_conditional_names:
            dow = datetime_obj.dayofweek
            max_dow = main_cfg.get("feeder_max_day_of_week", 6)
            date_features.extend([np.sin(2 * np.pi * dow / (max_dow + 1)), np.cos(2 * np.pi * dow / (max_dow + 1))])
        if "day_of_year" in date_conditional_names:
            doy = datetime_obj.dayofyear
            max_doy = main_cfg.get("feeder_max_day_of_year", 366)
            date_features.extend([np.sin(2 * np.pi * doy / max_doy), np.cos(2 * np.pi * doy / max_doy)])
        return np.array(date_features, dtype=np.float32)

    def generate(self,
                 feeder_outputs_sequence: List[Dict[str, np.ndarray]],
                 sequence_length_T: int,
                 initial_full_feature_window: Optional[np.ndarray] = None,
                 initial_datetimes_for_window: Optional[pd.Series] = None,
                 true_prev_close_for_initial_window_log_return: Optional[float] = None # NEW ARGUMENT
                ) -> np.ndarray:
        """
        Genera una secuencia de variables base de manera autorregresiva usando el modelo secuencial cargado.
        """
        decoder_input_window_size = self.params["decoder_input_window_size"]
        min_ohlc_hist_len = self.params["ti_calculation_min_lookback"]
        
        ohlc_names = self.params["ohlc_feature_names"]

        # Initialize self.previous_normalized_close for this generation sequence
        self.previous_normalized_close = None 
        # This will be properly set after the initial window is processed or from its last tick.

        ohlc_history_for_ti_list = [] 

        current_input_feature_window = np.zeros((decoder_input_window_size, self.num_all_features), dtype=np.float32)
        if initial_full_feature_window is not None:
            if initial_full_feature_window.shape == current_input_feature_window.shape:
                current_input_feature_window = initial_full_feature_window.astype(np.float32).copy()
                
                # Populate ohlc_history_for_ti_list from the initial window (denormalized)
                # This part is crucial for TI calculation during pre-fill and main loop.
                start_idx_for_ohlc_hist = max(0, decoder_input_window_size - min_ohlc_hist_len)
                for i in range(decoder_input_window_size): # Iterate full window to build complete history
                    row_ohlc_norm_values = {
                        name: current_input_feature_window[i, self.feature_to_idx[name]]
                        for name in ohlc_names if name in self.feature_to_idx and pd.notnull(current_input_feature_window[i, self.feature_to_idx[name]])
                    }
                    if len(row_ohlc_norm_values) == len(ohlc_names): # Ensure all OHLC are present
                        ohlc_dict_denorm = {
                            name: self._denormalize_value(row_ohlc_norm_values.get(name, np.nan), name)
                            for name in ohlc_names
                        }
                        if all(pd.notnull(v) for v in ohlc_dict_denorm.values()):
                            ohlc_history_for_ti_list.append(ohlc_dict_denorm)
                
                if len(ohlc_history_for_ti_list) > min_ohlc_hist_len + 50: # Trim if it became too long
                    ohlc_history_for_ti_list = ohlc_history_for_ti_list[-(min_ohlc_hist_len + 50):]


                # --- BEGIN PRE-FILL OF DERIVED FEATURES IN current_input_feature_window ---
                if initial_datetimes_for_window is not None and \
                   len(initial_datetimes_for_window) == decoder_input_window_size:
                    
                    print("GeneratorPlugin: Pre-filling derived features in initial_full_feature_window...")
                    _local_prev_norm_close_for_prefill = None
                    
                    # --- MODIFIED: Initialize _local_prev_norm_close_for_prefill ---
                    if true_prev_close_for_initial_window_log_return is not None and pd.notnull(true_prev_close_for_initial_window_log_return):
                        _local_prev_norm_close_for_prefill = true_prev_close_for_initial_window_log_return
                        print(f"DEBUG GeneratorPlugin: Pre-fill using true_prev_close_for_initial_window_log_return: {_local_prev_norm_close_for_prefill}")
                    elif 'CLOSE' in self.feature_to_idx and pd.notnull(current_input_feature_window[0, self.feature_to_idx['CLOSE']]):
                        # Fallback: if true previous close isn't available, use the first CLOSE in the window,
                        # which means the first log_return in the window will be 0.
                        _local_prev_norm_close_for_prefill = current_input_feature_window[0, self.feature_to_idx['CLOSE']]
                        print(f"DEBUG GeneratorPlugin: Pre-fill using fallback for _local_prev_norm_close_for_prefill (first CLOSE in window): {_local_prev_norm_close_for_prefill}")
                    else:
                        print("DEBUG GeneratorPlugin: Pre-fill: _local_prev_norm_close_for_prefill could not be initialized from true_prev or window's first CLOSE.")
                    # --- END MODIFICATION ---

                    for i_prefill in range(decoder_input_window_size):
                        dt_obj_prefill = initial_datetimes_for_window.iloc[i_prefill]
                        
                        # 1. Raw Date Features (day_of_month, hour_of_day, day_of_week)
                        raw_date_map_prefill = {
                            "day_of_month": dt_obj_prefill.day,
                            "hour_of_day": dt_obj_prefill.hour,
                            "day_of_week": dt_obj_prefill.dayofweek
                        }
                        for raw_feat_name, raw_val in raw_date_map_prefill.items():
                            if raw_feat_name in self.feature_to_idx:
                                current_input_feature_window[i_prefill, self.feature_to_idx[raw_feat_name]] = self._normalize_value(raw_val, raw_feat_name)

                        # 2. Sin/Cos Date Features
                        scaled_date_features_prefill_arr = self._get_scaled_date_features_for_plugin(dt_obj_prefill)
                        sincos_idx_counter = 0
                        for original_date_feat_name_cfg in self.params.get("date_conditional_feature_names", []):
                            for suffix in ["_sin", "_cos"]:
                                feat_name_transformed = f"{original_date_feat_name_cfg}{suffix}"
                                if feat_name_transformed in self.feature_to_idx:
                                    if sincos_idx_counter < len(scaled_date_features_prefill_arr):
                                        current_input_feature_window[i_prefill, self.feature_to_idx[feat_name_transformed]] = scaled_date_features_prefill_arr[sincos_idx_counter]
                                    else: # Should not happen if _get_scaled_date_features_for_plugin is correct
                                        current_input_feature_window[i_prefill, self.feature_to_idx[feat_name_transformed]] = 0.0 
                                    sincos_idx_counter += 1
                        
                        # Fundamental features: Assumed to be correctly populated from preprocessor output or remain 0.0/NaN

                        # 3. Technical Indicators
                        # Use the ohlc_history_for_ti_list (which contains denormalized OHLC for the whole initial window)
                        # We need to pass a DataFrame slice of this history to _calculate_technical_indicators
                        # The slice should be up to and including the current row i_prefill for TI calculation.
                        if len(ohlc_history_for_ti_list) > i_prefill : # Ensure we have history up to this point
                            history_slice_for_ti_df = pd.DataFrame(ohlc_history_for_ti_list[:i_prefill+1])
                            if not history_slice_for_ti_df.empty:
                                calculated_ti_for_prefill_df = self._calculate_technical_indicators(history_slice_for_ti_df) # Returns DF for last row
                                if not calculated_ti_for_prefill_df.empty:
                                    for ti_name in self.params["ti_feature_names"]:
                                        if ti_name in self.feature_to_idx and ti_name in calculated_ti_for_prefill_df.columns:
                                            val_ti_denorm = calculated_ti_for_prefill_df.iloc[0][ti_name]
                                            if pd.notnull(val_ti_denorm):
                                                current_input_feature_window[i_prefill, self.feature_to_idx[ti_name]] = self._normalize_value(val_ti_denorm, ti_name)
                                            else: # If TI is NaN, keep it NaN (will be 0.01 later if needed for model input)
                                                current_input_feature_window[i_prefill, self.feature_to_idx[ti_name]] = np.nan


                        # 4. Log Return
                        # CLOSE for current prefill step
                        norm_c_prefill = np.nan
                        if 'CLOSE' in self.feature_to_idx and pd.notnull(current_input_feature_window[i_prefill, self.feature_to_idx['CLOSE']]):
                             norm_c_prefill = current_input_feature_window[i_prefill, self.feature_to_idx['CLOSE']]
                        elif 'OPEN' in self.feature_to_idx and pd.notnull(current_input_feature_window[i_prefill, self.feature_to_idx['OPEN']]):
                             norm_c_prefill = current_input_feature_window[i_prefill, self.feature_to_idx['OPEN']] # Fallback

                        if "log_return" in self.feature_to_idx:
                            log_return_val_to_norm_prefill = 0.0
                            if _local_prev_norm_close_for_prefill is not None and pd.notnull(norm_c_prefill) and \
                               _local_prev_norm_close_for_prefill > 1e-9 and norm_c_prefill > 1e-9:
                                log_return_val_to_norm_prefill = np.log(norm_c_prefill / _local_prev_norm_close_for_prefill)
                            
                            current_input_feature_window[i_prefill, self.feature_to_idx["log_return"]] = self._normalize_value(log_return_val_to_norm_prefill, "log_return")
                        
                        if pd.notnull(norm_c_prefill):
                            _local_prev_norm_close_for_prefill = norm_c_prefill
                    
                    print("GeneratorPlugin: Finished pre-filling derived features in initial_full_feature_window.")
                # --- END PRE-FILL ---
                elif initial_full_feature_window is not None: # initial_datetimes_for_window was missing or mismatched
                     print("GeneratorPlugin: Warning - initial_full_feature_window provided, but initial_datetimes_for_window is missing or mismatched. Cannot pre-fill derived features in the window.")

            else: # Shape mismatch for initial_full_feature_window
                raise ValueError(f"Shape mismatch for initial_full_feature_window. Expected {current_input_feature_window.shape}, got {initial_full_feature_window.shape}")
        else: # No initial_full_feature_window provided
            print("GeneratorPlugin Warning: No initial_full_feature_window provided. Using zeros. TIs will be NaN initially.")

        # Initialize self.previous_normalized_close for the main generation loop
        # This should be based on the *last* tick of the (now potentially pre-filled) current_input_feature_window
        if 'CLOSE' in self.feature_to_idx and current_input_feature_window.shape[0] > 0:
            last_norm_close_in_window = current_input_feature_window[-1, self.feature_to_idx['CLOSE']]
            if pd.notnull(last_norm_close_in_window):
                self.previous_normalized_close = float(last_norm_close_in_window)
                print(f"GeneratorPlugin: Initialized previous_normalized_close for main loop from window's last tick: {self.previous_normalized_close}")
            elif self.initial_denormalized_close_anchor is not None: # Fallback to anchor if last window CLOSE is bad
                self.previous_normalized_close = self._normalize_value(self.initial_denormalized_close_anchor, "CLOSE")
                print(f"GeneratorPlugin: Initialized previous_normalized_close for main loop from initial_denormalized_close_anchor: {self.previous_normalized_close}")


        generated_sequence_all_features_list = []

        # --- MAIN GENERATION LOOP ---
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

            # Prepare the current window input for the model
            # The model expects a batch dimension, so expand dims for current_input_feature_window
            current_window_for_model = np.expand_dims(current_input_feature_window, axis=0)

            decoder_inputs = {
                self.params["decoder_input_name_latent"]: zt,
                self.params["decoder_input_name_window"]: current_window_for_model, # ADDED THIS LINE
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
            for i, name in enumerate(self.params["decoder_output_feature_names"]):
                if name in self.feature_to_idx:
                    current_tick_assembled_features[self.feature_to_idx[name]] = decoded_features_for_current_tick[i]
            
            # Extract norm_open, norm_high, norm_low from decoder outputs (already in current_tick_assembled_features)
            norm_open = current_tick_assembled_features[self.feature_to_idx[ohlc_names[0]]] if ohlc_names[0] in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx[ohlc_names[0]]]) else np.nan
            norm_high = current_tick_assembled_features[self.feature_to_idx[ohlc_names[1]]] if ohlc_names[1] in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx[ohlc_names[1]]]) else np.nan
            norm_low = current_tick_assembled_features[self.feature_to_idx[ohlc_names[2]]] if ohlc_names[2] in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx[ohlc_names[2]]]) else np.nan

            # 1.B. Calculate norm_close using OPEN and BC-BO if available from decoder
            calculated_norm_close_from_obcbo = np.nan
            if "OPEN" in self.feature_to_idx and "BC-BO" in self.feature_to_idx and "CLOSE" in self.feature_to_idx:
                # norm_open is already extracted above
                norm_bc_bo_for_calc = current_tick_assembled_features[self.feature_to_idx["BC-BO"]] if "BC-BO" in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx["BC-BO"]]) else np.nan

                # --- BEGIN DEBUG PRINTS for CLOSE calculation ---
                if t < 2: # Print for the first 2 ticks
                    print(f"DEBUG GeneratorPlugin (t={t}): ---- Start CLOSE Calculation ----")
                    open_idx = self.feature_to_idx.get('OPEN', -1)
                    bcbo_idx = self.feature_to_idx.get('BC-BO', -1)
                    print(f"DEBUG GeneratorPlugin (t={t}): norm_open raw from features: {current_tick_assembled_features[open_idx] if open_idx != -1 else 'OPEN_NOT_IN_feature_to_idx'}")
                    print(f"DEBUG GeneratorPlugin (t={t}): norm_open extracted: {norm_open}")
                    print(f"DEBUG GeneratorPlugin (t={t}): norm_bc_bo_for_calc raw from features: {current_tick_assembled_features[bcbo_idx] if bcbo_idx != -1 else 'BC-BO_NOT_IN_feature_to_idx'}")
                    print(f"DEBUG GeneratorPlugin (t={t}): norm_bc_bo_for_calc extracted: {norm_bc_bo_for_calc}")

                if pd.notnull(norm_open) and pd.notnull(norm_bc_bo_for_calc):
                    denorm_open_val = self._denormalize_value(norm_open, "OPEN")
                    denorm_bc_bo_val = self._denormalize_value(norm_bc_bo_for_calc, "BC-BO") # Assuming BC-BO is also normalized like other features
                    
                    if t < 2: # Print for the first 2 ticks
                        print(f"DEBUG GeneratorPlugin (t={t}): denorm_open_val: {denorm_open_val}, denorm_bc_bo_val: {denorm_bc_bo_val}")

                    if pd.notnull(denorm_open_val) and pd.notnull(denorm_bc_bo_val):
                        # Corrected logic: CLOSE = OPEN + (BC-BO)
                        denormalized_close_candidate = denorm_open_val + denorm_bc_bo_val
                        calculated_norm_close_from_obcbo = self._normalize_value(denormalized_close_candidate, "CLOSE")

                        if t < 2: print(f"DEBUG GeneratorPlugin (t={t}): calculated_norm_close_from_obcbo (from OPEN+BCBO): {calculated_norm_close_from_obcbo}, denormalized_candidate: {denormalized_close_candidate}")
                    elif t < 2:
                        print(f"DEBUG GeneratorPlugin (t={t}): Skipping CLOSE calculation from OPEN+BCBO due to NaN in denorm_open_val or denorm_bc_bo_val.")
                elif t < 2:
                    print(f"DEBUG GeneratorPlugin (t={t}): Skipping denormalization due to NaN in norm_open or norm_bc_bo_for_calc.")
                # --- END DEBUG PRINTS for CLOSE calculation ---

            # Determine final norm_close for the current tick
            norm_close = np.nan # Initialize to NaN for this step's logic
            if "CLOSE" in self.feature_to_idx:
                if pd.notnull(calculated_norm_close_from_obcbo):
                    norm_close = calculated_norm_close_from_obcbo
                # --- MODIFIED: CLOSE derivation for t=0 and fallbacks ---
                elif t == 0: # First synthetic tick
                    if pd.notnull(norm_open): # Try to use OPEN from decoder
                        norm_close = norm_open
                        if t < 2: print(f"DEBUG GeneratorPlugin t={t}: CLOSE from O+BCBO is NaN. Using norm_open: {norm_close} as CLOSE for first synthetic tick.")
                    elif self.previous_normalized_close is not None and pd.notnull(self.previous_normalized_close): # Use previous actual close from end of real window
                        norm_close = self.previous_normalized_close
                        if t < 2: print(f"DEBUG GeneratorPlugin t={t}: CLOSE from O+BCBO is NaN, norm_open is NaN. Using self.previous_normalized_close: {norm_close} as CLOSE for first synthetic tick.")
                    else: # Absolute last resort if decoder gives no OPEN and no previous_normalized_close
                        norm_close = self._normalize_value(self.initial_denormalized_close_anchor, "CLOSE") if self.initial_denormalized_close_anchor is not None else 0.01
                        if t < 2: print(f"DEBUG GeneratorPlugin t={t}: CLOSE from O+BCBO is NaN, norm_open is NaN, previous_normalized_close is None. Using initial_denormalized_close_anchor (fallback): {norm_close} as CLOSE.")
                elif pd.notnull(norm_open): # Fallback to OPEN if CLOSE cannot be derived (for t > 0)
                    norm_close = norm_open
                else: # Last resort, use previous close if available (for t > 0)
                    norm_close = self.previous_normalized_close if self.previous_normalized_close is not None else 0.01
                # --- END MODIFICATION ---

            elif t < 2: # If 'CLOSE' is not even in feature_to_idx (config error)
                 print(f"DEBUG GeneratorPlugin t={t}: 'CLOSE' not in self.feature_to_idx. Cannot derive norm_close.")


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
            }
            for raw_feat_name, raw_val in raw_date_map.items():
                if raw_feat_name in self.feature_to_idx:
                    current_tick_assembled_features[self.feature_to_idx[raw_feat_name]] = self._normalize_value(float(raw_val), raw_feat_name)

            # 4. Calculate and fill Technical Indicators
            # Uses norm_open, norm_high, norm_low (from decoder) and norm_close (newly derived)
            current_ohlc_for_ti_calc_normalized = {
                ohlc_names[0]: norm_open,
                ohlc_names[1]: norm_high,
                ohlc_names[2]: norm_low,
                ohlc_names[3]: norm_close 
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

            # 5. Calculate and fill derived OHLC features that are NOT direct decoder outputs
            decoder_outputs_set = set(self.params.get("decoder_output_feature_names", []))
            
            potential_derived_ohlc_map = {
                "BC-BO": lambda dn_o, dn_h, dn_l, dn_c: dn_c - dn_o if pd.notnull(dn_c) and pd.notnull(dn_o) else np.nan,
                "BH-BL": lambda dn_o, dn_h, dn_l, dn_c: dn_h - dn_l if pd.notnull(dn_h) and pd.notnull(dn_l) else np.nan,
                "BH-BO": lambda dn_o, dn_h, dn_l, dn_c: dn_h - dn_o if pd.notnull(dn_h) and pd.notnull(dn_o) else np.nan,
                "BO-BL": lambda dn_o, dn_h, dn_l, dn_c: dn_o - dn_l if pd.notnull(dn_o) and pd.notnull(dn_l) else np.nan,
            }

            # Denormalize the core OHLC values from current_tick_assembled_features
            # (norm_open, norm_high, norm_low from decoder; norm_close derived and filled)
            dn_o_step5 = self._denormalize_value(current_tick_assembled_features[self.feature_to_idx["OPEN"]], "OPEN") if "OPEN" in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx["OPEN"]]) else np.nan
            dn_h_step5 = self._denormalize_value(current_tick_assembled_features[self.feature_to_idx["HIGH"]], "HIGH") if "HIGH" in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx["HIGH"]]) else np.nan
            dn_l_step5 = self._denormalize_value(current_tick_assembled_features[self.feature_to_idx["LOW"]], "LOW") if "LOW" in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx["LOW"]]) else np.nan
            dn_c_step5 = self._denormalize_value(current_tick_assembled_features[self.feature_to_idx["CLOSE"]], "CLOSE") if "CLOSE" in self.feature_to_idx and pd.notnull(current_tick_assembled_features[self.feature_to_idx["CLOSE"]]) else np.nan

            for feat_name, calc_func in potential_derived_ohlc_map.items():
                if feat_name in self.feature_to_idx and feat_name not in decoder_outputs_set:
                    val_denorm = calc_func(dn_o_step5, dn_h_step5, dn_l_step5, dn_c_step5)
                    if pd.notnull(val_denorm):
                        val_norm = self._normalize_value(val_denorm, feat_name)
                        current_tick_assembled_features[self.feature_to_idx[feat_name]] = val_norm


            # 6. Calculate and fill log_return
            if "log_return" in self.feature_to_idx:
                log_return_val_to_normalize = 0.0
                current_close_for_log_return = norm_close 
                
                if (self.previous_normalized_close is not None and
                        self.previous_normalized_close > 1e-9 and # Avoid log(0) or log(negative)
                        pd.notnull(current_close_for_log_return) and
                        current_close_for_log_return > 1e-9): # Avoid log(0) or log(negative)
                    
                    ratio = current_close_for_log_return / self.previous_normalized_close
                    if ratio > 1e-9: # Ensure ratio is positive for log
                        log_return_val_to_normalize = np.log(ratio)
                
                current_tick_assembled_features[self.feature_to_idx["log_return"]] = self._normalize_value(log_return_val_to_normalize, "log_return")

            if pd.notnull(norm_close): 
               self.previous_normalized_close = norm_close


            # 7. Fill historical tick data (e.g., CLOSE_15m_tick_X)
            # This section is now mostly skipped if tick features are in decoder_output_feature_names.
            # decoder_outputs_set was defined in Step 5
            
            needs_tick_derivation = False 
            potential_tick_feature_prefixes = ["CLOSE_15m_tick_", "CLOSE_30m_tick_"]
            current_full_feature_names = self.params.get("full_feature_names_ordered")

            if current_full_feature_names:
                for feat_name_full in current_full_feature_names:
                    is_potential_tick_feature = any(feat_name_full.startswith(prefix) for prefix in potential_tick_feature_prefixes)
                    if is_potential_tick_feature and feat_name_full in self.feature_to_idx and feat_name_full not in decoder_outputs_set:
                        needs_tick_derivation = True
                        # print(f"GeneratorPlugin: Tick feature '{feat_name_full}' identified for derivation.")
                        break 
            
            if needs_tick_derivation:
                tick_configs = [
                    ("CLOSE_15m_tick_{}", 15, 8),
                    ("CLOSE_30m_tick_{}", 30, 8),
                ]
                # print(f"GeneratorPlugin: 'tick_configs' defined for deriving tick features: {tick_configs}")
                # ... (Actual tick derivation logic would go here if needed) ...
                pass # Placeholder, as current config likely has ticks in decoder_output_feature_names
            
            # 7b. Fill DATE_TIME (placeholder float index) 
            if 'DATE_TIME' in self.feature_to_idx:
                current_tick_assembled_features[self.feature_to_idx['DATE_TIME']] = np.float32(t)

            # 8. Placeholder for any remaining unfilled (NaN) features
            for i, feat_name_iter in enumerate(self.params["full_feature_names_ordered"]): # Renamed feat_name to feat_name_iter
                if np.isnan(current_tick_assembled_features[i]):
                    # --- BEGIN DEBUG for Step 8 ---
                    if t < 2 and feat_name_iter == "CLOSE":
                        print(f"DEBUG GeneratorPlugin (t={t}, Step 8): Filling NaN for CLOSE.")
                        prev_win_val_close_idx = self.feature_to_idx.get('CLOSE', -1)
                        prev_win_val_close = current_input_feature_window[-1, i] if prev_win_val_close_idx != -1 and i == prev_win_val_close_idx else 'N/A_OR_CLOSE_NOT_FOUND'
                        print(f"DEBUG GeneratorPlugin (t={t}, Step 8): prev_window_val for CLOSE: {prev_win_val_close}")
                    # --- END DEBUG for Step 8 ---
                    prev_window_val = current_input_feature_window[-1, i]
                    if pd.notnull(prev_window_val) and not np.isnan(prev_window_val):
                        current_tick_assembled_features[i] = prev_window_val
                    else:
                        current_tick_assembled_features[i] = np.random.uniform(0.01, 0.1) 

            generated_sequence_all_features_list.append(current_tick_assembled_features)

            # --- MODIFICATION: Ensure no NaNs are fed into the rolling window for the model ---
            # current_tick_features_for_window = np.nan_to_num(current_tick_assembled_features, nan=0.01) # OLD
            # The current_tick_assembled_features might have NaNs for TIs if history is too short.
            # These NaNs should be replaced before updating current_input_feature_window.
            
            # Create a copy for the window update, replacing NaNs
            current_tick_features_for_model_window_update = np.nan_to_num(current_tick_assembled_features.copy(), nan=0.01)


            current_input_feature_window = np.roll(current_input_feature_window, -1, axis=0)
            current_input_feature_window[-1, :] = current_tick_features_for_model_window_update # Use NaN-replaced version for model's input window
        
            # Manage ohlc_history_for_ti_list (append new denormalized OHLC, pop old)
            # This was inside the TI calculation block, ensure it's correctly placed relative to history usage.
            # The current ohlc_history_for_ti_list management seems okay, it appends the latest denormalized OHLC
            # and trims if it gets too long. This happens *after* TIs for current tick are calculated.
            if len(ohlc_history_for_ti_list) > min_ohlc_hist_len + 50: 
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
