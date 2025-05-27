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

import numpy as np
from typing import Dict, Any, List, Optional
from tensorflow.keras.models import load_model, Model
import pandas as pd # Add pandas
import pandas_ta as ta # Add pandas-ta


class GeneratorPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "sequential_model_file": None,  # Ruta al modelo generador secuencial Keras (decoder)
        "decoder_input_window_size": 144, # Window size expected by the decoder's x_window input
        "full_feature_names_ordered": [], # List of all 45 feature names in order from normalized_d2.csv
        "decoder_output_feature_names": [], # Features directly output by the Keras decoder model
        "ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"], # For TI calculation
        "ti_feature_names": [ # Technical indicators to calculate based on normalized_d2.csv
            "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
            "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-",
            "ATR", "CCI", "WilliamsR", "Momentum", "ROC"
        ],
        "date_conditional_feature_names": ["day_of_month", "hour_of_day", "day_of_week"], # Expected from FeederPlugin
        "feeder_conditional_feature_names": ["S&P500_Close", "vix_close"], # Expected from FeederPlugin
        "ti_calculation_min_lookback": 200, # Min OHLC history for reliable TIs (e.g., 200-period EMA)
        "ti_params": { # Parameters for pandas-ta, align with tech_indicator.py or pandas-ta defaults
            "rsi_length": 14, "ema_length": 14, # pandas-ta EMA default is 10, using 14 to match common short-term
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "stoch_k": 14, "stoch_d": 3, "stoch_smooth_k": 3,
            "adx_length": 14, "atr_length": 14, "cci_length": 14, # pandas-ta CCI default is 20, using 14
            "willr_length": 14, "mom_length": 14, "roc_length": 14 # pandas-ta MOM/ROC default is 10, using 14
        },
        "batch_size_inference": 1,   # Típicamente, la generación secuencial se hace una secuencia a la vez
        # CRITICAL: Update these to match your Keras decoder's actual input layer names
        "decoder_input_name_latent": "input_latent_z",         # Placeholder name
        "decoder_input_name_window": "input_x_window",         # Placeholder name
        "decoder_input_name_conditions": "input_conditions_t", # Placeholder name
        "decoder_input_name_context": "input_context_h"        # Placeholder name
    }
    # Variables incluidas en el debug
    plugin_debug_vars = [
        "sequential_model_file", "decoder_input_window_size", "batch_size_inference",
        "full_feature_names_ordered", "decoder_output_feature_names",
        "ti_calculation_min_lookback"
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el GeneratorPlugin y carga el modelo generador secuencial.

        :param config: Diccionario con al menos 'sequential_model_file' y opcionalmente 'batch_size_inference', 'history_k', 'num_base_features', 'num_fundamental_features'.
        :raises ValueError: si no se proporciona 'sequential_model_file'.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        # Copia parámetros por defecto y aplica la configuración
        self.params = self.plugin_params.copy()
        self.set_params(**config)

        model_path = self.params.get("sequential_model_file")
        if not model_path:
            raise ValueError("El parámetro 'sequential_model_file' debe especificar la ruta al modelo generador secuencial.")

        if not self.params.get("full_feature_names_ordered"):
            raise ValueError("El parámetro 'full_feature_names_ordered' es obligatorio.")
        if not self.params.get("decoder_output_feature_names"):
            raise ValueError("El parámetro 'decoder_output_feature_names' es obligatorio.")

        try:
            self.sequential_model: Model = load_model(model_path, compile=False)
            print(f"GeneratorPlugin: Modelo decoder cargado desde {model_path}")
        except Exception as e:
            raise IOError(f"Error al cargar el modelo secuencial desde {model_path}. Detalle del error: {e}")
        
        self.model = self.sequential_model

        self.feature_to_idx = {name: i for i, name in enumerate(self.params["full_feature_names_ordered"])}
        self.num_all_features = len(self.params["full_feature_names_ordered"])
        self._validate_feature_name_consistency()

    def _validate_feature_name_consistency(self):
        """Validates that all configured feature name lists are consistent with full_feature_names_ordered."""
        full_set = set(self.params["full_feature_names_ordered"])
        
        name_lists_to_check = [
            "decoder_output_feature_names", "ohlc_feature_names", "ti_feature_names",
            "date_conditional_feature_names", "feeder_conditional_feature_names"
        ]
        for key in name_lists_to_check:
            sub_list = self.params.get(key, [])
            if not sub_list and key in ["decoder_output_feature_names", "ohlc_feature_names", "ti_feature_names"]: # These should not be empty
                 print(f"GeneratorPlugin Warning: Feature list '{key}' is empty in config. This might lead to errors if features are expected.")
            for feature_name in sub_list:
                if feature_name not in full_set:
                    raise ValueError(f"Feature '{feature_name}' en '{key}' no se encuentra en 'full_feature_names_ordered'.")
        
        if "DATE_TIME" not in full_set: # DATE_TIME is used as a placeholder column
            raise ValueError("'DATE_TIME' must be included in 'full_feature_names_ordered'.")
        if not self.params["ohlc_feature_names"] or len(self.params["ohlc_feature_names"]) != 4:
            raise ValueError("'ohlc_feature_names' must be a list of 4 names (e.g., OPEN, HIGH, LOW, CLOSE).")


    def set_params(self, **kwargs):
        """
        Actualiza parámetros del plugin.

        :param kwargs: pares clave-valor para actualizar plugin_params.
        """
        # Store old values that trigger model/data reprocessing if changed
        old_model_file = self.params.get("sequential_model_file")
        old_full_feature_names = self.params.get("full_feature_names_ordered")

        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value
        
        new_model_file = self.params.get("sequential_model_file")
        if new_model_file != old_model_file:
            if new_model_file:
                try:
                    self.sequential_model = load_model(new_model_file, compile=False)
                    self.model = self.sequential_model
                    print(f"GeneratorPlugin: Modelo decoder recargado desde {new_model_file}")
                except Exception as e:
                    raise IOError(f"Error al recargar el modelo secuencial desde {new_model_file}. Detalle del error: {e}")
            else:
                raise ValueError("No se puede establecer 'sequential_model_file' a una ruta vacía durante set_params.")

        if self.params.get("full_feature_names_ordered") != old_full_feature_names or \
           any(key in kwargs for key in ["decoder_output_feature_names", "ohlc_feature_names", 
                                         "ti_feature_names", "date_conditional_feature_names", 
                                         "feeder_conditional_feature_names"]):
            if self.params.get("full_feature_names_ordered"): # Ensure it's not empty before rebuilding
                self.feature_to_idx = {name: i for i, name in enumerate(self.params["full_feature_names_ordered"])}
                self.num_all_features = len(self.params["full_feature_names_ordered"])
                self._validate_feature_name_consistency()
            elif old_full_feature_names: # It became empty, which is an issue
                 raise ValueError("'full_feature_names_ordered' cannot be empty after update.")


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

        El `self.sequential_model` cargado es responsable de la lógica central de generación.
        Este método prepara las entradas para él paso a paso y recopila las salidas.

        :param feeder_outputs_sequence: Lista de diccionarios, uno por paso de tiempo. Cada diccionario de
                                        FeederPlugin.generate contiene 'Z', 'time_encodings',
                                        'current_fundamentals'.
        :param sequence_length_T: El número total de pasos de tiempo a generar.
        :param initial_full_feature_window: Ventana de características completa inicial opcional.
                                            Forma (decoder_input_window_size, num_all_features).
        :return: Secuencia de variables base generadas, forma (1, sequence_length_T, num_base_features).
        """
        decoder_input_window_size = self.params["decoder_input_window_size"]
        min_ohlc_hist_len = self.params["ti_calculation_min_lookback"]
        
        ohlc_indices_in_full = [self.feature_to_idx[name] for name in self.params["ohlc_feature_names"]]

        ohlc_history_for_ti_list = [] 

        current_input_feature_window = np.zeros((decoder_input_window_size, self.num_all_features), dtype=np.float32)
        if initial_full_feature_window is not None:
            if initial_full_feature_window.shape == current_input_feature_window.shape:
                current_input_feature_window = initial_full_feature_window.astype(np.float32).copy()
                start_idx_for_ohlc_hist = max(0, decoder_input_window_size - min_ohlc_hist_len)
                for i in range(start_idx_for_ohlc_hist, decoder_input_window_size):
                    row_ohlc_values = current_input_feature_window[i, ohlc_indices_in_full]
                    ohlc_dict = {name: row_ohlc_values[j] for j, name in enumerate(self.params["ohlc_feature_names"])}
                    ohlc_history_for_ti_list.append(ohlc_dict)
            else:
                raise ValueError(f"Shape mismatch for initial_full_feature_window. Expected {current_input_feature_window.shape}, got {initial_full_feature_window.shape}")
        else:
            print("GeneratorPlugin Warning: No initial_full_feature_window provided. Using zeros.")

        generated_sequence_all_features_list = []

        for t in range(sequence_length_T):
            feeder_step_output = feeder_outputs_sequence[t]
            zt = feeder_step_output["Z"]                           
            
            # conditional_data_t must be prepared by FeederPlugin to match CVAE's input_conditions_t layer
            # It should contain scaled date features (day_of_month, hour_of_day, day_of_week) 
            # and scaled fundamental features (S&P500_Close, vix_close) in the correct order.
            conditional_data_t = feeder_step_output["conditional_data"] # Shape (1, num_total_conditions)
            if conditional_data_t.ndim == 1: conditional_data_t = np.expand_dims(conditional_data_t, axis=0)

            # context_h_t must be prepared by FeederPlugin to match CVAE's input_context_h layer
            context_h_t = feeder_step_output.get("context_h", np.zeros((1,1))) # Default if not provided
            if context_h_t.ndim == 1: context_h_t = np.expand_dims(context_h_t, axis=0)


            decoder_input_x_window_t_expanded = np.expand_dims(current_input_feature_window, axis=0)

            decoder_inputs = {
                self.params["decoder_input_name_latent"]: zt,
                self.params["decoder_input_name_window"]: decoder_input_x_window_t_expanded,
                self.params["decoder_input_name_conditions"]: conditional_data_t,
                self.params["decoder_input_name_context"]: context_h_t
            }
            
            generated_decoder_output_step_t = self.sequential_model.predict(decoder_inputs)
            
            if generated_decoder_output_step_t.ndim == 3 and generated_decoder_output_step_t.shape[1] == 1:
                decoded_features_for_current_tick = generated_decoder_output_step_t[0, 0, :]
            elif generated_decoder_output_step_t.ndim == 2 and generated_decoder_output_step_t.shape[0] == 1:
                decoded_features_for_current_tick = generated_decoder_output_step_t[0, :]
            else:
                raise ValueError(f"Unexpected decoder output shape: {generated_decoder_output_step_t.shape}.")

            current_tick_assembled_features = np.zeros(self.num_all_features, dtype=np.float32)

            for i, name in enumerate(self.params["decoder_output_feature_names"]):
                current_tick_assembled_features[self.feature_to_idx[name]] = decoded_features_for_current_tick[i]
            
            # Fill conditional features from conditional_data_t
            # Assumes conditional_data_t has date_conditionalFeatures then feeder_conditional_features
            cond_input_idx = 0
            for name in self.params["date_conditional_feature_names"]:
                current_tick_assembled_features[self.feature_to_idx[name]] = conditional_data_t[0, cond_input_idx]
                cond_input_idx += 1
            for name in self.params["feeder_conditional_feature_names"]:
                current_tick_assembled_features[self.feature_to_idx[name]] = conditional_data_t[0, cond_input_idx]
                cond_input_idx += 1
            
            current_ohlc_values_for_ti_dict = {
                name: current_tick_assembled_features[self.feature_to_idx[name]]
                for name in self.params["ohlc_feature_names"]
            }
            ohlc_history_for_ti_list.append(current_ohlc_values_for_ti_dict)
            
            if len(ohlc_history_for_ti_list) >= 1:
                ohlc_df_for_ti_calc = pd.DataFrame(ohlc_history_for_ti_list)
                ohlc_df_for_ti_calc.columns = self.params["ohlc_feature_names"]
                
                calculated_tis_series = self._calculate_technical_indicators(ohlc_df_for_ti_calc).iloc[0]
                
                for ti_name in self.params["ti_feature_names"]:
                    current_tick_assembled_features[self.feature_to_idx[ti_name]] = calculated_tis_series[ti_name]
            else:
                for ti_name in self.params["ti_feature_names"]:
                     current_tick_assembled_features[self.feature_to_idx[ti_name]] = np.nan

            current_tick_assembled_features[self.feature_to_idx['DATE_TIME']] = np.float32(t)

            generated_sequence_all_features_list.append(current_tick_assembled_features)

            current_input_feature_window = np.roll(current_input_feature_window, -1, axis=0)
            current_input_feature_window[-1, :] = current_tick_assembled_features

            if len(ohlc_history_for_ti_list) > self.params["ti_calculation_min_lookback"] + 50: # Buffer
                ohlc_history_for_ti_list.pop(0)
        
        final_generated_sequence = np.array(generated_sequence_all_features_list, dtype=np.float32)
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
