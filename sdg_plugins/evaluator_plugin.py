"""
EvaluatorPlugin: Evalúa la calidad de los datos sintéticos comparándolos
con la señal real, usando métricas estadísticas y de correlación.

Interfaz:
- plugin_params: configuración por defecto.
- plugin_debug_vars: variables expuestas en get_debug_info.
- __init__(config): inicializa parámetros.
- set_params(**kwargs): actualiza parámetros.
- get_debug_info(): devuelve info de debug.
- add_debug_info(info): añade debug info al diccionario.
- evaluate(synthetic_data, config): carga datos reales, crea sliding windows,
  extrae primer elemento por ventana y calcula métricas comparativas.

"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class EvaluatorPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "real_data_file": None,  # Ruta al archivo de datos reales (CSV o NPY)
        "window_size": 288       # Tamaño de la ventana deslizante
    }
    # Variables incluidas en el debug
    plugin_debug_vars = ["real_data_file", "window_size"]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el EvaluatorPlugin y valida parámetros básicos.

        :param config: Diccionario con 'real_data_file' y 'window_size'.
        :raises ValueError: si faltan parámetros obligatorios.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        self.params = self.plugin_params.copy()
        self.set_params(**config)

        real_file = self.params.get("real_data_file")
        if not real_file:
            raise ValueError("El parámetro 'real_data_file' es obligatorio.")
        if not os.path.exists(real_file):
            raise ValueError(f"Archivo de datos reales no encontrado: {real_file}")

    def set_params(self, **kwargs):
        """
        Actualiza parámetros del plugin.

        :param kwargs: pares clave-valor para actualizar plugin_params.
        """
        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario con valores de las variables de debug.

        :return: {var: valor} para cada var en plugin_debug_vars.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        """
        Añade información de debug al diccionario proporcionado.

        :param debug_info: diccionario destino.
        """
        debug_info.update(self.get_debug_info())

    def evaluate(
        self,
        synthetic_data: np.ndarray,
        real_data_processed: np.ndarray,  # Use this directly
        real_dates: Optional[pd.Index],
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compares 'synthetic_data' with 'real_data_processed'.
        Assumes real_data_processed is already in the shape (n_samples, n_features)
        and aligned with synthetic_data.
        """
        if config: # Allow overriding params like window_size if still needed for other logic
            self.set_params(**config)

        # The 'real_data_processed' is now the definitive real signal for comparison.
        # It should already be shaped (n_samples, n_features).
        real_signal = real_data_processed

        if real_signal.ndim != 2:
            raise ValueError(f"real_data_processed must be 2D (samples, features), got ndim={real_signal.ndim}")

        # Validar que synthetic_data coincide en forma
        if synthetic_data.ndim != 2 or synthetic_data.shape != real_signal.shape:
            raise ValueError(
                f"synthetic_data shape {synthetic_data.shape} != expected real_data_processed shape {real_signal.shape}"
            )
        
        n_features = real_signal.shape[1]
        if len(feature_names) != n_features:
            print(f"WARN: Length of feature_names ({len(feature_names)}) does not match number of features in data ({n_features}). Metrics might be misaligned if names are used.")


        # Cálculo de métricas por característica
        mae_per = np.mean(np.abs(synthetic_data - real_signal), axis=0)
        mse_per = np.mean((synthetic_data - real_signal) ** 2, axis=0)
        
        pearson_r_per = []
        acf1_per = [] # For ACF of the real_signal features

        for f_idx in range(n_features):
            y_true_feat = real_signal[:, f_idx]
            y_syn_feat = synthetic_data[:, f_idx]

            # Pearson r
            if len(y_true_feat) > 1 and len(y_syn_feat) > 1: # Need at least 2 points for std
                cov = np.cov(y_true_feat, y_syn_feat, ddof=0)[0, 1] # Use ddof=0 for population covariance if comparing directly
                std_true = np.std(y_true_feat)
                std_syn = np.std(y_syn_feat)
                stds_prod = std_true * std_syn
                r = cov / stds_prod if stds_prod != 0 else np.nan
            else:
                r = np.nan
            pearson_r_per.append(r)

            # ACF lag 1 for the real signal feature
            if len(y_true_feat) > 1:
                # Simplified ACF-1 calculation using pandas for robustness
                acf_series = pd.Series(y_true_feat).acf(nlags=1, fft=False) # fft=False for direct method
                acf1 = acf_series[1] if len(acf_series) > 1 else np.nan
            else:
                acf1 = np.nan
            acf1_per.append(acf1)

        pearson_r_per = np.array(pearson_r_per)
        acf1_per = np.array(acf1_per)

        # Métricas globales agregadas
        metrics = {
            "mae_per_feature": mae_per.tolist(), # Convert to list for JSON serialization
            "mse_per_feature": mse_per.tolist(),
            "pearson_r_per_feature": np.nan_to_num(pearson_r_per).tolist(), # Handle NaNs from constant series
            "acf1_per_feature": np.nan_to_num(acf1_per).tolist(),
            "overall_mae": np.mean(mae_per),
            "overall_mse": np.mean(mse_per),
            "overall_pearson_r": np.nanmean(pearson_r_per), # nanmean ignores NaNs
            "overall_acf1": np.nanmean(acf1_per)
        }
        # Add feature names to metrics if available and lengths match
        if feature_names and len(feature_names) == n_features:
            metrics["feature_names"] = feature_names

        return metrics
