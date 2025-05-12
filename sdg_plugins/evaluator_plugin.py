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
from statsmodels.tsa.stattools import acf as sm_acf # Import statsmodels acf directly

class EvaluatorPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "real_data_file": None,
        "window_size": 288
    }
    # Variables incluidas en el debug
    plugin_debug_vars = ["real_data_file", "window_size"]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        self.params = self.plugin_params.copy()
        self.set_params(**config)

        # real_file validation was removed as real_data_processed is now primary
        # However, if you still need real_data_file for other potential logic,
        # ensure it's handled or validated appropriately if self.params["real_data_file"] is used.
        # For now, assuming it's not strictly needed by this evaluate method anymore.

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.plugin_params: # Only update known params
                self.params[key] = value
            # else:
            #     print(f"WARN: Unknown parameter '{key}' passed to EvaluatorPlugin.set_params.")


    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    def evaluate(
        self,
        synthetic_data: np.ndarray,
        real_data_processed: np.ndarray,
        real_dates: Optional[pd.Index],
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if config:
            self.set_params(**config)

        real_signal = real_data_processed

        if real_signal.ndim != 2:
            raise ValueError(f"real_data_processed must be 2D (samples, features), got ndim={real_signal.ndim}")

        if synthetic_data.ndim != 2 or synthetic_data.shape != real_signal.shape:
            raise ValueError(
                f"synthetic_data shape {synthetic_data.shape} != expected real_data_processed shape {real_signal.shape}"
            )
        
        n_features = real_signal.shape[1]
        if feature_names and len(feature_names) != n_features: # Check if feature_names is provided
            print(f"WARN: Length of feature_names ({len(feature_names)}) does not match number of features in data ({n_features}). Metrics might be misaligned if names are used or feature-specific metrics might lack names.")
        elif not feature_names:
            print(f"WARN: feature_names not provided to evaluator. Feature-specific metrics will not have names.")


        mae_per = np.mean(np.abs(synthetic_data - real_signal), axis=0)
        mse_per = np.mean((synthetic_data - real_signal) ** 2, axis=0)
        
        pearson_r_per = []
        acf1_per = []

        for f_idx in range(n_features):
            y_true_feat = real_signal[:, f_idx]
            y_syn_feat = synthetic_data[:, f_idx]
            current_feature_name = feature_names[f_idx] if feature_names and f_idx < len(feature_names) else f"Feature_{f_idx}"

            # Pearson r
            r = np.nan
            if len(y_true_feat) > 1 and len(y_syn_feat) > 1:
                # Check for constant series which result in std_dev = 0
                if np.std(y_true_feat) > 1e-9 and np.std(y_syn_feat) > 1e-9: # Add tolerance for near-constant
                    # Using pandas for potentially more robust correlation calculation
                    s_true = pd.Series(y_true_feat)
                    s_syn = pd.Series(y_syn_feat)
                    r = s_true.corr(s_syn, method='pearson')
                else:
                    # If one or both series are constant (or near-constant)
                    if np.allclose(y_true_feat, y_true_feat[0]) and np.allclose(y_syn_feat, y_syn_feat[0]) and np.allclose(y_true_feat[0], y_syn_feat[0]):
                        r = 1.0 # Both constant and equal
                    elif np.allclose(y_true_feat, y_true_feat[0]) and np.allclose(y_syn_feat, y_syn_feat[0]):
                        r = np.nan # Both constant but different, correlation is undefined or NaN
                    else:
                        r = 0.0 # One is constant, other is not, or other edge cases. Pearson is typically 0 or undefined.
            pearson_r_per.append(r)


            # ACF lag 1 for the real signal feature
            acf1 = np.nan
            if len(y_true_feat) > 1:
                try:
                    # Using statsmodels.tsa.stattools.acf directly
                    # nlags includes lag 0, so for lag 1 we need nlags=1. Result will have 2 elements.
                    computed_acf_values = sm_acf(y_true_feat, nlags=1, fft=False, adjusted=False) # fft=False for direct method
                    if len(computed_acf_values) > 1:
                        acf1 = computed_acf_values[1] 
                    else:
                        print(f"WARN: Feature '{current_feature_name}': sm_acf returned fewer than 2 values. Length of y_true_feat: {len(y_true_feat)}")
                except Exception as e_acf:
                    print(f"ERROR: Feature '{current_feature_name}': Exception during ACF calculation using sm_acf. Error: {e_acf}")
            acf1_per.append(acf1)

        pearson_r_per_np = np.array(pearson_r_per)
        acf1_per_np = np.array(acf1_per)

        # Convert all numpy types to Python native types for JSON serialization
        metrics = {
            "mae_per_feature": mae_per.tolist(),
            "mse_per_feature": mse_per.tolist(),
            "pearson_r_per_feature": [float(x) if not np.isnan(x) else None for x in pearson_r_per_np], # Convert NaN to None for JSON
            "acf1_per_feature": [float(x) if not np.isnan(x) else None for x in acf1_per_np],       # Convert NaN to None for JSON
            "overall_mae": float(np.mean(mae_per)),
            "overall_mse": float(np.mean(mse_per)),
            "overall_pearson_r": float(np.nanmean(pearson_r_per_np)) if not np.all(np.isnan(pearson_r_per_np)) else None,
            "overall_acf1": float(np.nanmean(acf1_per_np)) if not np.all(np.isnan(acf1_per_np)) else None
        }
        if feature_names and len(feature_names) == n_features:
            metrics["feature_names"] = feature_names

        return metrics
