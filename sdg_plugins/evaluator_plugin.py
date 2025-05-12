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
        real_data_processed: np.ndarray,  # New
        real_dates: Optional[pd.Index],   # New
        feature_names: List[str],         # New
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compara 'synthetic_data' con la señal real extraída de 'real_data_file'.
        Realiza sliding windows sobre la señal real de tamaño 'window_size',
        extrae el primer elemento de cada ventana y calcula métricas.

        :param synthetic_data: Array shape (n_samples, n_features).
        :param real_data_processed: Array shape (n_samples, n_features).
        :param real_dates: Opcional, índice de fechas de los datos reales.
        :param feature_names: Lista de nombres de características.
        :param config: Opcional, puede sobreescribir 'real_data_file' o 'window_size'.
        :return: Diccionario con métricas:
                 - 'mae_per_feature'
                 - 'mse_per_feature'
                 - 'pearson_r_per_feature'
                 - 'acf1_per_feature'
                 - 'overall_mae'
                 - 'overall_mse'
                 - 'overall_pearson_r'
                 - 'overall_acf1'
        :raises ValueError: si las formas no coinciden o faltan datos.
        """
        # Sobreescribir parámetros si están en config
        if config:
            self.set_params(**config)

        real_file = self.params.get("real_data_file")
        window_size = self.params.get("window_size")

        # Carga de datos reales
        ext = os.path.splitext(real_file)[1].lower()
        if ext == '.npy':
            real = np.load(real_file)
        else:
            real = pd.read_csv(real_file).values

        # Validar dimensiones
        if real.ndim != 2:
            raise ValueError(f"Datos reales deben ser 2D, ndim={real.ndim}")

        n_ticks, n_features = real.shape
        expected_samples = n_ticks - window_size + 1
        if expected_samples <= 0:
            raise ValueError("window_size es mayor que la longitud de la señal real.")

        # Crear sliding windows manualmente
        windows = np.zeros((expected_samples, window_size, n_features), dtype=real.dtype)
        for i in range(expected_samples):
            windows[i] = real[i:i + window_size]

        # Extraer primer elemento de cada ventana: real_signal shape (n_samples, n_features)
        real_signal = windows[:, 0, :]

        # Validar que synthetic_data coincide en forma
        if synthetic_data.ndim != 2 or synthetic_data.shape != real_signal.shape:
            raise ValueError(
                f"synthetic_data shape {synthetic_data.shape} != expected {real_signal.shape}"
            )

        # Cálculo de métricas por característica
        mae_per = np.mean(np.abs(synthetic_data - real_signal), axis=0)
        mse_per = np.mean((synthetic_data - real_signal) ** 2, axis=0)
        # Pearson r
        pearson_r_per = []
        acf1_per = []
        for f in range(n_features):
            # Pearson r
            y_true = real_signal[:, f]
            y_syn = synthetic_data[:, f]
            cov = np.cov(y_true, y_syn, ddof=0)[0, 1]
            stds = np.std(y_true) * np.std(y_syn)
            r = cov / stds if stds != 0 else np.nan
            pearson_r_per.append(r)
            # ACF lag 1
            y_mean = np.mean(y_true)
            y_lag = y_true[:-1] - y_mean
            y_cur = y_true[1:] - y_mean
            acf1 = np.corrcoef(y_lag, y_cur)[0, 1] if y_lag.size > 0 else np.nan
            acf1_per.append(acf1)

        pearson_r_per = np.array(pearson_r_per)
        acf1_per = np.array(acf1_per)

        # Métricas globales agregadas
        metrics = {
            "mae_per_feature": mae_per,
            "mse_per_feature": mse_per,
            "pearson_r_per_feature": pearson_r_per,
            "acf1_per_feature": acf1_per,
            "overall_mae": np.mean(mae_per),
            "overall_mse": np.mean(mse_per),
            "overall_pearson_r": np.nanmean(pearson_r_per),
            "overall_acf1": np.nanmean(acf1_per)
        }

        return metrics
