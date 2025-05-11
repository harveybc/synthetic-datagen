"""
GeneratorPlugin: Transforma vectores latentes en datos sintéticos per-tick (primer elemento de la ventana).

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
from typing import Dict, Any
from tensorflow.keras.models import load_model, Model


class GeneratorPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "decoder_file": None,  # Ruta al modelo decoder preentrenado
        "batch_size": 32       # Batch size para inferencia
    }
    # Variables incluidas en el debug
    plugin_debug_vars = ["decoder_file", "batch_size"]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el GeneratorPlugin y carga el modelo decoder.

        :param config: Diccionario con al menos 'decoder_file' y opcional 'batch_size'.
        :raises ValueError: si no se proporciona 'decoder_file'.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        # Copia parámetros por defecto y aplica la configuración
        self.params = self.plugin_params.copy()
        self.set_params(**config)

        decoder_path = self.params.get("decoder_file")
        if not decoder_path:
            raise ValueError("El parámetro 'decoder_file' debe especificar la ruta al modelo decoder.")

        # Carga del modelo decoder preentrenado sin compilar
        self.decoder: Model = load_model(decoder_path, compile=False)

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

    def generate(self, Z: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
        """
        Genera datos sintéticos por tick a partir de vectores latentes.

        Flujo:
          1. Sobreescribe parámetros si se pasan en config.
          2. Inferencia con el decoder: Z -> ventanas sintéticas.
          3. Extrae el primer elemento de cada ventana (por feature).
          4. Devuelve matriz de forma (n_samples, n_features).

        :param Z: matriz de vectores latentes (n_samples, latent_dim).
        :param config: (opcional) puede incluir 'batch_size'.
        :return: Datos sintéticos por tick (n_samples, n_features).
        :raises ValueError: si la forma de salida no es la esperada.
        """
        # Sobreescribir batch_size si se especifica en config
        if config:
            self.set_params(**config)

        batch_size = self.params.get("batch_size")
        if batch_size is None:
            raise ValueError("'batch_size' debe estar definido en params.")

        # Inferencia: Z -> ventanas [n_samples, window_size, n_features]
        windows = self.decoder.predict(Z, batch_size=batch_size)

        # Validación de dimensiones: se espera un tensor 3D
        if windows.ndim != 3:
            raise ValueError(f"Salida del decoder debe ser 3D, se obtuvo ndim={windows.ndim}.")

        # Extraer primer elemento temporal de cada ventana
        # Resultado: [n_samples, n_features]
        synthetic = windows[:, 0, :]

        return synthetic
