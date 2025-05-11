"""
FeederPlugin: Genera vectores latentes para el generador de datos sintéticos.

Interfaz:
- plugin_params: parámetros configurables por defecto.
- plugin_debug_vars: parámetros que aparecerán en get_debug_info.
- __init__(config): inicializa el plugin con configuración.
- set_params(**kwargs): actualiza parámetros del plugin.
- get_debug_info(): devuelve diccionario con valores de debug.
- add_debug_info(info): añade información de debug a un diccionario.
- generate(config): genera la matriz de códigos latentes Z.
"""

import numpy as np
from typing import Dict, Any


class FeederPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "latent_dim": 16,    # Dimensión de cada vector latente
        "n_samples": 1000,   # Número de vectores latentes a generar
        "random_seed": None  # Semilla aleatoria para reproducibilidad
    }
    # Variables incluidas en el debug
    plugin_debug_vars = ["latent_dim", "n_samples", "random_seed"]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el FeederPlugin con la configuración proporcionada.

        :param config: Diccionario con parámetros que actualizarán plugin_params.
        :raises ValueError: Si config es None.
        """
        if config is None:
            raise ValueError("La configuración ('config') es requerida.")
        # Copia parámetros por defecto y aplica la configuración
        self.params = self.plugin_params.copy()
        self.set_params(**config)

    def set_params(self, **kwargs):
        """
        Actualiza los parámetros del plugin.

        :param kwargs: Pares clave-valor de parámetros a actualizar.
        """
        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario con los valores de las variables de debug.

        :return: Diccionario {var: valor} para cada var en plugin_debug_vars.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        """
        Añade la información de debug al diccionario proporcionado.

        :param debug_info: Diccionario donde se agregará la información de debug.
        """
        debug_info.update(self.get_debug_info())

    def generate(self, config: Dict[str, Any] = None) -> np.ndarray:
        """
        Genera vectores latentes aleatorios desde N(0,1).

        :param config: Opcional, diccionario que puede sobreescribir los parámetros:
                       'latent_dim', 'n_samples', 'random_seed'.
        :return: Matriz Z de forma (n_samples, latent_dim).
        :raises ValueError: Si los parámetros no están definidos o no son válidos.
        """
        # Sobreescribir parámetros si están en config
        if config:
            self.set_params(**config)

        latent_dim = self.params.get("latent_dim")
        n_samples = self.params.get("n_samples")
        random_seed = self.params.get("random_seed")

        # Validación de parámetros
        if latent_dim is None or n_samples is None:
            raise ValueError("'latent_dim' y 'n_samples' deben estar definidos en params.")

        # Inicializar semilla para reproducibilidad si se proporciona
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generación de la matriz Z ~ N(0,1)
        Z = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(n_samples, latent_dim)
        )

        return Z

