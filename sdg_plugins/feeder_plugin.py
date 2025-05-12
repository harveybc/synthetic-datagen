"""
FeederPlugin: Genera vectores latentes para el generador de datos sintéticos.
Puede muestrear de una N(0,1) estándar o utilizando un encoder
para obtener representaciones latentes de datos reales y muestrear de ellas.

Interfaz:
- plugin_params: parámetros configurables por defecto.
- plugin_debug_vars: parámetros que aparecerán en get_debug_info.
- __init__(config): inicializa el plugin con configuración.
- set_params(**kwargs): actualiza parámetros del plugin.
- get_debug_info(): devuelve diccionario con valores de debug.
- add_debug_info(info): añade información de debug a un diccionario.
- generate(n_samples): genera la matriz de códigos latentes Z.
"""

import numpy as np
from typing import Dict, Any

# TensorFlow import will be attempted if 'from_encoder' method is used.

class FeederPlugin:
    plugin_params = {
        "latent_dim": 16,
        "random_seed": None,
        "sampling_method": "standard_normal", # Options: "standard_normal", "from_encoder"
        "encoder_model_file": None,       # Path to the Keras encoder model file
        "real_data_file": None            # Path to the .npy file containing real data (X_real) for the encoder
    }
    plugin_debug_vars = ["latent_dim", "random_seed", "sampling_method", "encoder_model_file", "real_data_file"]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("La configuración ('config') es requerida.")
        self.params = self.plugin_params.copy()
        self.set_params(**config) # Apply initial config

        if self.params.get("random_seed") is not None:
            np.random.seed(self.params["random_seed"])

        self._encoder = None
        self._empirical_latents = None # To store Z_real = encoder(X_real)

        # Initial load if method is 'from_encoder' and necessary files are provided in config
        if self.params.get("sampling_method") == "from_encoder":
            if self.params.get("encoder_model_file") and self.params.get("real_data_file"):
                try:
                    self._load_and_process_for_encoder_mode()
                except Exception as e:
                    print(f"FeederPlugin: Warning - Failed to initialize 'from_encoder' mode during __init__: {e}")
            else:
                print("FeederPlugin: Warning - 'sampling_method' is 'from_encoder', but "
                      "'encoder_model_file' or 'real_data_file' may be missing in initial config.")

    def _load_and_process_for_encoder_mode(self):
        """Loads the encoder model and generates latent codes from real_data_file."""
        encoder_path = self.params.get("encoder_model_file")
        data_path = self.params.get("real_data_file")

        if not encoder_path or not data_path:
            # This state means we can't operate in 'from_encoder' mode.
            # Clear any previous state and raise error if called with intent to load.
            self._encoder = None
            self._empirical_latents = None
            if self.params.get("sampling_method") == "from_encoder": # Explicit check
                 raise ValueError("FeederPlugin: For 'from_encoder' method, 'encoder_model_file' and 'real_data_file' must be set.")
            return

        try:
            import tensorflow as tf
            self._encoder = tf.keras.models.load_model(encoder_path, compile=False)
            # print(f"FeederPlugin: Encoder model loaded from '{encoder_path}'")
        except ImportError:
            raise ImportError("FeederPlugin: TensorFlow is required for 'from_encoder' sampling method. Please install it.")
        except Exception as e:
            self._encoder = None # Ensure encoder is None on failure
            raise RuntimeError(f"FeederPlugin: Failed to load encoder model from '{encoder_path}'. Error: {e}")

        try:
            real_data = np.load(data_path)
            # print(f"FeederPlugin: Real data loaded from '{data_path}', shape: {real_data.shape}")
        except Exception as e:
            self._empirical_latents = None # Ensure latents are None on failure
            raise RuntimeError(f"FeederPlugin: Failed to load real data from '{data_path}'. Error: {e}")

        try:
            # print("FeederPlugin: Encoding real data to get latent representations...")
            encoded_output = self._encoder.predict(real_data)
            
            if isinstance(encoded_output, list):
                if not encoded_output: # Empty list
                    raise ValueError("FeederPlugin: Encoder predicted an empty list.")
                # Heuristic: For VAEs [z_mean, z_log_var, z] or [z_mean, z_log_var],
                # assume the first element is the primary latent vector (e.g., z_mean).
                # This might need to be configurable if encoder outputs vary.
                self._empirical_latents = encoded_output[0]
                # print(f"FeederPlugin: Encoder output was a list, using first element. Shape: {self._empirical_latents.shape}")
            else:
                self._empirical_latents = encoded_output
                # print(f"FeederPlugin: Encoder output shape: {self._empirical_latents.shape}")

            if self._empirical_latents.shape[1] != self.params.get("latent_dim"):
                current_latent_dim = self._empirical_latents.shape[1]
                self._empirical_latents = None # Invalidate state
                raise ValueError(
                    f"FeederPlugin: Encoder output latent dimension ({current_latent_dim}) "
                    f"does not match configured 'latent_dim' ({self.params.get('latent_dim')})."
                )
        except Exception as e:
            self._empirical_latents = None # Ensure latents are None on failure
            raise RuntimeError(f"FeederPlugin: Error during encoding real data. Error: {e}")

    def set_params(self, **kwargs):
        old_method = self.params.get("sampling_method")
        old_encoder_file = self.params.get("encoder_model_file")
        old_data_file = self.params.get("real_data_file")

        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value
        
        new_method = self.params.get("sampling_method")
        new_encoder_file = self.params.get("encoder_model_file")
        new_data_file = self.params.get("real_data_file")

        # If critical parameters for 'from_encoder' mode changed, attempt to reload/reprocess.
        if new_method == "from_encoder":
            if (old_method != new_method or 
                old_encoder_file != new_encoder_file or 
                old_data_file != new_data_file):
                if new_encoder_file and new_data_file: # Only if new paths are valid
                    try:
                        # print("FeederPlugin: Relevant parameters changed, re-initializing for 'from_encoder' mode.")
                        self._load_and_process_for_encoder_mode()
                    except Exception as e:
                        print(f"FeederPlugin: Error during re-initialization for 'from_encoder' mode in set_params: {e}")
                        # Invalidate state if loading failed
                        self._encoder = None
                        self._empirical_latents = None
                else: # Files are not (or no longer) set, invalidate previous state
                    self._encoder = None
                    self._empirical_latents = None
        elif old_method == "from_encoder" and new_method != "from_encoder": # Switched away from encoder mode
            self._encoder = None
            self._empirical_latents = None

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    def generate(self, n_samples: int) -> np.ndarray:
        latent_dim = self.params.get("latent_dim")
        method = self.params.get("sampling_method")

        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError(
                "FeederPlugin: 'latent_dim' must be a positive integer."
            )

        if method == "from_encoder":
            if self._empirical_latents is None:
                # Attempt to load if not already loaded (e.g., if params set after __init__ without full info)
                if self.params.get("encoder_model_file") and self.params.get("real_data_file"):
                    print("FeederPlugin: Empirical latents not loaded, attempting to load now in generate().")
                    try:
                        self._load_and_process_for_encoder_mode()
                    except Exception as e:
                         raise RuntimeError(f"FeederPlugin: Failed to load/process for 'from_encoder' mode in generate(): {e}")
                
                if self._empirical_latents is None: # Check again after attempt
                    raise RuntimeError(
                        "FeederPlugin: 'from_encoder' method selected, but empirical latent samples "
                        "could not be loaded. Ensure 'encoder_model_file' and 'real_data_file' are correctly set and processed."
                    )
            
            num_available_empirical = self._empirical_latents.shape[0]
            if n_samples > num_available_empirical:
                print(f"FeederPlugin: Warning - Requesting {n_samples} samples, but only {num_available_empirical} "
                      "unique empirical latent codes available from encoder. Sampling with replacement.")
            
            # Sample with replacement from the empirically obtained Z vectors
            indices = np.random.choice(num_available_empirical, size=n_samples, replace=True)
            Z = self._empirical_latents[indices]
        
        elif method == "standard_normal":
            # Sample from standard normal distribution
            Z = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
        
        else:
            raise ValueError(f"FeederPlugin: Unknown sampling_method '{method}'. Supported: 'standard_normal', 'from_encoder'.")
            
        return Z
