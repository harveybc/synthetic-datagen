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
from scipy.stats import gaussian_kde # For KDE sampling

# TensorFlow import will be attempted if 'from_encoder' method is used.

class FeederPlugin:
    plugin_params = {
        "latent_dim": 16,
        "random_seed": None,
        "sampling_method": "standard_normal", # Options: "standard_normal", "from_encoder"
        "encoder_sampling_technique": "direct", # Options for "from_encoder": "direct", "kde"
        "encoder_model_file": None,       # Path to the Keras encoder model file
        "real_data_file": None            # Path to the .npy file containing real data (X_real) for the encoder
    }
    plugin_debug_vars = ["latent_dim", "random_seed", "sampling_method", "encoder_sampling_technique", "encoder_model_file", "real_data_file"]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("La configuración ('config') es requerida.")
        self.params = self.plugin_params.copy()
        self.set_params(**config) # Apply initial config

        if self.params.get("random_seed") is not None:
            np.random.seed(self.params["random_seed"])

        self._encoder = None
        # For VAE: store z_mean and z_log_var from real data
        self._empirical_z_means = None 
        self._empirical_z_log_vars = None
        self._joint_kde = None # For KDE sampling

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

    def _invalidate_encoder_state(self):
        """Resets encoder-related attributes."""
        self._encoder = None
        self._empirical_z_means = None
        self._empirical_z_log_vars = None
        self._joint_kde = None

    def _load_and_process_for_encoder_mode(self):
        """Loads encoder, processes real data, and optionally fits KDE."""
        encoder_path = self.params.get("encoder_model_file")
        data_path = self.params.get("real_data_file")
        technique = self.params.get("encoder_sampling_technique")

        if not encoder_path or not data_path:
            self._invalidate_encoder_state()
            if self.params.get("sampling_method") == "from_encoder":
                 raise ValueError("FeederPlugin: For 'from_encoder' method, 'encoder_model_file' and 'real_data_file' must be set.")
            return

        try:
            import tensorflow as tf
            # Load the Keras model. For VAEs, this model should output [z_mean, z_log_var].
            self._encoder = tf.keras.models.load_model(encoder_path, compile=False)
            # print(f"FeederPlugin: Encoder model loaded from '{encoder_path}'")
        except ImportError:
            self._invalidate_encoder_state()
            raise ImportError("FeederPlugin: TensorFlow is required for 'from_encoder' sampling method. Please install it.")
        except Exception as e:
            self._invalidate_encoder_state()
            raise RuntimeError(f"FeederPlugin: Failed to load encoder model from '{encoder_path}'. Error: {e}")

        try:
            real_data = np.load(data_path)
            # print(f"FeederPlugin: Real data loaded from '{data_path}', shape: {real_data.shape}")
        except Exception as e:
            self._invalidate_encoder_state()
            raise RuntimeError(f"FeederPlugin: Failed to load real data from '{data_path}'. Error: {e}")

        try:
            # print("FeederPlugin: Encoding real data to get latent representations (z_mean, z_log_var)...")
            # Encoder is expected to return a list: [z_mean_array, z_log_var_array]
            encoded_outputs = self._encoder.predict(real_data)
            
            if not (isinstance(encoded_outputs, list) and len(encoded_outputs) == 2):
                self._invalidate_encoder_state()
                raise ValueError(
                    "FeederPlugin: Encoder output was not a list of two elements (expected [z_mean, z_log_var]). "
                    f"Got type: {type(encoded_outputs)}, length: {len(encoded_outputs) if isinstance(encoded_outputs, list) else 'N/A'}"
                )
            
            z_mean_array, z_log_var_array = encoded_outputs
            # print(f"FeederPlugin: Extracted z_mean (shape: {z_mean_array.shape}) and z_log_var (shape: {z_log_var_array.shape}).")

            if z_mean_array.shape[1] != self.params.get("latent_dim"):
                current_latent_dim = z_mean_array.shape[1]
                self._invalidate_encoder_state()
                raise ValueError(
                    f"FeederPlugin: Encoder output latent dimension for z_mean ({current_latent_dim}) "
                    f"does not match configured 'latent_dim' ({self.params.get('latent_dim')})."
                )
            if z_log_var_array.shape != z_mean_array.shape:
                self._invalidate_encoder_state()
                raise ValueError(
                    f"FeederPlugin: z_mean (shape: {z_mean_array.shape}) and z_log_var (shape: {z_log_var_array.shape}) "
                    "do not have matching shapes."
                )

            self._empirical_z_means = z_mean_array
            self._empirical_z_log_vars = z_log_var_array

            if technique == "kde":
                if self._empirical_z_means.shape[0] < 2 * self._empirical_z_means.shape[1]: # Heuristic for KDE stability
                    print(f"FeederPlugin: Warning - Number of empirical samples ({self._empirical_z_means.shape[0]}) "
                          f"is low relative to latent dimension ({self._empirical_z_means.shape[1]}) for robust KDE. "
                          "Consider using 'direct' sampling or more empirical data.")
                
                # Fit KDE on the joint (z_mean, z_log_var) distribution
                # Transpose because gaussian_kde expects features in rows, samples in columns
                joint_empirical_data = np.hstack((self._empirical_z_means, self._empirical_z_log_vars))
                if joint_empirical_data.shape[0] > 1: # KDE needs more than 1 data point
                    print(f"FeederPlugin: Fitting joint KDE for 'from_encoder_kde' mode on {joint_empirical_data.shape[0]} samples of dimension {joint_empirical_data.shape[1]}...")
                    self._joint_kde = gaussian_kde(joint_empirical_data.T)
                    print("FeederPlugin: Joint KDE fitting complete.")
                else:
                    print("FeederPlugin: Warning - Not enough data points to fit KDE. Falling back to 'direct' sampling logic if KDE is selected.")
                    self._joint_kde = None # Ensure it's None if fitting fails
            
        except Exception as e:
            self._invalidate_encoder_state()
            raise RuntimeError(f"FeederPlugin: Error during encoding real data or processing outputs. Error: {e}")

    def set_params(self, **kwargs):
        old_method = self.params.get("sampling_method")
        old_encoder_file = self.params.get("encoder_model_file")
        old_data_file = self.params.get("real_data_file")
        old_technique = self.params.get("encoder_sampling_technique")

        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value
        
        new_method = self.params.get("sampling_method")
        new_encoder_file = self.params.get("encoder_model_file")
        new_data_file = self.params.get("real_data_file")
        new_technique = self.params.get("encoder_sampling_technique")

        if new_method == "from_encoder":
            if (old_method != new_method or 
                old_encoder_file != new_encoder_file or 
                old_data_file != new_data_file or
                (new_technique == "kde" and old_technique != "kde") or # Re-fit KDE if technique changes to KDE
                (new_technique == "kde" and self._joint_kde is None) # Re-fit KDE if it wasn't successfully fitted before
                ):
                if new_encoder_file and new_data_file: 
                    try:
                        # print("FeederPlugin: Relevant parameters changed, re-initializing for 'from_encoder' mode.")
                        self._load_and_process_for_encoder_mode()
                    except Exception as e:
                        print(f"FeederPlugin: Error during re-initialization for 'from_encoder' mode in set_params: {e}")
                        self._invalidate_encoder_state()
                else: 
                    self._invalidate_encoder_state()
        elif old_method == "from_encoder" and new_method != "from_encoder": 
            self._invalidate_encoder_state()

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    def generate(self, n_samples: int) -> np.ndarray:
        latent_dim = self.params.get("latent_dim")
        method = self.params.get("sampling_method")
        technique = self.params.get("encoder_sampling_technique")

        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError(
                "FeederPlugin: 'latent_dim' must be a positive integer."
            )

        if method == "from_encoder":
            # Check if empirical distributions are loaded
            if self._empirical_z_means is None or self._empirical_z_log_vars is None:
                if self.params.get("encoder_model_file") and self.params.get("real_data_file"):
                    print("FeederPlugin: Empirical z_mean/z_log_var not loaded, attempting to load now in generate().")
                    try:
                        self._load_and_process_for_encoder_mode()
                    except Exception as e:
                         raise RuntimeError(f"FeederPlugin: Failed to load/process for 'from_encoder' mode in generate(): {e}")
                
                if self._empirical_z_means is None or self._empirical_z_log_vars is None: # Check again
                    raise RuntimeError(
                        "FeederPlugin: 'from_encoder' method selected, but empirical z_mean/z_log_var "
                        "could not be loaded. Ensure 'encoder_model_file' and 'real_data_file' are correctly set and processed."
                    )
            
            sampled_z_means = None
            sampled_z_log_vars = None

            if technique == "kde" and self._joint_kde is not None:
                print(f"FeederPlugin: Sampling {n_samples} (z_mean, z_log_var) pairs using joint KDE...")
                # Sample from the KDE. Result is (features_dim x n_samples)
                kde_samples_transposed = self._joint_kde.resample(size=n_samples)
                # Transpose back to (n_samples x features_dim)
                kde_samples = kde_samples_transposed.T
                
                sampled_z_means = kde_samples[:, :latent_dim]
                sampled_z_log_vars = kde_samples[:, latent_dim:]
                print("FeederPlugin: KDE sampling complete.")
            
            else: # Default to 'direct' or fallback if KDE failed
                if technique == "kde" and self._joint_kde is None:
                    print("FeederPlugin: Warning - KDE technique selected but KDE not fitted. Falling back to 'direct' sampling.")
                
                num_available_empirical = self._empirical_z_means.shape[0]
                if n_samples > num_available_empirical:
                    print(f"FeederPlugin: Warning (direct sampling) - Requesting {n_samples} samples, but only {num_available_empirical} "
                          "empirical pairs available. Sampling with replacement.")
                
                indices = np.random.choice(num_available_empirical, size=n_samples, replace=True)
                sampled_z_means = self._empirical_z_means[indices]
                sampled_z_log_vars = self._empirical_z_log_vars[indices]

            epsilon = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
            Z = sampled_z_means + np.exp(0.5 * sampled_z_log_vars) * epsilon
        
        elif method == "standard_normal":
            # Sample from standard normal distribution
            Z = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
        
        else:
            raise ValueError(f"FeederPlugin: Unknown sampling_method '{method}'. Supported: 'standard_normal', 'from_encoder'.")
            
        return Z
