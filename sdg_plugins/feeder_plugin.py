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
from typing import Dict, Any, List
from scipy.stats import gaussian_kde, norm, rankdata, spearmanr # For KDE sampling and Copula

# TensorFlow import will be attempted if 'from_encoder' method is used.

class FeederPlugin:
    plugin_params = {
        "latent_dim": 16,
        "random_seed": None,
        "sampling_method": "standard_normal", # Options: "standard_normal", "from_encoder"
        "encoder_sampling_technique": "direct", # Options for "from_encoder": "direct", "kde", "copula"
        "encoder_model_file": None,       # Path to the Keras encoder model file
        "real_data_file": None,           # Path to the .npy file containing real data (X_real) for the encoder
        "copula_kde_bw_method": None      # Bandwidth method for marginal KDEs in copula. Examples: 'scott' (default), 'silverman', or a scalar float.
    }
    plugin_debug_vars = ["latent_dim", "random_seed", "sampling_method", "encoder_sampling_technique", "encoder_model_file", "real_data_file", "copula_kde_bw_method"]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("La configuración ('config') es requerida.")
        self.params = self.plugin_params.copy()
        self.set_params(**config) # Apply initial config

        if self.params.get("random_seed") is not None:
            np.random.seed(self.params["random_seed"])

        self._encoder = None
        self._empirical_z_means = None 
        self._empirical_z_log_vars = None
        self._joint_kde = None # For 'kde' technique
        self._marginal_kdes: List[gaussian_kde] = [] # For 'copula' technique
        self._copula_spearman_corr = None # For 'copula' technique

        if self.params.get("sampling_method") == "from_encoder":
            if self.params.get("encoder_model_file") and self.params.get("real_data_file"):
                try:
                    self._load_and_process_for_encoder_mode()
                except Exception as e: # This is near your selected line 50
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
        self._marginal_kdes = []
        self._copula_spearman_corr = None

    def _find_nearest_psd(self, matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        """Finds the nearest positive semi-definite matrix to a given matrix."""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square.")
        
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2.0
        
        try:
            # Check if already PSD
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # print("FeederPlugin: Info (Copula) - Matrix not PSD, attempting to find nearest PSD.")
            pass

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip non-positive eigenvalues
        # Ensure eigenvalues don't go below a small positive threshold to maintain some variance
        # and avoid issues with near-zero eigenvalues making the matrix singular.
        clamped_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct the matrix
        psd_matrix = eigenvectors @ np.diag(clamped_eigenvalues) @ eigenvectors.T
        
        # Ensure symmetry again due to potential floating point issues
        psd_matrix = (psd_matrix + psd_matrix.T) / 2.0
        
        # Verify (optional, can be slow for many calls)
        # try:
        #     np.linalg.cholesky(psd_matrix)
        # except np.linalg.LinAlgError:
        #     print("FeederPlugin: Warning (Copula) - Nearest PSD correction failed to produce a strictly PSD matrix.")
            # Fallback: add small identity matrix component
            # psd_matrix = psd_matrix + np.eye(matrix.shape[0]) * min_eigenvalue 
            # psd_matrix = (psd_matrix + psd_matrix.T) / 2.0


        return psd_matrix

    def _load_and_process_for_encoder_mode(self):
        """Loads encoder, processes real data, and fits KDE/Copula structures if needed."""
        encoder_path = self.params.get("encoder_model_file")
        data_path = self.params.get("real_data_file")
        technique = self.params.get("encoder_sampling_technique")
        copula_bw_method = self.params.get("copula_kde_bw_method") # Get the new parameter

        # Invalidate previous state before loading new
        self._invalidate_encoder_state()

        if not encoder_path or not data_path:
            if self.params.get("sampling_method") == "from_encoder":
                 raise ValueError("FeederPlugin: For 'from_encoder' method, 'encoder_model_file' and 'real_data_file' must be set.")
            return

        try:
            import tensorflow as tf
            self._encoder = tf.keras.models.load_model(encoder_path, compile=False)
        except ImportError:
            raise ImportError("FeederPlugin: TensorFlow is required for 'from_encoder' sampling method. Please install it.")
        except Exception as e:
            raise RuntimeError(f"FeederPlugin: Failed to load encoder model from '{encoder_path}'. Error: {e}")

        try:
            real_data = np.load(data_path)
        except Exception as e:
            raise RuntimeError(f"FeederPlugin: Failed to load real data from '{data_path}'. Error: {e}")

        try:
            encoded_outputs = self._encoder.predict(real_data)
            if not (isinstance(encoded_outputs, list) and len(encoded_outputs) == 2):
                raise ValueError("FeederPlugin: Encoder output must be a list of two elements [z_mean, z_log_var].")
            
            z_mean_array, z_log_var_array = encoded_outputs

            if z_mean_array.shape[1] != self.params.get("latent_dim"):
                current_latent_dim = z_mean_array.shape[1]
                raise ValueError(
                    f"FeederPlugin: Encoder z_mean latent dim ({current_latent_dim}) "
                    f"mismatches configured 'latent_dim' ({self.params.get('latent_dim')})."
                )
            if z_log_var_array.shape != z_mean_array.shape:
                raise ValueError("FeederPlugin: z_mean and z_log_var shapes mismatch.")

            self._empirical_z_means = z_mean_array
            self._empirical_z_log_vars = z_log_var_array
            
            joint_empirical_data = np.hstack((self._empirical_z_means, self._empirical_z_log_vars))
            num_samples, total_dims = joint_empirical_data.shape

            if num_samples <= 1:
                print("FeederPlugin: Warning - Not enough empirical samples (<2) to fit KDE or Copula. 'direct' sampling will be used if those techniques are selected.")
                return 

            if technique == "kde":
                if num_samples < 2 * total_dims : 
                    print(f"FeederPlugin: Warning (KDE) - Low sample count ({num_samples}) vs dim ({total_dims}). KDE might be unstable.")
                print(f"FeederPlugin: Fitting joint KDE on {num_samples} samples of dimension {total_dims}...")
                self._joint_kde = gaussian_kde(joint_empirical_data.T)
                print("FeederPlugin: Joint KDE fitting complete.")
            
            elif technique == "copula":
                print(f"FeederPlugin: Fitting Gaussian Copula for {num_samples} samples of dimension {total_dims}...")
                self._marginal_kdes = []
                for i in range(total_dims):
                    marginal_data = joint_empirical_data[:, i]
                    if len(np.unique(marginal_data)) < 2 :
                         print(f"FeederPlugin: Warning (Copula) - Marginal dimension {i} is constant. Using a special handler for its PPF.")
                         const_val = marginal_data[0]
                         class ConstantKDE: # Simplified placeholder for constant marginals
                             def __init__(self, val): self.val = val
                             def ppf(self, q): return np.full_like(q, self.val)
                             def resample(self, size): return np.full((1,size), self.val) 
                             def evaluate(self, points): 
                                 return np.where(points == self.val, np.inf, 0)

                         self._marginal_kdes.append(ConstantKDE(const_val))
                    else:
                         # Use the new bw_method parameter for gaussian_kde
                         self._marginal_kdes.append(gaussian_kde(marginal_data, bw_method=copula_bw_method))

                if num_samples > 1 and total_dims > 0:
                    # spearmanr can handle 1D arrays if total_dims is 1 after hstack (e.g. latent_dim=0, means=1col, logvars=0col - unlikely)
                    # but typically total_dims will be > 1.
                    # If total_dims is 1, spearmanr(col, axis=0) might behave unexpectedly or error.
                    # It's safer if total_dims > 1 for matrix output.
                    if total_dims == 1: # e.g. latent_dim=0, and only z_mean with 1 dim, no z_log_var
                        spearman_corr_matrix = np.array([[1.0]])
                    else:
                        spearman_corr_matrix, _ = spearmanr(joint_empirical_data, axis=0, nan_policy='propagate') # 'propagate' is default
                    
                    # Handle NaNs that spearmanr might produce (e.g., if a column was all NaNs, or constant)
                    if np.isnan(spearman_corr_matrix).any():
                        print("FeederPlugin: Warning (Copula) - NaNs in Spearman matrix from spearmanr. Setting NaNs to 0, diagonal to 1.")
                        # Convert NaNs to 0 (assuming they represent zero correlation for problematic columns)
                        spearman_corr_matrix = np.nan_to_num(spearman_corr_matrix, nan=0.0)
                        # Ensure diagonal is 1, as a column's correlation with itself should be 1
                        np.fill_diagonal(spearman_corr_matrix, 1.0)
                    
                    # Ensure the matrix is positive semi-definite for MVN sampling
                    self._copula_spearman_corr = self._find_nearest_psd(spearman_corr_matrix)
                elif total_dims > 0 : # num_samples <= 1 but total_dims > 0
                    self._copula_spearman_corr = np.eye(total_dims)
                else: # total_dims == 0 (e.g. latent_dim = 0 and no z_mean/z_log_var)
                    self._copula_spearman_corr = np.array([[]]) # Empty correlation

                print("FeederPlugin: Gaussian Copula fitting complete.")
            
        except Exception as e:
            self._invalidate_encoder_state()
            raise RuntimeError(f"FeederPlugin: Error during encoding or KDE/Copula fitting. Error: {e}")

    def set_params(self, **kwargs):
        old_method = self.params.get("sampling_method")
        old_encoder_file = self.params.get("encoder_model_file")
        old_data_file = self.params.get("real_data_file")
        old_technique = self.params.get("encoder_sampling_technique")
        old_copula_bw_method = self.params.get("copula_kde_bw_method") # Store old value of the new param

        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value
        
        new_method = self.params.get("sampling_method")
        new_encoder_file = self.params.get("encoder_model_file")
        new_data_file = self.params.get("real_data_file")
        new_technique = self.params.get("encoder_sampling_technique")
        new_copula_bw_method = self.params.get("copula_kde_bw_method") # Get new value of the new param

        # Determine if re-initialization is needed
        reinitialize = False
        if new_method == "from_encoder":
            if (old_method != new_method or 
                old_encoder_file != new_encoder_file or 
                old_data_file != new_data_file):
                reinitialize = True
            elif new_technique != old_technique: # Technique changed
                reinitialize = True
            # If technique is copula and bw_method changed, or if copula structures are missing
            elif new_technique == "copula" and \
                 (old_copula_bw_method != new_copula_bw_method or \
                  self._copula_spearman_corr is None or \
                  not self._marginal_kdes):
                reinitialize = True
            elif new_technique == "kde" and self._joint_kde is None: # KDE selected but not fitted
                reinitialize = True
        
        if reinitialize:
            if new_encoder_file and new_data_file: 
                try:
                    self._load_and_process_for_encoder_mode()
                except Exception as e:
                    print(f"FeederPlugin: Error during re-initialization for 'from_encoder' mode in set_params: {e}")
                    self._invalidate_encoder_state() # Ensure clean state on error
            else: 
                self._invalidate_encoder_state() # Not enough info to initialize
        elif old_method == "from_encoder" and new_method != "from_encoder": 
            self._invalidate_encoder_state() # Switched away from encoder mode

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    def generate(self, n_samples: int) -> np.ndarray:
        latent_dim = self.params.get("latent_dim")
        method = self.params.get("sampling_method")
        technique = self.params.get("encoder_sampling_technique")

        if not isinstance(latent_dim, int) or latent_dim <= 0:
            # Allow latent_dim = 0 if z_mean/z_log_var are still produced with some fixed dimension by encoder
            # However, the problem context implies latent_dim > 0.
            # For safety, let's assume latent_dim must be positive for meaningful Z.
             if self.params.get("sampling_method") != "standard_normal" and latent_dim == 0 and \
               (self._empirical_z_means is not None and self._empirical_z_means.shape[1] > 0):
                 # If encoder produces fixed-dim z_mean/z_log_var even if config latent_dim is 0
                 # This is an edge case, usually latent_dim in config should match encoder output.
                 pass # Allow if empirical data defines dimensions
             elif latent_dim <=0:
                  raise ValueError("FeederPlugin: 'latent_dim' must be a positive integer.")


        if method == "from_encoder":
            # ... (loading logic as before) ...
            # Ensure data is loaded if needed (this logic can be simplified a bit)
            is_copula_ready = technique == "copula" and self._copula_spearman_corr is not None and self._marginal_kdes
            is_kde_ready = technique == "kde" and self._joint_kde is not None
            is_direct_ready = self._empirical_z_means is not None # Direct only needs empirical data

            needs_load_processing = False
            if not is_direct_ready: # If basic empirical data isn't there, all techniques fail
                needs_load_processing = True
            elif technique == "copula" and not is_copula_ready:
                needs_load_processing = True
            elif technique == "kde" and not is_kde_ready:
                needs_load_processing = True
            
            if needs_load_processing:
                if self.params.get("encoder_model_file") and self.params.get("real_data_file"):
                    print("FeederPlugin: Empirical data/structures not loaded or incomplete, attempting to load/process now in generate().")
                    try:
                        self._load_and_process_for_encoder_mode()
                        # After loading, re-check readiness
                        is_copula_ready = technique == "copula" and self._copula_spearman_corr is not None and self._marginal_kdes
                        is_kde_ready = technique == "kde" and self._joint_kde is not None
                        is_direct_ready = self._empirical_z_means is not None
                    except Exception as e:
                         raise RuntimeError(f"FeederPlugin: Failed to load/process for 'from_encoder' mode in generate(): {e}")
                
                if not is_direct_ready: # Still no empirical data after attempt
                    raise RuntimeError("FeederPlugin: 'from_encoder' selected, but empirical z_mean/z_log_var could not be loaded.")
                if technique == "copula" and not is_copula_ready:
                     print("FeederPlugin: Warning (generate) - Copula technique selected but Copula structures not fitted. Falling back to 'direct' sampling.")
                if technique == "kde" and not is_kde_ready:
                     print("FeederPlugin: Warning (generate) - KDE technique selected but KDE not fitted. Falling back to 'direct' sampling.")

            sampled_z_means = None
            sampled_z_log_vars = None
            
            # Determine actual latent_dim from empirical data if available and config latent_dim might be misleading
            current_latent_dim = self._empirical_z_means.shape[1] if self._empirical_z_means is not None else latent_dim
            if current_latent_dim == 0 and latent_dim > 0 : # Mismatch, prefer configured if positive
                current_latent_dim = latent_dim
            elif current_latent_dim == 0 and latent_dim == 0:
                 # This case means no latent variables to sample, Z would be empty or problematic.
                 # The VAE should likely not produce 0-dim latent variables if it's meant to generate.
                 # For now, let's assume current_latent_dim will be > 0 if we reach sampling.
                 if n_samples > 0: # only raise if we actually need to generate samples
                    raise ValueError("FeederPlugin: Latent dimension is zero, cannot generate samples.")
                 else: # n_samples is 0
                    return np.empty((0,0))


            total_dims_for_sampling = 2 * current_latent_dim # Based on z_mean and z_log_var

            if technique == "copula" and is_copula_ready:
                print(f"FeederPlugin: Sampling {n_samples} (z_mean, z_log_var) pairs using Gaussian Copula...")
                try:
                    # Ensure mean vector matches dimensions of correlation matrix
                    mean_vec = np.zeros(self._copula_spearman_corr.shape[0])
                    mvn_samples = np.random.multivariate_normal(mean_vec, self._copula_spearman_corr, size=n_samples)
                except np.linalg.LinAlgError:
                    # This should ideally be less frequent if _find_nearest_psd works well.
                    print(f"FeederPlugin: Warning (Copula Sampling) - LinAlgError with Spearman correlation matrix. "
                          "This might indicate issues even after PSD correction. Attempting fallback regularization.")
                    reg_corr = self._copula_spearman_corr + np.eye(self._copula_spearman_corr.shape[0]) * 1e-6
                    reg_corr = (reg_corr + reg_corr.T) / 2.0 # Ensure symmetry
                    try:
                        mvn_samples = np.random.multivariate_normal(mean_vec, reg_corr, size=n_samples)
                    except Exception as e_reg:
                        raise RuntimeError(f"FeederPlugin: Error (Copula Sampling) - Failed to sample from MVN even after fallback regularization: {e_reg}")
                
                uniform_samples = norm.cdf(mvn_samples)
                original_scale_samples = np.empty_like(uniform_samples)
                for i in range(total_dims_for_sampling): # Iterate up to actual dims of copula_spearman_corr
                    original_scale_samples[:, i] = self._marginal_kdes[i].ppf(uniform_samples[:, i])
                
                sampled_z_means = original_scale_samples[:, :current_latent_dim]
                sampled_z_log_vars = original_scale_samples[:, current_latent_dim:total_dims_for_sampling] # Ensure correct slicing
                print("FeederPlugin: Gaussian Copula sampling complete.")

            elif technique == "kde" and is_kde_ready:
                # ... (KDE sampling as before, ensure dimensions match current_latent_dim) ...
                print(f"FeederPlugin: Sampling {n_samples} (z_mean, z_log_var) pairs using joint KDE...")
                kde_samples_transposed = self._joint_kde.resample(size=n_samples)
                kde_samples = kde_samples_transposed.T
                sampled_z_means = kde_samples[:, :current_latent_dim]
                sampled_z_log_vars = kde_samples[:, current_latent_dim:total_dims_for_sampling]
                print("FeederPlugin: Joint KDE sampling complete.")
            
            else: # Default to 'direct' or fallback
                # ... (direct sampling as before, ensure dimensions match current_latent_dim) ...
                if technique in ["kde", "copula"] and is_direct_ready: 
                    print(f"FeederPlugin: Warning - Technique '{technique}' selected but structures not fully available/ready. Falling back to 'direct' sampling.")
                elif not is_direct_ready: # Should have been caught by earlier checks
                     raise RuntimeError("FeederPlugin: Cannot perform 'direct' sampling as empirical data is unavailable.")

                num_available_empirical = self._empirical_z_means.shape[0]
                # ... (rest of direct sampling logic) ...
                indices = np.random.choice(num_available_empirical, size=n_samples, replace=True)
                sampled_z_means = self._empirical_z_means[indices]
                sampled_z_log_vars = self._empirical_z_log_vars[indices]


            if current_latent_dim > 0 :
                epsilon = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, current_latent_dim))
                Z = sampled_z_means + np.exp(0.5 * sampled_z_log_vars) * epsilon
            elif n_samples > 0: # current_latent_dim is 0 but trying to generate samples
                # This case should ideally be prevented by VAE design or earlier checks.
                # If z_mean/z_log_var are empty, Z should also be conceptually empty or 0-dim.
                print("FeederPlugin: Warning - current_latent_dim is 0. Generating empty Z array for non-zero n_samples.")
                Z = np.empty((n_samples, 0)) # Or handle as error
            else: # n_samples is 0
                Z = np.empty((0,0))

        elif method == "standard_normal":
            if latent_dim <= 0:
                 raise ValueError("FeederPlugin: 'latent_dim' must be positive for 'standard_normal' sampling.")
            Z = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
        
        else:
            raise ValueError(f"FeederPlugin: Unknown sampling_method '{method}'.")
            
        return Z
