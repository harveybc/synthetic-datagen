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
from scipy.stats import gaussian_kde, norm, rankdata # For KDE sampling and Copula

# TensorFlow import will be attempted if 'from_encoder' method is used.

class FeederPlugin:
    plugin_params = {
        "latent_dim": 16,
        "random_seed": None,
        "sampling_method": "standard_normal", # Options: "standard_normal", "from_encoder"
        "encoder_sampling_technique": "direct", # Options for "from_encoder": "direct", "kde", "copula"
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

    def _load_and_process_for_encoder_mode(self):
        """Loads encoder, processes real data, and fits KDE/Copula structures if needed."""
        encoder_path = self.params.get("encoder_model_file")
        data_path = self.params.get("real_data_file")
        technique = self.params.get("encoder_sampling_technique")

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
                print("FeederPlugin: Warning - Not enough empirical samples (<2) to fit KDE or Copula. 'direct' sampling will be used.")
                return # Fallback to direct sampling if technique was kde/copula

            if technique == "kde":
                if num_samples < 2 * total_dims : # Heuristic
                    print(f"FeederPlugin: Warning (KDE) - Low sample count ({num_samples}) vs dim ({total_dims}). KDE might be unstable.")
                print(f"FeederPlugin: Fitting joint KDE on {num_samples} samples of dimension {total_dims}...")
                self._joint_kde = gaussian_kde(joint_empirical_data.T)
                print("FeederPlugin: Joint KDE fitting complete.")
            
            elif technique == "copula":
                print(f"FeederPlugin: Fitting Gaussian Copula for {num_samples} samples of dimension {total_dims}...")
                # 1. Fit marginal KDEs
                self._marginal_kdes = []
                for i in range(total_dims):
                    marginal_data = joint_empirical_data[:, i]
                    if len(np.unique(marginal_data)) < 2 : # KDE fails if data is constant
                         # Fallback for constant marginal: use a tiny normal around the constant
                         print(f"FeederPlugin: Warning (Copula) - Marginal dimension {i} is constant or near-constant. Using a narrow normal for this marginal.")
                         # This is a tricky edge case. A proper solution might involve more sophisticated marginal modeling.
                         # For now, a simple normal. This might not be ideal.
                         # A gaussian_kde of a constant value will error.
                         # We need a callable ppf.
                         const_val = marginal_data[0]
                         # Create a dummy KDE-like object with a ppf for this constant case
                         class ConstantKDE:
                             def __init__(self, val):
                                 self.val = val
                             def ppf(self, q):
                                 return np.full_like(q, self.val) # Return constant for any quantile
                         self._marginal_kdes.append(ConstantKDE(const_val))
                    else:
                         self._marginal_kdes.append(gaussian_kde(marginal_data))

                # 2. Calculate Spearman's rank correlation matrix
                # Convert to ranks first, then compute Pearson correlation on ranks
                rank_data = np.empty_like(joint_empirical_data)
                for i in range(total_dims):
                    rank_data[:, i] = rankdata(joint_empirical_data[:, i], method='average')
                
                # Handle cases where a column in rank_data might be constant (if original was constant)
                # which would lead to NaNs in correlation.
                # If a column is constant, its correlation with others should be treated as 0.
                # A more robust way for spearman:
                if num_samples > 1:
                    # scipy.stats.spearmanr handles NaNs better if all values in a column are same
                    # but let's compute pearson on ranks for clarity with the Gaussian copula concept
                    # For Pearson on ranks, ensure no constant columns in rank_data after ranking,
                    # or handle resulting NaNs in correlation matrix.
                    # A simpler approach for spearman_corr:
                    # temp_df = pd.DataFrame(joint_empirical_data)
                    # self._copula_spearman_corr = temp_df.corr(method='spearman').values
                    # Using numpy for pearson on ranks:
                    # Check for constant columns in rank_data (std_dev near zero)
                    rank_std = np.std(rank_data, axis=0)
                    # Create a correlation matrix of ranks (Pearson on ranks)
                    # Initialize with identity to handle constant columns correctly (corr=0 with others, 1 with self)
                    spearman_corr_matrix = np.eye(total_dims)
                    # Compute for non-constant pairs
                    valid_cols = rank_std > 1e-9 # Columns that are not constant
                    if np.sum(valid_cols) > 1: # Need at least two varying columns
                        valid_rank_data = rank_data[:, valid_cols]
                        partial_corr = np.corrcoef(valid_rank_data, rowvar=False)
                        # Fill into the full spearman_corr_matrix
                        # Create an indexing mesh
                        idx = np.ix_(valid_cols, valid_cols)
                        spearman_corr_matrix[idx] = partial_corr
                    
                    # Ensure positive semi-definiteness (e.g., by eigenvalue clipping or finding nearest)
                    # For simplicity, we'll assume it's good enough for now, or use a robust library if issues arise.
                    # A common quick fix if not PSD: add small epsilon to diagonal or use `np.cov(..., bias=True)` like approach
                    # For now, let's proceed with the computed matrix.
                    self._copula_spearman_corr = spearman_corr_matrix

                else: # num_samples <=1
                    self._copula_spearman_corr = np.eye(total_dims) # Default to identity if not enough data

                print("FeederPlugin: Gaussian Copula fitting complete.")
            
        except Exception as e:
            # Invalidate everything if any step fails
            self._invalidate_encoder_state()
            raise RuntimeError(f"FeederPlugin: Error during encoding or KDE/Copula fitting. Error: {e}")

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

        # Determine if re-initialization is needed
        reinitialize = False
        if new_method == "from_encoder":
            if (old_method != new_method or 
                old_encoder_file != new_encoder_file or 
                old_data_file != new_data_file):
                reinitialize = True
            elif new_technique != old_technique: # Technique changed
                reinitialize = True
            elif new_technique == "kde" and self._joint_kde is None: # KDE selected but not fitted
                reinitialize = True
            elif new_technique == "copula" and (self._copula_spearman_corr is None or not self._marginal_kdes): # Copula selected but not fitted
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
            raise ValueError("FeederPlugin: 'latent_dim' must be a positive integer.")

        if method == "from_encoder":
            # Ensure data is loaded if needed
            needs_load = self._empirical_z_means is None or self._empirical_z_log_vars is None
            if technique == "kde" and self._joint_kde is None and needs_load: # Check specific structures too
                needs_load = True
            if technique == "copula" and (self._copula_spearman_corr is None or not self._marginal_kdes) and needs_load:
                needs_load = True
            
            if needs_load:
                if self.params.get("encoder_model_file") and self.params.get("real_data_file"):
                    print("FeederPlugin: Empirical data/structures not loaded, attempting to load now in generate().")
                    try:
                        self._load_and_process_for_encoder_mode()
                    except Exception as e:
                         raise RuntimeError(f"FeederPlugin: Failed to load/process for 'from_encoder' mode in generate(): {e}")
                # Re-check after attempting load
                if self._empirical_z_means is None or self._empirical_z_log_vars is None:
                    raise RuntimeError("FeederPlugin: 'from_encoder' selected, but empirical z_mean/z_log_var could not be loaded.")
                if technique == "kde" and self._joint_kde is None:
                     print("FeederPlugin: Warning (generate) - KDE technique selected but KDE not fitted. Falling back to 'direct' sampling.")
                if technique == "copula" and (self._copula_spearman_corr is None or not self._marginal_kdes):
                     print("FeederPlugin: Warning (generate) - Copula technique selected but Copula structures not fitted. Falling back to 'direct' sampling.")


            sampled_z_means = None
            sampled_z_log_vars = None
            total_dims_for_sampling = 2 * latent_dim

            if technique == "copula" and self._copula_spearman_corr is not None and self._marginal_kdes:
                print(f"FeederPlugin: Sampling {n_samples} (z_mean, z_log_var) pairs using Gaussian Copula...")
                # 1. Sample from MVN using Spearman correlation of ranks
                # Ensure covariance matrix is positive semi-definite for MVN sampling
                # A simple way to attempt to make it PSD is to add a small value to the diagonal
                # or find the nearest PSD matrix. For now, we assume it's usable.
                # If self._copula_spearman_corr might not be PSD, handle np.linalg.LinAlgError
                try:
                    mvn_samples = np.random.multivariate_normal(np.zeros(total_dims_for_sampling), self._copula_spearman_corr, size=n_samples)
                except np.linalg.LinAlgError as e:
                    print(f"FeederPlugin: Warning (Copula Sampling) - LinAlgError ({e}) with Spearman correlation matrix. It might not be positive semi-definite. Adding small epsilon to diagonal and retrying.")
                    # Attempt to regularize the correlation matrix
                    epsilon_diag = 1e-6 
                    regularized_corr = self._copula_spearman_corr + np.eye(total_dims_for_sampling) * epsilon_diag
                    try:
                        mvn_samples = np.random.multivariate_normal(np.zeros(total_dims_for_sampling), regularized_corr, size=n_samples)
                    except np.linalg.LinAlgError as e2:
                         raise RuntimeError(f"FeederPlugin: Error (Copula Sampling) - Failed to sample from MVN even after regularization: {e2}. Copula structure might be problematic.")


                # 2. Transform to uniform marginals using Normal CDF
                uniform_samples = norm.cdf(mvn_samples)
                
                # 3. Transform to original scales using inverse CDF (ppf) of marginal KDEs
                original_scale_samples = np.empty_like(uniform_samples)
                for i in range(total_dims_for_sampling):
                    original_scale_samples[:, i] = self._marginal_kdes[i].ppf(uniform_samples[:, i])
                
                sampled_z_means = original_scale_samples[:, :latent_dim]
                sampled_z_log_vars = original_scale_samples[:, latent_dim:]
                print("FeederPlugin: Gaussian Copula sampling complete.")

            elif technique == "kde" and self._joint_kde is not None:
                print(f"FeederPlugin: Sampling {n_samples} (z_mean, z_log_var) pairs using joint KDE...")
                kde_samples_transposed = self._joint_kde.resample(size=n_samples)
                kde_samples = kde_samples_transposed.T
                sampled_z_means = kde_samples[:, :latent_dim]
                sampled_z_log_vars = kde_samples[:, latent_dim:]
                print("FeederPlugin: Joint KDE sampling complete.")
            
            else: # Default to 'direct' or fallback
                if technique in ["kde", "copula"]: # Print fallback message only if specific technique failed
                    print(f"FeederPlugin: Warning - Technique '{technique}' selected but structures not available. Falling back to 'direct' sampling.")
                
                num_available_empirical = self._empirical_z_means.shape[0]
                if n_samples > num_available_empirical and num_available_empirical > 0 : # only warn if we have some data
                    print(f"FeederPlugin: Warning (direct sampling) - Requesting {n_samples} samples, but only {num_available_empirical} "
                          "empirical pairs available. Sampling with replacement.")
                
                if num_available_empirical == 0:
                    raise RuntimeError("FeederPlugin: 'direct' sampling chosen, but no empirical data available.")

                indices = np.random.choice(num_available_empirical, size=n_samples, replace=True)
                sampled_z_means = self._empirical_z_means[indices]
                sampled_z_log_vars = self._empirical_z_log_vars[indices]

            epsilon = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
            Z = sampled_z_means + np.exp(0.5 * sampled_z_log_vars) * epsilon
        
        elif method == "standard_normal":
            Z = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
        
        else:
            raise ValueError(f"FeederPlugin: Unknown sampling_method '{method}'. Supported: 'standard_normal', 'from_encoder'.")
            
        return Z
