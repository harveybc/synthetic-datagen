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
from typing import Dict, Any, List, Optional # Ensure Optional is imported
from scipy.stats import gaussian_kde, norm, rankdata, spearmanr # For KDE sampling and Copula
import pandas as pd # Added for datetime operations

# TensorFlow import will be attempted if 'from_encoder' method is used.

class FeederPlugin:
    plugin_params = {
        "latent_dim": 16,
        "random_seed": None,
        "sampling_method": "standard_normal", # Options: "standard_normal", "from_encoder"
        "encoder_sampling_technique": "direct", # Options for "from_encoder": "direct", "kde", "copula"
        "encoder_model_file": None,       # Path to the Keras encoder model file
        "real_data_file": None,           # Path to the .csv file containing real data (X_real) for the encoder, plus datetimes and fundamentals
        "feature_columns_for_encoder": [], # List of column names from real_data_file to be used as input to the VAE encoder
        "real_data_file_has_header": True, # Whether the real_data_file (CSV) has a header row
        "datetime_col_in_real_data": "DATE_TIME",  # Name of the datetime column in real_data_file
        "date_feature_names_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"], # Date features to generate
        "fundamental_feature_names_for_conditioning": ["S&P500_Close", "vix_close"], # Fundamental features to use from real_data_file
        "max_day_of_month": 31,
        "max_hour_of_day": 23, # Hours 0-23
        "max_day_of_week": 6,  # Days 0-6
        "context_vector_dim": 16, # Dimension of the context_h vector for the decoder
        "context_vector_strategy": "zeros", # "zeros" or future "sample_from_real_context"
        "copula_kde_bw_method": None,
    }
    plugin_debug_vars = [
        "latent_dim", "random_seed", "sampling_method", 
        "encoder_sampling_technique", "encoder_model_file", "real_data_file", 
        "feature_columns_for_encoder", "datetime_col_in_real_data",
        "date_feature_names_for_conditioning", "fundamental_feature_names_for_conditioning",
        "context_vector_dim", "context_vector_strategy",
        "copula_kde_bw_method"
    ]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("La configuración ('config') es requerida.")
        self.params = self.plugin_params.copy()
        self.set_params(**config) 

        if self.params.get("random_seed") is not None:
            np.random.seed(self.params["random_seed"])

        self._encoder = None
        self._empirical_z_means = None 
        self._empirical_z_log_vars = None
        self._joint_kde = None 
        self._marginal_kdes: List[gaussian_kde] = [] 
        self._copula_spearman_corr = None 
        
        self._real_datetimes_pd_series: Optional[pd.Series] = None
        self._real_fundamental_features_df_scaled: Optional[pd.DataFrame] = None
        # Add placeholder for real context vectors if that strategy is implemented
        # self._real_context_vectors_h = None 

        if self.params.get("sampling_method") == "from_encoder" or \
           len(self.params.get("fundamental_feature_names_for_conditioning", [])) > 0:
            # Load real data if using encoder OR if fundamental conditioning is needed,
            # as fundamentals are sourced from real_data_file.
            if self.params.get("real_data_file"):
                try:
                    self._load_and_process_for_encoder_mode() # This method now also loads fundamentals
                except Exception as e: 
                    print(f"FeederPlugin: Warning - Failed to initialize data during __init__: {e}")
            else:
                print("FeederPlugin: Warning - 'real_data_file' may be missing in initial config, needed for encoder or fundamental conditioning.")

    def _invalidate_encoder_state(self):
        """Resets encoder-related and real_data_related attributes."""
        self._encoder = None
        self._empirical_z_means = None
        self._empirical_z_log_vars = None
        self._joint_kde = None
        self._marginal_kdes = []
        self._copula_spearman_corr = None
        self._real_datetimes_pd_series = None
        self._real_fundamental_features_df_scaled = None
        # self._real_context_vectors_h = None

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
        """
        Loads encoder (if sampling_method='from_encoder'), processes real data CSV to extract
        datetimes, fundamental features (assumed scaled), and features for the encoder.
        Fits KDE/Copula structures if needed for Z sampling.
        """
        encoder_path = self.params.get("encoder_model_file")
        data_path = self.params.get("real_data_file")
        technique = self.params.get("encoder_sampling_technique")
        copula_bw_method = self.params.get("copula_kde_bw_method")
        datetime_col_name = self.params.get("datetime_col_in_real_data")
        fundamental_cols = self.params.get("fundamental_feature_names_for_conditioning", [])
        encoder_feature_cols = self.params.get("feature_columns_for_encoder", [])
        has_header = self.params.get("real_data_file_has_header", True)

        self._invalidate_encoder_state() # Clear previous real data and encoder state

        if not data_path:
            if self.params.get("sampling_method") == "from_encoder" or len(fundamental_cols) > 0:
                 raise ValueError("FeederPlugin: 'real_data_file' must be set for 'from_encoder' method or if fundamental features are conditioned.")
            return # No data to load if no path and no need for fundamentals/encoder

        try:
            print(f"FeederPlugin: Loading real data from CSV: {data_path}")
            # Load the entire CSV into a pandas DataFrame
            df_real_data_full = pd.read_csv(data_path, header=0 if has_header else None)
            if not has_header and encoder_feature_cols and isinstance(encoder_feature_cols[0], str):
                raise ValueError("FeederPlugin: 'feature_columns_for_encoder' provided as names, but 'real_data_file_has_header' is false.")
            if not has_header and fundamental_cols and isinstance(fundamental_cols[0], str):
                 raise ValueError("FeederPlugin: 'fundamental_feature_names_for_conditioning' provided as names, but 'real_data_file_has_header' is false.")
            if not has_header and isinstance(datetime_col_name, str):
                 raise ValueError("FeederPlugin: 'datetime_col_in_real_data' provided as name, but 'real_data_file_has_header' is false.")


            # Extract and store datetimes
            if datetime_col_name in df_real_data_full.columns:
                self._real_datetimes_pd_series = pd.to_datetime(df_real_data_full[datetime_col_name])
            elif isinstance(datetime_col_name, int) and datetime_col_name < len(df_real_data_full.columns):
                self._real_datetimes_pd_series = pd.to_datetime(df_real_data_full.iloc[:, datetime_col_name])
            else:
                raise ValueError(f"FeederPlugin: Datetime column '{datetime_col_name}' not found in real_data_file.")

            # Extract and store (assumed scaled) fundamental features
            if fundamental_cols:
                try:
                    self._real_fundamental_features_df_scaled = df_real_data_full[fundamental_cols].astype(np.float32)
                except KeyError as e:
                    raise ValueError(f"FeederPlugin: One or more fundamental columns not found in real_data_file: {e}")
            else:
                self._real_fundamental_features_df_scaled = pd.DataFrame() # Empty DF if no fundamentals specified

        except Exception as e:
            raise RuntimeError(f"FeederPlugin: Failed to load or process real data CSV from '{data_path}'. Error: {e}")

        # Proceed with encoder loading and Z sampling structure fitting ONLY if method is 'from_encoder'
        if self.params.get("sampling_method") == "from_encoder":
            if not encoder_path:
                raise ValueError("FeederPlugin: 'sampling_method' is 'from_encoder', but 'encoder_model_file' is not set.")
            if not encoder_feature_cols:
                raise ValueError("FeederPlugin: 'sampling_method' is 'from_encoder', but 'feature_columns_for_encoder' is not set.")

            try:
                import tensorflow as tf
                self._encoder = tf.keras.models.load_model(encoder_path, compile=False)
            except ImportError:
                raise ImportError("FeederPlugin: TensorFlow is required for 'from_encoder' sampling method. Please install it.")
            except Exception as e:
                raise RuntimeError(f"FeederPlugin: Failed to load encoder model from '{encoder_path}'. Error: {e}")

            try:
                # Select only the specified columns for the encoder input
                real_data_for_encoder = df_real_data_full[encoder_feature_cols].astype(np.float32).values
                if real_data_for_encoder.ndim == 1: # If only one feature column selected
                    real_data_for_encoder = real_data_for_encoder.reshape(-1,1)

                print(f"FeederPlugin: Predicting with encoder using data of shape {real_data_for_encoder.shape}")
                encoded_outputs = self._encoder.predict(real_data_for_encoder)
                if not (isinstance(encoded_outputs, list) and len(encoded_outputs) >= 2): # Encoder might output more than z_mean, z_log_var
                    raise ValueError("FeederPlugin: Encoder output must be a list of at least two elements [z_mean, z_log_var, ...].")
                
                z_mean_array, z_log_var_array = encoded_outputs[0], encoded_outputs[1]

                # Store also context vectors if encoder provides them and strategy is to sample them
                # if self.params.get("context_vector_strategy") == "sample_from_real_context" and len(encoded_outputs) >= 4:
                #    self._real_context_vectors_h_means = encoded_outputs[2]
                #    self._real_context_vectors_h_log_vars = encoded_outputs[3]


                if z_mean_array.shape[1] != self.params.get("latent_dim"):
                    current_latent_dim_from_encoder = z_mean_array.shape[1]
                    # Optionally, update self.params["latent_dim"] or just use the encoder's dim internally
                    print(f"FeederPlugin: Warning - Encoder z_mean latent dim ({current_latent_dim_from_encoder}) "
                          f"mismatches configured 'latent_dim' ({self.params.get('latent_dim')}). Using encoder's dimension for Z sampling.")
                    # self.params["latent_dim"] = current_latent_dim_from_encoder # Or handle this mismatch as error
                
                if z_log_var_array.shape != z_mean_array.shape:
                    raise ValueError("FeederPlugin: z_mean and z_log_var shapes mismatch from encoder.")

                self._empirical_z_means = z_mean_array
                self._empirical_z_log_vars = z_log_var_array
                
                joint_empirical_data = np.hstack((self._empirical_z_means, self._empirical_z_log_vars))
                num_samples, total_dims = joint_empirical_data.shape

                if num_samples <= 1:
                    print("FeederPlugin: Warning - Not enough empirical samples (<2) from encoder to fit KDE or Copula. 'direct' sampling will be used if those techniques are selected.")
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
                             class ConstantKDE: 
                                 def __init__(self, val): self.val = val
                                 def ppf(self, q): return np.full_like(q, self.val)
                                 def resample(self, size): return np.full((1,size), self.val) 
                                 def evaluate(self, points): 
                                     return np.where(points == self.val, np.inf, 0)
                             self._marginal_kdes.append(ConstantKDE(const_val))
                        else:
                             self._marginal_kdes.append(gaussian_kde(marginal_data, bw_method=copula_bw_method))

                    if num_samples > 1 and total_dims > 0:
                        if total_dims == 1:
                            spearman_corr_matrix = np.array([[1.0]])
                        else:
                            spearman_corr_matrix, _ = spearmanr(joint_empirical_data, axis=0, nan_policy='propagate')
                        
                        if np.isnan(spearman_corr_matrix).any():
                            print("FeederPlugin: Warning (Copula) - NaNs in Spearman matrix. Setting NaNs to 0, diagonal to 1.")
                            spearman_corr_matrix = np.nan_to_num(spearman_corr_matrix, nan=0.0)
                            np.fill_diagonal(spearman_corr_matrix, 1.0)
                        
                        self._copula_spearman_corr = self._find_nearest_psd(spearman_corr_matrix)
                    elif total_dims > 0 : 
                        self._copula_spearman_corr = np.eye(total_dims)
                    else: 
                        self._copula_spearman_corr = np.array([[]]) 

                    print("FeederPlugin: Gaussian Copula fitting complete.")
                
            except Exception as e:
                self._invalidate_encoder_state() # Clear partially loaded state
                raise RuntimeError(f"FeederPlugin: Error during encoder processing or KDE/Copula fitting. Error: {e}")

    def set_params(self, **kwargs):
        old_method = self.params.get("sampling_method")
        old_encoder_file = self.params.get("encoder_model_file")
        old_data_file = self.params.get("real_data_file")
        old_technique = self.params.get("encoder_sampling_technique")
        old_copula_bw_method = self.params.get("copula_kde_bw_method") 
        old_encoder_features = self.params.get("feature_columns_for_encoder")
        old_fundamental_features = self.params.get("fundamental_feature_names_for_conditioning")
        old_datetime_col = self.params.get("datetime_col_in_real_data")


        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value
        
        new_method = self.params.get("sampling_method")
        new_encoder_file = self.params.get("encoder_model_file")
        new_data_file = self.params.get("real_data_file")
        new_technique = self.params.get("encoder_sampling_technique")
        new_copula_bw_method = self.params.get("copula_kde_bw_method") 
        new_encoder_features = self.params.get("feature_columns_for_encoder")
        new_fundamental_features = self.params.get("fundamental_feature_names_for_conditioning")
        new_datetime_col = self.params.get("datetime_col_in_real_data")


        reinitialize_encoder_related = False
        if new_method == "from_encoder":
            if (old_method != new_method or 
                old_encoder_file != new_encoder_file or 
                old_data_file != new_data_file or # Data file change always triggers reload
                old_encoder_features != new_encoder_features or # Encoder feature selection change
                old_technique != new_technique or
                (new_technique == "copula" and old_copula_bw_method != new_copula_bw_method)
                ):
                reinitialize_encoder_related = True
            # If technique is copula and structures are missing
            elif new_technique == "copula" and (self._copula_spearman_corr is None or not self._marginal_kdes):
                reinitialize_encoder_related = True
            elif new_technique == "kde" and self._joint_kde is None:
                reinitialize_encoder_related = True
        elif old_method == "from_encoder" and new_method != "from_encoder": # Switched away
            self._invalidate_encoder_state() # Clear encoder specific, but fundamentals might still be needed

        # Re-load fundamentals if data file, fundamental cols, or datetime col changed,
        # OR if encoder reinitialization is happening (as it reloads the file)
        # OR if fundamentals were not loaded but are now needed.
        reinitialize_fundamentals = False
        if (old_data_file != new_data_file or
            old_fundamental_features != new_fundamental_features or
            old_datetime_col != new_datetime_col or
            (len(new_fundamental_features) > 0 and self._real_fundamental_features_df_scaled is None) or
            (len(new_fundamental_features) > 0 and self._real_datetimes_pd_series is None)
            ):
            reinitialize_fundamentals = True
        
        if reinitialize_encoder_related or reinitialize_fundamentals:
            if self.params.get("real_data_file"): 
                try:
                    print("FeederPlugin: Re-initializing data due to parameter change.")
                    self._load_and_process_for_encoder_mode()
                except Exception as e:
                    print(f"FeederPlugin: Error during re-initialization in set_params: {e}")
                    self._invalidate_encoder_state() 
            else: 
                self._invalidate_encoder_state() # Not enough info to initialize

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    def _get_scaled_date_features(self, datetime_obj: pd.Timestamp) -> np.ndarray:
        """Generates scaled (sin/cos) date features for a given datetime."""
        date_features = []
        if "day_of_month" in self.params["date_feature_names_for_conditioning"]:
            dom = datetime_obj.day
            date_features.append(np.sin(2 * np.pi * dom / self.params["max_day_of_month"]))
            date_features.append(np.cos(2 * np.pi * dom / self.params["max_day_of_month"]))
        if "hour_of_day" in self.params["date_feature_names_for_conditioning"]:
            hod = datetime_obj.hour
            date_features.append(np.sin(2 * np.pi * hod / (self.params["max_hour_of_day"] + 1))) # Max is 23, so 24 distinct values
            date_features.append(np.cos(2 * np.pi * hod / (self.params["max_hour_of_day"] + 1)))
        if "day_of_week" in self.params["date_feature_names_for_conditioning"]:
            dow = datetime_obj.dayofweek # Monday=0, Sunday=6
            date_features.append(np.sin(2 * np.pi * dow / (self.params["max_day_of_week"] + 1))) # Max is 6, so 7 distinct values
            date_features.append(np.cos(2 * np.pi * dow / (self.params["max_day_of_week"] + 1)))
        return np.array(date_features, dtype=np.float32)

    def _get_scaled_fundamental_features(self, datetime_obj: pd.Timestamp) -> np.ndarray:
        """
        Retrieves scaled fundamental features for a given datetime.
        Assumes self._real_datetimes_pd_series and self._real_fundamental_features_df_scaled are populated.
        Uses forward-fill if exact datetime match is not found.
        """
        if self._real_datetimes_pd_series is None or self._real_fundamental_features_df_scaled is None or \
           self._real_fundamental_features_df_scaled.empty:
            if len(self.params["fundamental_feature_names_for_conditioning"]) > 0:
                # print(f"FeederPlugin: Warning - Real data for fundamentals not loaded. Returning NaNs for fundamentals for {datetime_obj}.")
                pass # Avoid printing for every tick
            return np.full(len(self.params["fundamental_feature_names_for_conditioning"]), np.nan, dtype=np.float32)

        # Find the index in _real_datetimes_pd_series that is closest to or before datetime_obj
        # `searchsorted` can find insertion point; use 'right' to get index of first element > datetime_obj
        # then subtract 1 to get element <= datetime_obj (for forward fill)
        idx_pos = self._real_datetimes_pd_series.searchsorted(datetime_obj, side='right')
        
        if idx_pos == 0: # datetime_obj is before the first known real datetime
            # print(f"FeederPlugin: Warning - Requested datetime {datetime_obj} is before any known fundamental data. Using first available.")
            # Use the first available fundamental data point or NaNs if preferred
            # For now, using first available. Could also return NaNs.
            # return self._real_fundamental_features_df_scaled.iloc[0].values
            return np.full(len(self.params["fundamental_feature_names_for_conditioning"]), np.nan, dtype=np.float32)


        # idx_pos is the insertion point, so index idx_pos-1 is the latest known data at or before datetime_obj
        actual_idx = idx_pos - 1
        
        # Check if the found datetime is reasonably close (e.g., within a day or a few hours)
        # This is optional, to prevent very old data from being used if there's a large gap.
        # time_diff = datetime_obj - self._real_datetimes_pd_series.iloc[actual_idx]
        # if time_diff > pd.Timedelta(days=1): # Example threshold
        #     print(f"FeederPlugin: Warning - Stale fundamental data for {datetime_obj} (older than 1 day).")

        return self._real_fundamental_features_df_scaled.iloc[actual_idx].values.astype(np.float32)


    def generate(self, n_ticks_to_generate: int, target_datetimes: pd.Series) -> List[Dict[str, np.ndarray]]:
        if len(target_datetimes) != n_ticks_to_generate:
            raise ValueError("Length of target_datetimes must match n_ticks_to_generate.")

        latent_dim_config = self.params.get("latent_dim")
        method = self.params.get("sampling_method")
        technique = self.params.get("encoder_sampling_technique")
        context_dim = self.params.get("context_vector_dim")
        context_strategy = self.params.get("context_vector_strategy")

        # Determine actual latent_dim for Z sampling
        if method == "from_encoder" and self._empirical_z_means is not None:
            current_latent_dim_for_z = self._empirical_z_means.shape[1]
            if current_latent_dim_for_z == 0 and latent_dim_config > 0: # Encoder produced 0-dim, but config expects >0
                 print(f"FeederPlugin: Warning - Encoder produced 0-dim latent space, but config latent_dim is {latent_dim_config}. Using config dim for Z if standard_normal, or erroring for encoder methods if Z is truly 0-dim.")
                 current_latent_dim_for_z = latent_dim_config # Fallback to config for standard_normal
            elif current_latent_dim_for_z != latent_dim_config and latent_dim_config > 0:
                 print(f"FeederPlugin: Info - Encoder latent dim ({current_latent_dim_for_z}) differs from config ({latent_dim_config}). Using encoder's dim for Z sampling.")
        else: # standard_normal or encoder data not loaded
            current_latent_dim_for_z = latent_dim_config
        
        if current_latent_dim_for_z <= 0 and n_ticks_to_generate > 0:
            raise ValueError(f"FeederPlugin: Effective latent dimension for Z is {current_latent_dim_for_z}, must be positive to generate samples.")

        # --- Z Sampling (similar to existing logic, but for n_ticks_to_generate) ---
        Z_samples = np.empty((n_ticks_to_generate, current_latent_dim_for_z), dtype=np.float32)

        if method == "from_encoder":
            is_copula_ready = technique == "copula" and self._copula_spearman_corr is not None and self._marginal_kdes
            is_kde_ready = technique == "kde" and self._joint_kde is not None
            is_direct_ready = self._empirical_z_means is not None

            if not is_direct_ready:
                raise RuntimeError("FeederPlugin: 'from_encoder' selected, but empirical z_mean/z_log_var could not be loaded/processed.")
            if technique == "copula" and not is_copula_ready:
                 print("FeederPlugin: Warning (generate) - Copula technique selected but structures not fitted. Falling back to 'direct' sampling.")
            if technique == "kde" and not is_kde_ready:
                 print("FeederPlugin: Warning (generate) - KDE technique selected but KDE not fitted. Falling back to 'direct' sampling.")

            sampled_z_means_batch = None
            sampled_z_log_vars_batch = None
            total_dims_for_joint_sampling = 2 * current_latent_dim_for_z

            if technique == "copula" and is_copula_ready:
                mean_vec = np.zeros(self._copula_spearman_corr.shape[0])
                try:
                    mvn_samples = np.random.multivariate_normal(mean_vec, self._copula_spearman_corr, size=n_ticks_to_generate)
                except np.linalg.LinAlgError:
                    reg_corr = self._copula_spearman_corr + np.eye(self._copula_spearman_corr.shape[0]) * 1e-6
                    reg_corr = (reg_corr + reg_corr.T) / 2.0
                    mvn_samples = np.random.multivariate_normal(mean_vec, reg_corr, size=n_ticks_to_generate)
                
                uniform_samples = norm.cdf(mvn_samples)
                original_scale_samples = np.empty_like(uniform_samples)
                for i in range(total_dims_for_joint_sampling):
                    original_scale_samples[:, i] = self._marginal_kdes[i].ppf(uniform_samples[:, i])
                
                sampled_z_means_batch = original_scale_samples[:, :current_latent_dim_for_z]
                sampled_z_log_vars_batch = original_scale_samples[:, current_latent_dim_for_z:total_dims_for_joint_sampling]
            
            elif technique == "kde" and is_kde_ready:
                kde_samples_transposed = self._joint_kde.resample(size=n_ticks_to_generate)
                kde_samples = kde_samples_transposed.T
                sampled_z_means_batch = kde_samples[:, :current_latent_dim_for_z]
                sampled_z_log_vars_batch = kde_samples[:, current_latent_dim_for_z:total_dims_for_joint_sampling]
            
            else: # Default to 'direct' or fallback
                num_available_empirical = self._empirical_z_means.shape[0]
                indices = np.random.choice(num_available_empirical, size=n_ticks_to_generate, replace=True)
                sampled_z_means_batch = self._empirical_z_means[indices]
                sampled_z_log_vars_batch = self._empirical_z_log_vars[indices]

            epsilon_batch = np.random.normal(loc=0.0, scale=1.0, size=(n_ticks_to_generate, current_latent_dim_for_z))
            Z_samples = sampled_z_means_batch + np.exp(0.5 * sampled_z_log_vars_batch) * epsilon_batch
        
        elif method == "standard_normal":
            Z_samples = np.random.normal(loc=0.0, scale=1.0, size=(n_ticks_to_generate, current_latent_dim_for_z))
        else:
            raise ValueError(f"FeederPlugin: Unknown sampling_method '{method}'.")

        # --- Prepare output list of dictionaries ---
        output_sequence = []
        for i in range(n_ticks_to_generate):
            datetime_obj = target_datetimes.iloc[i]

            # Get Z for the current tick
            z_tick = Z_samples[i, :].reshape(1, -1) # Ensure shape (1, latent_dim)

            # Get scaled date features
            scaled_date_features_tick = self._get_scaled_date_features(datetime_obj) # Shape (num_date_features*2,)

            # Get scaled fundamental features
            scaled_fundamental_features_tick = self._get_scaled_fundamental_features(datetime_obj) # Shape (num_fundamental_features,)
            
            # Concatenate to form conditional_data for the decoder
            # Order: date features, then fundamental features
            conditional_data_for_tick = np.concatenate(
                [scaled_date_features_tick, scaled_fundamental_features_tick]
            ).reshape(1, -1) # Shape (1, num_date_feats*2 + num_fund_feats)

            # Get context_h vector
            context_h_tick = np.zeros((1, context_dim), dtype=np.float32) # Default "zeros" strategy
            if context_strategy == "zeros":
                pass # Already zeros
            # elif context_strategy == "sample_from_real_context":
                # Implement sampling if self._real_context_vectors_h is populated

            output_sequence.append({
                "Z": z_tick.astype(np.float32),
                "conditional_data": conditional_data_for_tick.astype(np.float32),
                "context_h": context_h_tick.astype(np.float32),
                "datetimes": datetime_obj # Store the original pandas Timestamp
            })
            
        return output_sequence
