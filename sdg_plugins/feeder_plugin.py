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
    plugin_name_prefix = "feeder"
    plugin_params = {
        "latent_shape": [18, 32], # Default shape (sequence_length, features)
        "random_seed": None,
        "sampling_method": "standard_normal", # Options: "standard_normal", "from_encoder"
        "encoder_sampling_technique": "direct", # Options for "from_encoder": "direct", "kde", "copula"
        "encoder_model_file": None,       # Path to the Keras encoder model file
        # "real_data_file": None,           # REMOVED - Will use x_train_file from main_config
        "feature_columns_for_encoder": [], # List of column names from x_train_file to be used as input to the VAE encoder
        "real_data_file_has_header": True, # Whether the x_train_file (CSV) has a header row
        "datetime_col_in_real_data": "DATE_TIME",  # Name of the datetime column in x_train_file
        "date_feature_names_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week", "day_of_year"],
        "fundamental_feature_names_for_conditioning": ["S&P500_Close", "vix_close"], # Fundamental features to use from x_train_file
        "max_day_of_month": 31,
        "max_hour_of_day": 23,
        "max_day_of_week": 6,
        "max_day_of_year": 366, 
        "context_vector_dim": 64, # Default updated to 64
        "context_vector_strategy": "zeros",
        "copula_kde_bw_method": None,
    }
    plugin_debug_vars = [
        "latent_shape", "random_seed", "sampling_method", 
        "encoder_sampling_technique", "encoder_model_file", # "real_data_file" removed
        "feature_columns_for_encoder", "datetime_col_in_real_data",
        "date_feature_names_for_conditioning", "fundamental_feature_names_for_conditioning",
        "max_day_of_month", "max_hour_of_day", "max_day_of_week", "max_day_of_year",
        "context_vector_dim", "context_vector_strategy",
        "copula_kde_bw_method"
    ]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("La configuración ('config') es requerida.")
        self.main_config = config.copy() # Store the main config
        self.params = self.plugin_params.copy()
        
        # Initialize encoder state attributes
        self._encoder = None
        self._empirical_z_means: Optional[np.ndarray] = None 
        self._empirical_z_log_vars: Optional[np.ndarray] = None
        self._joint_kde = None 
        self._marginal_kdes: List[gaussian_kde] = [] 
        self._copula_spearman_corr = None 
        self._real_datetimes_pd_series: Optional[pd.Series] = None
        self._real_fundamental_features_df_scaled: Optional[pd.DataFrame] = None

        self.set_params(**config) 

        if self.params.get("random_seed") is not None:
            np.random.seed(self.params["random_seed"])

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
        Loads encoder, processes real data (from x_train_file), and fits structures for Z sampling.
        Handles 3D latent spaces from the encoder.
        """
        encoder_path = self.params.get("encoder_model_file")
        # data_path = self.params.get("real_data_file") # REMOVED
        data_path = self.main_config.get("x_train_file") # Use x_train_file from main config
        
        technique = self.params.get("encoder_sampling_technique")
        copula_bw_method = self.params.get("copula_kde_bw_method")
        datetime_col_name = self.params.get("datetime_col_in_real_data")
        fundamental_cols = self.params.get("fundamental_feature_names_for_conditioning", [])
        encoder_feature_cols = self.params.get("feature_columns_for_encoder", [])
        has_header = self.params.get("real_data_file_has_header", True)

        self._invalidate_encoder_state() # Clear previous real data and encoder state

        if not data_path:
            if self.params.get("sampling_method") == "from_encoder" or len(fundamental_cols) > 0:
                 raise ValueError("FeederPlugin: 'x_train_file' (via main_config) must be set for 'from_encoder' method or if fundamental features are conditioned.")
            return # No data to load if no path and no need for fundamentals/encoder

        try:
            print(f"FeederPlugin: Loading real data from CSV (via x_train_file): {data_path}")
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
                real_data_for_encoder = df_real_data_full[encoder_feature_cols].astype(np.float32).values
                if real_data_for_encoder.ndim == 1: 
                    real_data_for_encoder = real_data_for_encoder.reshape(-1,1)

                print(f"FeederPlugin: Predicting with encoder using data of shape {real_data_for_encoder.shape}")
                encoded_outputs = self._encoder.predict(real_data_for_encoder)
                if not (isinstance(encoded_outputs, list) and len(encoded_outputs) >= 2):
                    raise ValueError("FeederPlugin: Encoder output must be a list of at least two elements [z_mean, z_log_var, ...].")
                
                z_mean_array, z_log_var_array = encoded_outputs[0], encoded_outputs[1]

                # Ensure z_mean_array and z_log_var_array are 3D
                if z_mean_array.ndim != 3 or z_log_var_array.ndim != 3:
                    raise ValueError(
                        f"FeederPlugin: Encoder 'z_mean' and 'z_log_var' must be 3D (samples, seq_len, features). "
                        f"Got z_mean shape: {z_mean_array.shape}, z_log_var shape: {z_log_var_array.shape}"
                    )

                expected_latent_shape = tuple(self.params.get("latent_shape", [0,0])) # (seq_len, features)
                if (z_mean_array.shape[1] != expected_latent_shape[0] or
                    z_mean_array.shape[2] != expected_latent_shape[1]):
                    current_encoder_shape = (z_mean_array.shape[1], z_mean_array.shape[2])
                    print(f"FeederPlugin: Warning - Encoder latent shape {current_encoder_shape} "
                          f"mismatches configured 'latent_shape' {expected_latent_shape}. "
                          f"Using encoder's shape: {current_encoder_shape} for Z sampling.")
                    self.params["latent_shape"] = list(current_encoder_shape)
                
                if z_log_var_array.shape != z_mean_array.shape:
                    raise ValueError("FeederPlugin: z_mean and z_log_var shapes mismatch from encoder.")

                self._empirical_z_means = z_mean_array # Should be (num_samples, seq_len, features)
                self._empirical_z_log_vars = z_log_var_array # Should be (num_samples, seq_len, features)
                
                num_samples = self._empirical_z_means.shape[0]

                if num_samples <= 1:
                    print("FeederPlugin: Warning - Not enough empirical samples (<2) from encoder. 'direct' sampling will be used.")
                    return 

                if technique in ["kde", "copula"]:
                    # Fitting KDE/Copula on 3D (per sample) latent codes is complex and
                    # might not preserve sequence structure well if simply flattened.
                    # User comment "generated totally correctly, not randomly" suggests direct use of learned sequences.
                    print(f"FeederPlugin: Warning - '{technique}' sampling for 3D latent spaces is complex. "
                          "Consider using 'direct' sampling to preserve learned sequence structures from the encoder. "
                          "Falling back to 'direct' sampling for now.")
                    self.params["encoder_sampling_technique"] = "direct" # Override to direct
                
            except Exception as e:
                self._invalidate_encoder_state() 
                raise RuntimeError(f"FeederPlugin: Error durante el procesamiento del encoder. Error: {e}")

    def set_params(self, **kwargs):
        old_method = self.params.get("sampling_method")
        old_encoder_file = self.params.get("encoder_model_file")
        # old_data_file = self.params.get("real_data_file") # REMOVED
        old_data_file = self.main_config.get("x_train_file") # Get from stored main_config initially
        old_technique = self.params.get("encoder_sampling_technique")
        old_copula_bw_method = self.params.get("copula_kde_bw_method") 
        old_encoder_features = self.params.get("feature_columns_for_encoder")
        old_fundamental_features = self.params.get("fundamental_feature_names_for_conditioning")
        old_datetime_col = self.params.get("datetime_col_in_real_data")
        old_latent_shape = list(self.params.get("latent_shape", [0,0])) # Make a copy

        # Update main_config if kwargs contains keys that are part of main config
        # This is important if set_params is called with a more complete config later (e.g. by Optimizer)
        self.main_config.update(kwargs)

        # Update self.params using kwargs, handling prefixed keys
        for param_key_short in self.plugin_params.keys():
            if param_key_short in kwargs:
                self.params[param_key_short] = kwargs[param_key_short]
            else:
                potential_prefixed_key = f"feeder_{param_key_short}" 
                if potential_prefixed_key in kwargs:
                    self.params[param_key_short] = kwargs[potential_prefixed_key]
        
        # Handle 'latent_dim' from optimizer potentially overriding the feature part of 'latent_shape'
        if 'latent_dim' in kwargs and isinstance(kwargs['latent_dim'], int):
            current_shape = list(self.params.get('latent_shape', [0,0])) # Get current or default
            if len(current_shape) == 2:
                print(f"FeederPlugin: Updating latent_shape feature count from 'latent_dim'={kwargs['latent_dim']}. Old shape: {current_shape}")
                current_shape[1] = kwargs['latent_dim']
                self.params['latent_shape'] = current_shape
            else:
                print(f"FeederPlugin: Warning - 'latent_dim' provided, but current 'latent_shape' ({current_shape}) is not 2D. Cannot update feature count.")

        # Ensure latent_shape is a list of two integers
        ls = self.params.get("latent_shape")
        if not (isinstance(ls, (list, tuple)) and len(ls) == 2 and all(isinstance(x, int) for x in ls)):
            print(f"FeederPlugin: Warning - Invalid 'latent_shape' ({ls}). Resetting to default [18,32].")
            self.params["latent_shape"] = [18, 32]


        new_method = self.params.get("sampling_method")
        new_encoder_file = self.params.get("encoder_model_file")
        # new_data_file = self.params.get("real_data_file") # REMOVED
        new_data_file = self.main_config.get("x_train_file") # Get updated from main_config
        new_technique = self.params.get("encoder_sampling_technique")
        new_copula_bw_method = self.params.get("copula_kde_bw_method") 
        new_encoder_features = self.params.get("feature_columns_for_encoder")
        new_fundamental_features = self.params.get("fundamental_feature_names_for_conditioning")
        new_datetime_col = self.params.get("datetime_col_in_real_data")
        new_latent_shape = list(self.params.get("latent_shape", [0,0]))


        reinitialize_encoder_related = False
        if new_method == "from_encoder":
            if (old_method != new_method or 
                old_encoder_file != new_encoder_file or 
                old_data_file != new_data_file or
                old_encoder_features != new_encoder_features or
                old_technique != new_technique or
                (new_technique == "copula" and old_copula_bw_method != new_copula_bw_method) or
                old_latent_shape != new_latent_shape # Compare shapes
                ):
                reinitialize_encoder_related = True
            elif new_technique == "copula" and (self._copula_spearman_corr is None or not self._marginal_kdes) and self._empirical_z_means is not None and self._empirical_z_means.ndim ==3 : # Copula for 3D not implemented
                print("FeederPlugin: Copula for 3D latent space not implemented, will use direct.")
                reinitialize_encoder_related = True # To ensure _load_and_process sets technique to direct
            elif new_technique == "kde" and self._joint_kde is None and self._empirical_z_means is not None and self._empirical_z_means.ndim == 3: # KDE for 3D not implemented
                print("FeederPlugin: KDE for 3D latent space not implemented, will use direct.")
                reinitialize_encoder_related = True # To ensure _load_and_process sets technique to direct
            elif self._empirical_z_means is None: 
                reinitialize_encoder_related = True

        elif old_method == "from_encoder" and new_method != "from_encoder":
            self._invalidate_encoder_state()

        reinitialize_fundamentals = False
        if (old_data_file != new_data_file or
            old_fundamental_features != new_fundamental_features or
            old_datetime_col != new_datetime_col or
            (len(new_fundamental_features) > 0 and self._real_fundamental_features_df_scaled is None) or
            (len(new_fundamental_features) > 0 and self._real_datetimes_pd_series is None)
            ):
            reinitialize_fundamentals = True
        
        if reinitialize_encoder_related or reinitialize_fundamentals:
            # Check against the potentially updated self.main_config.get("x_train_file")
            if self.main_config.get("x_train_file") or (new_method == "from_encoder" and self.params.get("encoder_model_file")):
                try:
                    print("FeederPlugin: Re-initializing data due to parameter change.")
                    self._load_and_process_for_encoder_mode()
                except Exception as e:
                    print(f"FeederPlugin: Error during re-initialization in set_params: {e}")
                    self._invalidate_encoder_state() 
            else: 
                if new_method == "from_encoder" or len(new_fundamental_features) > 0:
                    print(f"FeederPlugin: Warning - 'x_train_file' (from main config) not provided, but needed for current settings. Clearing related state.")
                self._invalidate_encoder_state()

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
        if "day_of_year" in self.params["date_feature_names_for_conditioning"]: # This is the crucial check
            doy = datetime_obj.dayofyear
            date_features.append(np.sin(2 * np.pi * doy / self.params["max_day_of_year"]))
            date_features.append(np.cos(2 * np.pi * doy / self.params["max_day_of_year"]))
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

        latent_s = self.params.get("latent_shape") # Should be [seq_len, features]
        if not (isinstance(latent_s, (list, tuple)) and len(latent_s) == 2):
            raise ValueError(f"Invalid latent_shape configuration: {latent_s}. Expected [sequence_length, features].")
        latent_seq_len, latent_features = latent_s[0], latent_s[1]

        method = self.params.get("sampling_method")
        technique = self.params.get("encoder_sampling_technique") # May have been forced to 'direct'
        context_dim = self.params.get("context_vector_dim")
        context_strategy = self.params.get("context_vector_strategy")
        
        if latent_features <= 0 or latent_seq_len <=0 and n_ticks_to_generate > 0 :
            raise ValueError(f"FeederPlugin: Effective latent_shape is ({latent_seq_len}, {latent_features}), must be positive to generate samples.")

        Z_samples = np.empty((n_ticks_to_generate, latent_seq_len, latent_features), dtype=np.float32)

        if method == "from_encoder":
            if self._empirical_z_means is None or self._empirical_z_log_vars is None:
                raise RuntimeError("FeederPlugin: 'from_encoder' selected, but empirical z_mean/z_log_var are not loaded. Call set_params or ensure config is correct.")
            if self._empirical_z_means.ndim != 3 or self._empirical_z_log_vars.ndim != 3:
                 raise RuntimeError(f"FeederPlugin: Empirical z_mean/z_log_var must be 3D for sequential latent space. Got shapes: {self._empirical_z_means.shape}, {self._empirical_z_log_vars.shape}")
            
            # For 3D latent spaces, 'direct' sampling of entire sequences is the primary supported method.
            # KDE/Copula for 3D sequences is complex and not implemented here.
            if technique not in ["direct"]:
                print(f"FeederPlugin: Warning (generate) - Encoder technique '{technique}' for 3D latent space is not fully supported. Using 'direct' sampling.")
            
            num_available_empirical = self._empirical_z_means.shape[0]
            indices = np.random.choice(num_available_empirical, size=n_ticks_to_generate, replace=True)
            
            sampled_z_means_batch = self._empirical_z_means[indices] # Shape: (n_ticks, seq_len, features)
            sampled_z_log_vars_batch = self._empirical_z_log_vars[indices] # Shape: (n_ticks, seq_len, features)

            epsilon_batch = np.random.normal(loc=0.0, scale=1.0, size=(n_ticks_to_generate, latent_seq_len, latent_features))
            Z_samples = sampled_z_means_batch + np.exp(0.5 * sampled_z_log_vars_batch) * epsilon_batch
        
        elif method == "standard_normal":
            Z_samples = np.random.normal(loc=0.0, scale=1.0, size=(n_ticks_to_generate, latent_seq_len, latent_features))
        else:
            raise ValueError(f"FeederPlugin: Unknown sampling_method '{method}'.")

        output_sequence = []
        for i in range(n_ticks_to_generate):
            datetime_obj = target_datetimes.iloc[i]
            z_tick = Z_samples[i, :, :] # Shape (latent_seq_len, latent_features)

            scaled_date_features_tick = self._get_scaled_date_features(datetime_obj)
            scaled_fundamental_features_tick = self._get_scaled_fundamental_features(datetime_obj)
            
            conditional_data_for_tick = np.concatenate(
                [scaled_date_features_tick, scaled_fundamental_features_tick]
            ).reshape(1, -1)

            context_h_tick = np.zeros((1, context_dim), dtype=np.float32)
            # ... (context_strategy logic if any) ...

            output_sequence.append({
                "Z": z_tick.astype(np.float32), # Now 2D: (seq_len, features)
                "conditional_data": conditional_data_for_tick.astype(np.float32),
                "context_h": context_h_tick.astype(np.float32),
                "datetimes": datetime_obj
            })
            
        return output_sequence
