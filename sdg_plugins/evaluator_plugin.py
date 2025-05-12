"""
EvaluatorPlugin: Evalúa la calidad de los datos sintéticos comparándolos
con la señal real, usando métricas estadísticas y de correlación.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from statsmodels.tsa.stattools import acf as sm_acf
from scipy.stats import wasserstein_distance, skew, kurtosis, ks_2samp
from sklearn.metrics.pairwise import rbf_kernel # For MMD calculation

class EvaluatorPlugin:
    plugin_params = {
        "real_data_file": None, # Retained for potential future use, but not used in evaluate if real_data_processed is given
        "window_size": 288, # Example, might not be directly used by this version of evaluate
        "mmd_gamma": None,  # Kernel bandwidth for MMD. If None, sklearn's default is used.
        "acf_nlags": 20     # Number of lags to consider for ACF comparison.
    }
    plugin_debug_vars = ["real_data_file", "window_size", "mmd_gamma", "acf_nlags"]

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        self.params = self.plugin_params.copy()
        self.set_params(**config)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    def _calculate_mmd_rbf(self, X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        if X.shape[0] == 0 or Y.shape[0] == 0: return np.nan

        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
        
        m = K_XX.shape[0]
        n = K_YY.shape[0]
        if m == 0 or n == 0: return np.nan

        # Adjust for unbiased MMD^2 estimator if m, n > 1
        term1 = np.sum(K_XX - np.diag(np.diag(K_XX))) / (m * (m - 1)) if m > 1 else 0
        term2 = np.sum(K_YY - np.diag(np.diag(K_YY))) / (n * (n - 1)) if n > 1 else 0
        term3 = 2 * np.sum(K_XY) / (m * n) if m > 0 and n > 0 else 0
        
        mmd_sq = term1 + term2 - term3
        return np.sqrt(max(0, mmd_sq))

    def evaluate(
        self,
        synthetic_data: np.ndarray,
        real_data_processed: np.ndarray,
        real_dates: Optional[pd.Index], # Currently unused, but kept for interface consistency
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if config:
            self.set_params(**config)

        real_signal = real_data_processed

        if real_signal.ndim != 2:
            raise ValueError(f"real_data_processed must be 2D (samples, features), got ndim={real_signal.ndim}")
        if synthetic_data.ndim != 2 or synthetic_data.shape != real_signal.shape:
            raise ValueError(f"synthetic_data shape {synthetic_data.shape} != real_data_processed shape {real_signal.shape}")
        
        n_features = real_signal.shape[1]
        if not feature_names or len(feature_names) != n_features:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            print(f"WARN: feature_names not provided or mismatched. Using generic names: {feature_names}")

        # --- Per-feature metrics ---
        metrics_per_feature = {}
        acf_nlags_to_calc = self.params.get("acf_nlags", 20)

        for f_idx in range(n_features):
            feat_name = feature_names[f_idx]
            y_true_feat = real_signal[:, f_idx]
            y_syn_feat = synthetic_data[:, f_idx]
            
            feat_metrics = {}

            # Statistical Moments
            feat_metrics["mean_real"] = float(np.mean(y_true_feat))
            feat_metrics["mean_synthetic"] = float(np.mean(y_syn_feat))
            feat_metrics["mean_abs_diff"] = float(np.abs(feat_metrics["mean_real"] - feat_metrics["mean_synthetic"]))

            feat_metrics["std_real"] = float(np.std(y_true_feat))
            feat_metrics["std_synthetic"] = float(np.std(y_syn_feat))
            feat_metrics["std_abs_diff"] = float(np.abs(feat_metrics["std_real"] - feat_metrics["std_synthetic"]))
            
            feat_metrics["skew_real"] = float(skew(y_true_feat)) if len(y_true_feat) > 0 else np.nan
            feat_metrics["skew_synthetic"] = float(skew(y_syn_feat)) if len(y_syn_feat) > 0 else np.nan
            feat_metrics["skew_abs_diff"] = float(np.abs(feat_metrics["skew_real"] - feat_metrics["skew_synthetic"])) if not (np.isnan(feat_metrics["skew_real"]) or np.isnan(feat_metrics["skew_synthetic"])) else np.nan

            feat_metrics["kurtosis_real"] = float(kurtosis(y_true_feat)) if len(y_true_feat) > 0 else np.nan # Fisher’s definition (normal ==> 0)
            feat_metrics["kurtosis_synthetic"] = float(kurtosis(y_syn_feat)) if len(y_syn_feat) > 0 else np.nan
            feat_metrics["kurtosis_abs_diff"] = float(np.abs(feat_metrics["kurtosis_real"] - feat_metrics["kurtosis_synthetic"])) if not (np.isnan(feat_metrics["kurtosis_real"]) or np.isnan(feat_metrics["kurtosis_synthetic"])) else np.nan

            # Distributional Distances
            if len(y_true_feat) > 0 and len(y_syn_feat) > 0:
                try:
                    feat_metrics["wasserstein_dist"] = float(wasserstein_distance(y_true_feat, y_syn_feat))
                except Exception as e: feat_metrics["wasserstein_dist"] = f"Error: {e}"
                
                try:
                    feat_metrics["mmd_rbf"] = float(self._calculate_mmd_rbf(y_true_feat, y_syn_feat, gamma=self.params.get("mmd_gamma")))
                except Exception as e: feat_metrics["mmd_rbf"] = f"Error: {e}"
                
                try:
                    ks_stat, _ = ks_2samp(y_true_feat, y_syn_feat)
                    feat_metrics["ks_stat"] = float(ks_stat)
                except Exception as e: feat_metrics["ks_stat"] = f"Error: {e}"
            else:
                feat_metrics["wasserstein_dist"] = np.nan
                feat_metrics["mmd_rbf"] = np.nan
                feat_metrics["ks_stat"] = np.nan

            # Temporal Dynamics (ACF)
            acf_real, acf_synthetic = np.array([np.nan]), np.array([np.nan]) # Default to NaN array
            if len(y_true_feat) > acf_nlags_to_calc :
                try:
                    acf_real = sm_acf(y_true_feat, nlags=acf_nlags_to_calc, fft=False, adjusted=False)[1:] # Exclude lag 0
                except Exception as e: feat_metrics["acf_mae_diff"] = f"Error real ACF: {e}"
            if len(y_syn_feat) > acf_nlags_to_calc:
                try:
                    acf_synthetic = sm_acf(y_syn_feat, nlags=acf_nlags_to_calc, fft=False, adjusted=False)[1:] # Exclude lag 0
                except Exception as e: feat_metrics["acf_mae_diff"] = f"Error synthetic ACF: {e}"

            if not isinstance(feat_metrics.get("acf_mae_diff"), str): # if no error string was set
                if acf_real.shape == acf_synthetic.shape and not (np.all(np.isnan(acf_real)) or np.all(np.isnan(acf_synthetic))):
                    feat_metrics["acf_mae_diff"] = float(np.mean(np.abs(acf_real - acf_synthetic)))
                else:
                    feat_metrics["acf_mae_diff"] = np.nan
            
            # Store all metrics for this feature
            metrics_per_feature[feat_name] = feat_metrics
        
        # --- Overall/Multivariate Metrics ---
        overall_metrics = {}
        
        # Correlation Matrix Difference
        if n_features > 1:
            try:
                corr_real = np.corrcoef(real_signal, rowvar=False)
                corr_synthetic = np.corrcoef(synthetic_data, rowvar=False)
                
                # Handle cases where correlation might be NaN (e.g., constant columns)
                corr_real = np.nan_to_num(corr_real)
                corr_synthetic = np.nan_to_num(corr_synthetic)

                corr_diff_frobenius = float(np.linalg.norm(corr_real - corr_synthetic, 'fro'))
                overall_metrics["correlation_matrix_frobenius_diff"] = corr_diff_frobenius
                
                # Mean Absolute Difference of off-diagonal elements
                mask = ~np.eye(n_features, dtype=bool)
                mad_corr_elements = float(np.mean(np.abs(corr_real[mask] - corr_synthetic[mask])))
                overall_metrics["correlation_matrix_mad_off_diagonal"] = mad_corr_elements

            except Exception as e:
                overall_metrics["correlation_matrix_frobenius_diff"] = f"Error: {e}"
                overall_metrics["correlation_matrix_mad_off_diagonal"] = f"Error: {e}"
        else:
            overall_metrics["correlation_matrix_frobenius_diff"] = "N/A (single feature)"
            overall_metrics["correlation_matrix_mad_off_diagonal"] = "N/A (single feature)"


        # --- Aggregate per-feature metrics for an overall view ---
        # (e.g., average Wasserstein distance across features)
        for metric_key in ["mean_abs_diff", "std_abs_diff", "skew_abs_diff", "kurtosis_abs_diff", 
                           "wasserstein_dist", "mmd_rbf", "ks_stat", "acf_mae_diff"]:
            values = [metrics_per_feature[f].get(metric_key) for f in feature_names if isinstance(metrics_per_feature[f].get(metric_key), (int, float))]
            if values: # Check if list is not empty and contains valid numbers
                overall_metrics[f"avg_{metric_key}"] = float(np.nanmean(values)) if values else np.nan
            else:
                overall_metrics[f"avg_{metric_key}"] = np.nan


        # --- Basic Point-wise errors (less emphasis for distributional quality) ---
        # You can choose to include these or not based on specific needs.
        # For now, I'm keeping them separate to highlight the distributional metrics.
        basic_errors = {
            "overall_mae_pointwise": float(np.mean(np.abs(synthetic_data - real_signal))),
            "overall_mse_pointwise": float(np.mean((synthetic_data - real_signal) ** 2)),
        }

        final_metrics = {
            "per_feature_fidelity": metrics_per_feature,
            "overall_multivariate_fidelity": overall_metrics,
            "basic_pointwise_errors": basic_errors # Optional
        }
        
        # Add feature names list to the top level for easy reference
        final_metrics["feature_names_evaluated"] = feature_names

        return final_metrics
