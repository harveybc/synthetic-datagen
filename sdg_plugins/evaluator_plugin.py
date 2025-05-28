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
from tqdm import tqdm # For progress bars

class EvaluatorPlugin:
    plugin_params = {
        "real_data_file": None, 
        "window_size": 288, 
        "mmd_gamma": None,  
        "acf_nlags": 20,
        "max_steps_val": None # Max steps (rows) to use for validation. If None, use all.
    }
    plugin_debug_vars = ["real_data_file", "window_size", "mmd_gamma", "acf_nlags", "max_steps_val"]

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

        term1 = np.sum(K_XX - np.diag(np.diag(K_XX))) / (m * (m - 1)) if m > 1 else 0
        term2 = np.sum(K_YY - np.diag(np.diag(K_YY))) / (n * (n - 1)) if n > 1 else 0
        term3 = 2 * np.sum(K_XY) / (m * n) if m > 0 and n > 0 else 0
        
        mmd_sq = term1 + term2 - term3
        return np.sqrt(max(0, mmd_sq))

    def evaluate(
        self,
        synthetic_data: np.ndarray,
        real_data_processed: np.ndarray,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if config:
            self.set_params(**config)

        eval_real_signal = real_data_processed
        eval_synthetic_data = synthetic_data
        
        max_steps = self.params.get("max_steps_val")
        if max_steps is not None and max_steps > 0:
            if eval_real_signal.shape[0] > max_steps:
                print(f"Limiting evaluation data to first {max_steps} steps (out of {eval_real_signal.shape[0]}).")
                eval_real_signal = eval_real_signal[:max_steps, :]
            if eval_synthetic_data.shape[0] > max_steps:
                # Ensure synthetic data is also sliced if real data was, or if it's independently larger
                eval_synthetic_data = eval_synthetic_data[:max_steps, :]


        if eval_real_signal.ndim != 2:
            raise ValueError(f"real_data_processed must be 2D (samples, features), got ndim={eval_real_signal.ndim}")
        
        # MODIFIED: Check only for ndim and feature count (shape[1]) mismatch for synthetic_data.
        # Allow different number of samples (shape[0]).
        if eval_synthetic_data.ndim != 2 or eval_synthetic_data.shape[1] != eval_real_signal.shape[1]:
            raise ValueError(
                f"Synthetic data feature count mismatch or dimension error. Synthetic shape: {eval_synthetic_data.shape}, Real shape: {eval_real_signal.shape}. Original shapes were {synthetic_data.shape} and {real_data_processed.shape}"
            )
        
        n_features = eval_real_signal.shape[1]
        if not feature_names or len(feature_names) != n_features:
            feature_names_eval = [f"Feature_{i}" for i in range(n_features)]
            # Use original feature_names for the final report if available and matches original n_features
            # This part can be tricky if n_features itself changes due to some upstream processing not reflected here.
            # For now, we'll use the feature_names list that corresponds to the (potentially sliced) data.
            if feature_names and len(feature_names) == real_data_processed.shape[1]:
                 report_feature_names = feature_names
            else:
                 report_feature_names = [f"OriginalFeature_{i}" for i in range(real_data_processed.shape[1])]
                 if len(feature_names_eval) != len(report_feature_names): # If slicing changed feature count (should not happen with row slicing)
                    print(f"WARN: Feature count mismatch after potential slicing. Using generic names for report.")
                    report_feature_names = feature_names_eval # Fallback

            print(f"WARN: feature_names not provided or mismatched for evaluation. Using generic names for processing: {feature_names_eval}")
        else:
            feature_names_eval = feature_names
            report_feature_names = feature_names


        metrics_per_feature = {}
        acf_nlags_to_calc = self.params.get("acf_nlags", 20)

        print("Calculating per-feature metrics...")
        for f_idx in tqdm(range(n_features), desc="Processing Features"):
            feat_name = feature_names_eval[f_idx]
            y_true_feat = eval_real_signal[:, f_idx]
            y_syn_feat = eval_synthetic_data[:, f_idx]
            
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

            feat_metrics["kurtosis_real"] = float(kurtosis(y_true_feat)) if len(y_true_feat) > 0 else np.nan
            feat_metrics["kurtosis_synthetic"] = float(kurtosis(y_syn_feat)) if len(y_syn_feat) > 0 else np.nan
            feat_metrics["kurtosis_abs_diff"] = float(np.abs(feat_metrics["kurtosis_real"] - feat_metrics["kurtosis_synthetic"])) if not (np.isnan(feat_metrics["kurtosis_real"]) or np.isnan(feat_metrics["kurtosis_synthetic"])) else np.nan

            # Distributional Distances
            if len(y_true_feat) > 0 and len(y_syn_feat) > 0:
                try: feat_metrics["wasserstein_dist"] = float(wasserstein_distance(y_true_feat, y_syn_feat))
                except Exception as e: feat_metrics["wasserstein_dist"] = f"Error: {e}"
                try: feat_metrics["mmd_rbf"] = float(self._calculate_mmd_rbf(y_true_feat, y_syn_feat, gamma=self.params.get("mmd_gamma")))
                except Exception as e: feat_metrics["mmd_rbf"] = f"Error: {e}"
                try:
                    ks_stat, _ = ks_2samp(y_true_feat, y_syn_feat)
                    feat_metrics["ks_stat"] = float(ks_stat)
                except Exception as e: feat_metrics["ks_stat"] = f"Error: {e}"
            else:
                feat_metrics["wasserstein_dist"], feat_metrics["mmd_rbf"], feat_metrics["ks_stat"] = np.nan, np.nan, np.nan

            # Temporal Dynamics (ACF)
            acf_real, acf_synthetic = np.array([np.nan]), np.array([np.nan])
            if len(y_true_feat) > acf_nlags_to_calc:
                try: acf_real = sm_acf(y_true_feat, nlags=acf_nlags_to_calc, fft=False, adjusted=False)[1:]
                except Exception as e: feat_metrics["acf_mae_diff"] = f"Error real ACF: {e}"
            if len(y_syn_feat) > acf_nlags_to_calc:
                try: acf_synthetic = sm_acf(y_syn_feat, nlags=acf_nlags_to_calc, fft=False, adjusted=False)[1:]
                except Exception as e: feat_metrics["acf_mae_diff"] = f"Error synthetic ACF: {e}"

            if not isinstance(feat_metrics.get("acf_mae_diff"), str):
                if acf_real.shape == acf_synthetic.shape and not (np.all(np.isnan(acf_real)) or np.all(np.isnan(acf_synthetic))):
                    feat_metrics["acf_mae_diff"] = float(np.mean(np.abs(acf_real - acf_synthetic)))
                else: feat_metrics["acf_mae_diff"] = np.nan
            
            metrics_per_feature[feat_name] = feat_metrics
        
        overall_metrics = {}
        print("\nCalculating overall/multivariate metrics...")
        if n_features > 1:
            try:
                # Use tqdm for correlation calculation if it's slow, though np.corrcoef is usually fast
                # For now, assuming it's not the main bottleneck compared to per-feature loops
                corr_real = np.corrcoef(eval_real_signal, rowvar=False)
                corr_synthetic = np.corrcoef(eval_synthetic_data, rowvar=False)
                corr_real, corr_synthetic = np.nan_to_num(corr_real), np.nan_to_num(corr_synthetic)
                overall_metrics["correlation_matrix_frobenius_diff"] = float(np.linalg.norm(corr_real - corr_synthetic, 'fro'))
                mask = ~np.eye(n_features, dtype=bool)
                overall_metrics["correlation_matrix_mad_off_diagonal"] = float(np.mean(np.abs(corr_real[mask] - corr_synthetic[mask])))
            except Exception as e:
                overall_metrics["correlation_matrix_frobenius_diff"] = f"Error: {e}"
                overall_metrics["correlation_matrix_mad_off_diagonal"] = f"Error: {e}"
        else:
            overall_metrics["correlation_matrix_frobenius_diff"] = "N/A (single feature)"
            overall_metrics["correlation_matrix_mad_off_diagonal"] = "N/A (single feature)"

        print("Aggregating per-feature metrics...")
        for metric_key in tqdm(["mean_abs_diff", "std_abs_diff", "skew_abs_diff", "kurtosis_abs_diff", 
                                "wasserstein_dist", "mmd_rbf", "ks_stat", "acf_mae_diff"], desc="Aggregating Metrics"):
            values = [metrics_per_feature[f].get(metric_key) for f in feature_names_eval if isinstance(metrics_per_feature[f].get(metric_key), (int, float))]
            overall_metrics[f"avg_{metric_key}"] = float(np.nanmean(values)) if values else np.nan
        
        basic_errors = {}
        print("Calculating basic point-wise errors...")
        # These are usually very fast, so tqdm might be overkill here unless data is truly massive even after slicing
        
        # MODIFIED: Calculate point-wise errors only if sample counts are identical.
        if eval_synthetic_data.shape[0] == eval_real_signal.shape[0]:
            basic_errors["overall_mae_pointwise"] = float(np.mean(np.abs(eval_synthetic_data - eval_real_signal)))
            basic_errors["overall_mse_pointwise"] = float(np.mean((eval_synthetic_data - eval_real_signal) ** 2))
            print(f"INFO: Point-wise MAE/MSE calculated as sample sizes match ({eval_synthetic_data.shape[0]} samples).")
        else:
            basic_errors["overall_mae_pointwise"] = np.nan
            basic_errors["overall_mse_pointwise"] = np.nan
            print(f"INFO: Point-wise MAE/MSE not calculated. Synthetic samples: {eval_synthetic_data.shape[0]}, Real samples: {eval_real_signal.shape[0]}. These metrics require identical sample counts for direct comparison.")

        final_metrics = {
            "per_feature_fidelity": metrics_per_feature,
            "overall_multivariate_fidelity": overall_metrics,
            "basic_pointwise_errors": basic_errors 
        }
        
        final_metrics["feature_names_evaluated_on"] = feature_names_eval # Names used for keys in per_feature_fidelity
        final_metrics["original_feature_names_reported"] = report_feature_names # Original names for context
        final_metrics["evaluation_max_steps_used"] = max_steps if max_steps is not None and max_steps > 0 and real_data_processed.shape[0] > max_steps else "all"
        final_metrics["evaluation_actual_steps_used"] = eval_real_signal.shape[0]


        print("Metric calculation complete.")
        return final_metrics
