"""
config.py for SDG (Synthetic Data Generator)

This module defines the default values for every command-line parameter
supported by the sdg application. These defaults are used when no value
is provided via CLI, config file, or remote config.
"""

DEFAULT_VALUES = {
    # Plugin selection
    "feeder": "default_feeder",
    "generator": "default_generator",
    "evaluator": "default_evaluator",
    "optimizer": "default_optimizer",

    # Data for evaluation
    "real_data_file": "examples/data/phase_3/normalized_d4.csv",
    "x_train_file": "examples/data/phase_3/normalized_d4.csv",
    "y_train_file": "examples/data/phase_3/normalized_d4.csv",
    "x_validation_file": "examples/data/phase_3/normalized_d5.csv",
    "y_validation_file": "examples/data/phase_3/normalized_d5.csv",
    "x_test_file": "examples/data/phase_3/normalized_d6.csv",
    "y_test_file": "examples/data/phase_3/normalized_d6.csv",
    "target_column": "CLOSE",
    "stl_period":24,
    "predicted_horizons": [24,48,72,96,120,144],
    "use_stl": True,
    "use_wavelets": True,
    "use_multi_tapper": True,

    "dataset_periodicity": "1h", 

     # Generation parameters
    "n_samples": 6300,  # Retained for compatibility, but main.py now uses num_synthetic_samples_to_generate
    "latent_dim": 16, # FeederPlugin's latent_dim, main.py might override it based on decoder
    "batch_size": 32, # May not be directly used by new sequential generator logic
    
    # --- Parameters for FeederPlugin ---
    "feeder_sampling_method": "standard_normal", # "standard_normal", "from_encoder"
    "feeder_encoder_sampling_technique": "direct", # "direct", "kde", "copula"
    "encoder_model_file": "examples/results/phase_4_2/phase_4_2_cnn_small_encoder_model.h5.keras", # Used by Feeder if method is "from_encoder"
    "feeder_feature_columns_for_encoder": [], # List of col names from feeder_real_data_file for VAE encoder input
    "feeder_real_data_file_has_header": True,
    "feeder_datetime_col_in_real_data": "DATE_TIME",
    "feeder_date_features_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"],
    "feeder_fundamental_features_for_conditioning": ["S&P500_Close", "vix_close"],
    "feeder_max_day_of_month": 31,
    "feeder_max_hour_of_day": 23,
    "feeder_max_day_of_week": 6,
    "feeder_context_vector_dim": 16, # Example, should match decoder's context input if used
    "feeder_context_vector_strategy": "zeros",
    "feeder_copula_kde_bw_method": None,

    # --- Parameters for GeneratorPlugin ---
    "generator_sequential_model_file": "examples/results/phase_4_2/phase_4_2_cnn_small_decoder_model.h5.keras", # Path to Keras decoder
    "generator_decoder_input_window_size": 144,
    "generator_normalization_params_file": "examples/data/phase_3/phase_3_debug_out.json", # Path to min/max JSON
    "generator_full_feature_names_ordered": [
        "DATE_TIME", "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI",
        "WilliamsR", "Momentum", "ROC", "OPEN", "HIGH", "LOW", "CLOSE",
        "BC-BO", "BH-BL", "BH-BO", "BO-BL", "S&P500_Close", "vix_close",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8",
        "day_of_month_sin", "day_of_month_cos", 
        "hour_of_day_sin", "hour_of_day_cos", 
        "day_of_week_sin", "day_of_week_cos"
    ],
    # IMPORTANT: Verify this list matches your Keras decoder's actual output features.
    # This list assumes the decoder outputs OHLC, derived OHLC, and lagged close prices.
    "generator_decoder_output_feature_names": [
        "OPEN", "HIGH", "LOW", "CLOSE",
        "BC-BO", "BH-BL", "BH-BO", "BO-BL",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ],
    "generator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
    "generator_ti_feature_names": [
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-",
        "ATR", "CCI", "WilliamsR", "Momentum", "ROC"
    ],
    "generator_date_conditional_feature_names": ["day_of_month", "hour_of_day", "day_of_week"], # Original names feeder uses for transformation
    "generator_feeder_conditional_feature_names": ["S&P500_Close", "vix_close"], # Should match feeder's fundamental features
    "generator_ti_calculation_min_lookback": 200,
    "generator_ti_params": {
        "rsi_length": 14, "ema_length": 14,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "stoch_k": 14, "stoch_d": 3, "stoch_smooth_k": 3,
        "adx_length": 14, "atr_length": 14, "cci_length": 14,
        "willr_length": 14, "mom_length": 14, "roc_length": 14
    },
    "generator_decoder_input_name_latent": "input_latent_z",
    "generator_decoder_input_name_window": "input_x_window",
    "generator_decoder_input_name_conditions": "input_conditions_t",
    "generator_decoder_input_name_context": "input_context_h",

    # --- Parameters for main.py generation control ---
    "start_datetime": None, # e.g., "2023-01-01 00:00:00" or None to use eval data start
    "num_synthetic_samples_to_generate": 0, # 0 to match length of evaluation data segment

     "max_steps_train": 6300,
     "max_steps_val": 6300,
     "max_steps_test": 6300,

     # Output paths
    "output_file": "examples/results/phase_4_2/normalized_d4_25200_synthetic_50400_prepended.csv", # Retained for compatibility if old workflow parts use it
    "synthetic_data_output_file": "examples/results/phase_4_2/generated_full_synthetic_data.csv", # For the new generator output
     "metrics_file": "examples/results/phase_4_2/normalized_d4_25200_synthetic_50400_metrics.json",

    # Optimizer parameters
    "latent_dim_range": [8, 64],
    "iterations": 10,

    # Remote config & logging
    "remote_log": None,
    "remote_load_config": None,
    "remote_save_config": None,
    "username": None,
    "password": None,

    # Local config persistence
    "load_config": None,
    "save_config": "examples/results/phase_4_2/config_out.json",
    "save_log": "examples/results/phase_4_2/debug_out.json",

    # CLI behavior
    "quiet_mode": False,
}
