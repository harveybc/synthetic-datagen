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

    # Data for evaluation and base for generation
    # "real_data_file": "examples/data/phase_3/normalized_d4.csv", # REMOVED - Redundant, use x_train_file
    "x_train_file": "examples/data/phase_3/normalized_d4.csv", # Primary data source
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
    "n_samples": 50400,
    "max_steps_train": 25200,
    "latent_shape": [18, 32], 
    "batch_size": 32, 
    
    # --- Parameters for FeederPlugin ---
    "feeder_sampling_method": "standard_normal", 
    "feeder_encoder_sampling_technique": "direct", 
    "encoder_model_file": "examples/results/phase_4_2/phase_4_2_cnn_small_encoder_model.keras", 
    "feeder_feature_columns_for_encoder": [], 
    "feeder_real_data_file_has_header": True,
    "feeder_datetime_col_in_real_data": "DATE_TIME",
    "feeder_date_features_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week", "day_of_year"], # ADDED "day_of_year"
    "feeder_fundamental_features_for_conditioning": ["S&P500_Close", "vix_close"],
    "feeder_max_day_of_month": 31,
    "feeder_max_hour_of_day": 23,
    "feeder_max_day_of_week": 6,
    "feeder_max_day_of_year": 366, # ADDED
    "feeder_context_vector_dim": 64, # CHANGED from 16 to 64 to match Generator/main config
    "feeder_context_vector_strategy": "random",
    "feeder_copula_kde_bw_method": None,

    # --- Parameters for GeneratorPlugin ---
    "generator_sequential_model_file": "examples/results/phase_4_2/phase_4_2_cnn_small_decoder_model.keras",
    "generator_decoder_input_window_size": 144, 
    "generator_full_feature_names_ordered": [
        "DATE_TIME", 
        "OPEN", "HIGH", "LOW", "CLOSE", # CLOSE is derived
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        "day_of_month_sin", "day_of_month_cos",
        "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_year_sin", "day_of_year_cos",
        "S&P500_Close", "vix_close",
        "log_return", # log_return is now expected from decoder
        "stl_trend", "stl_seasonal", "stl_resid",
        "wav_approx_L2", "wav_detail_L1", "wav_detail_L2",
        "mtm_band_0", "mtm_band_1", "mtm_band_2", "mtm_band_3",
        "BC-BO", "BH-BL", "BH-BO", "BO-BL",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8",
        "day_of_month", "hour_of_day", "day_of_week" 
    ], 
    "generator_decoder_output_feature_names": [
        # Based on cvae_target_feature_names from phase_4_2_cnn_small_debug_out.json
        "OPEN", "LOW", "HIGH", # "vix_close", "S&P500_Close" are removed from this list
        "BC-BO", "BH-BL", 
        # "S&P500_Close", # REMOVED
        # "vix_close", # REMOVED
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
        # "log_return" REMOVED - It's NOT in the cvae_target_feature_names of the trained model
    ], 
    "generator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
    "generator_ti_feature_names": [ 
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-",
        "ATR", "CCI", "WilliamsR", "Momentum", "ROC"
    ],
    "generator_date_conditional_feature_names": ["day_of_month", "hour_of_day", "day_of_week", "day_of_year"], 
    "generator_feeder_conditional_feature_names": ["S&P500_Close", "vix_close"], 
    "generator_ti_calculation_min_lookback": 200, 
    "generator_ti_params": { 
        "rsi_length": 14, "ema_length": 14, 
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "stoch_k": 14, "stoch_d": 3, "stoch_smooth_k": 3,
        "adx_length": 14, "atr_length": 14, "cci_length": 14, 
        "willr_length": 14, "mom_length": 14, "roc_length": 14
    },
    "generator_normalization_params_file": "examples/data/phase_3/phase_3_debug_out.json",
    "generator_decoder_input_name_latent": "decoder_input_z_seq",       
    "generator_decoder_input_name_window": "input_x_window",          
    "generator_decoder_input_name_conditions": "decoder_input_conditions", 
    "generator_decoder_input_name_context": "decoder_input_h_context",   
    "context_vector_dim": 64, # This is the main config value, Feeder should align

    # --- Parameters for EvaluatorPlugin ---
    "evaluator_metrics": ["mmd", "acf", "wasserstein", "kstest", "discriminative_score", "predictive_score", "visual"],
    "evaluator_mmd_gamma": None, 
    "evaluator_acf_nlags": 20,
    "evaluator_predictive_model_type": "LSTM", 
    "evaluator_predictive_epochs": 10,
    "evaluator_predictive_batch_size": 32,
    "evaluator_plot_max_features": 10, 
    "evaluator_plot_max_lags_acf": 50, 

    # --- Parameters for OptimizerPlugin ---
    "hyperparameter_optimization_mode": False, 
    "run_hyperparameter_optimization": True, 
    "population_size": 10,
    "n_generations": 5,
    "cxpb": 0.6,
    "mutpb": 0.3,
    "hyperparameter_bounds": {
        "latent_dim": (8, 64), 
        "batch_size": (16, 128), 
    },
    "optimizer_n_samples_per_eval": 1000, 
    "optimizer_start_datetime": None, 

    # General execution parameters
    "random_seed": 42,
    "num_synthetic_samples_to_generate": 0, 
    "start_datetime": None, 
    "output_file": "examples/results/phase_4_2/normalized_d4_25200_synthetic_50400_prepended_o.csv",
    #"synthetic_data_output_file": "examples/results/phase_4_2/normalized_d4_25200_synthetic_25200_prepended.csv",
    "metrics_file": "examples/results/phase_4_2/normalized_d4_25200_synthetic_50400_metrics.json",
    "save_config": "examples/results/phase_4_2/config_out_50400.json",
    "save_log": "examples/results/phase_4_2/debug_out__50400.json",
    "quiet_mode": False,
    "datetime_col_name": "DATE_TIME",
    "target_column_order": [],
    "num_base_features_generated": 6, # Example, adjust if necessary
    "preprocessor_plugin": "stl_preprocessor", # Example, adjust if necessary
    "gan_training_mode": False # Example, adjust if necessary
}
