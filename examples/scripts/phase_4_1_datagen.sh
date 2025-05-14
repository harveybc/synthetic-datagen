#!/bin/bash
# Dgenerate synthetic data for phase 4_1 daily transformer configurations
RESULTS_DIR="examples/data/phase_4_1"

# Process the JSON configuration files(TODO: Fix that n_samples must be the same as max_steps_train in the code)
sh sdg.sh  --n_samples 12600 --max_steps_train 25200 \
    --output_file "$RESULTS_DIR/normalized_d4_25200_synthetic_3150_prepended.csv" \
    --metrics_file "$RESULTS_DIR/normalized_d4_25200_synthetic_3150_metrics.csv" 
sh sdg.sh  --n_samples 25200 --max_steps_train 25200 \
    --output_file "$RESULTS_DIR/normalized_d4_25200_synthetic_25200_prepended.csv" \
    --metrics_file "$RESULTS_DIR/normalized_d4_25200_synthetic_25200_metrics.csv"
sh sdg.sh  --n_samples 50400 --max_steps_train 25200 \
    --output_file "$RESULTS_DIR/normalized_d4_25200_synthetic_50400_prepended.csv" \
    --metrics_file "$RESULTS_DIR/normalized_d4_25200_synthetic_50400_metrics.csv"




sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_3150_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_6300_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_12600_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_25200_1d_config.json"

echo All daily configurations processed.