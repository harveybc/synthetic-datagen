# synthetic-datagen

Self-contained synthetic **typical_price** timeseries generator with plugin architecture.

Trains generative models (VAE, GAN, VAE-GAN) on real EUR/USD typical_price data, then generates realistic but unpredictable synthetic timeseries — critical for [DOIN](https://github.com/harveybc/doin-core) verification.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train a VAE-GAN on real data
sdg --mode train --trainer vae_gan_trainer \
    --train_data examples/data/d1.csv examples/data/d2.csv examples/data/d3.csv \
    --save_model examples/models/generator.keras \
    --epochs 400 --latent_dim 16

# Generate synthetic data
sdg --mode generate \
    --load_model examples/models/generator.keras \
    --n_samples 5000 --seed 42 \
    --output_file synthetic_typical_price.csv

# Evaluate quality
sdg --mode evaluate \
    --synthetic_data synthetic_typical_price.csv \
    --real_data examples/data/d4.csv \
    --metrics_file metrics.json

# Optimize hyper-parameters via GA
sdg --mode optimize --trainer vae_gan_trainer \
    --train_data examples/data/d1.csv \
    --population_size 20 --n_generations 50
```

## Architecture

```
synthetic-datagen/
├── app/
│   ├── main.py              # Entry point & CLI dispatch
│   ├── cli.py               # Argument parsing
│   ├── config.py            # Default configuration
│   ├── data_processor.py    # Data loading, returns, windowing
│   └── plugin_loader.py     # Plugin discovery (entry_points)
├── sdg_plugins/
│   ├── trainer/
│   │   ├── vae_trainer.py        # Pure VAE
│   │   ├── gan_trainer.py        # Pure GAN
│   │   └── vae_gan_trainer.py    # VAE-GAN (recommended)
│   ├── generator/
│   │   └── typical_price_generator.py
│   ├── evaluator/
│   │   └── distribution_evaluator.py
│   └── optimizer/
│       └── ga_optimizer.py
├── examples/
│   ├── data/                # Real typical_price datasets (d1–d6)
│   ├── models/              # Trained models
│   └── config/              # Example JSON configs
└── tests/
```

## Operation Modes

| Mode | Description |
|------|-------------|
| **train** | Train a generative model on real typical_price CSVs |
| **generate** | Generate synthetic data from a trained model + seed |
| **evaluate** | Compare synthetic vs real data quality metrics |
| **optimize** | GA search for optimal hyper-parameters |

## Output Format

Matches predictor's expected input exactly:

```csv
DATE_TIME,typical_price
2020-01-01 00:00:00,1.3007625
2020-01-01 04:00:00,1.2966883333333332
```

## Key Design Decisions

- **Single feature**: typical_price only — no OHLC, no indicators
- **Self-contained**: trains AND generates — no dependency on feature-extractor
- **Returns-based**: models log-returns (stationary), reconstructs prices
- **Seed-deterministic**: same model + same seed = identical output
- **Plugin architecture**: all components replaceable via entry_points
- **Downsampling**: supports 1h→4h by averaging

## Reference Parameters (from MDSc phase_4_2)

| Parameter | Value |
|-----------|-------|
| window_size | 144 (24 days @ 4h) |
| batch_size | 128 |
| epochs | 400 |
| latent_dim | 16 |
| activation | tanh |
| kl_weight | 1e-3 |
| mmd_lambda | 1e-2 |
| use_returns | true |

## DOIN Integration

In DOIN evaluators, each gets a different seed derived from:
```
seed = hash(commitment + domain + evaluator_id + chain_tip_hash)
```
Same model + different seed = different but valid synthetic data → optimizer can't predict evaluation data.

## Tests

```bash
pytest tests/ -v
```

## License

MIT
