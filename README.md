# synthetic-datagen

Self-contained synthetic **typical_price** timeseries generator with plugin architecture.

Trains generative models (VAE, GAN, VAE-GAN) on real EUR/USD typical_price data, then generates realistic but unpredictable synthetic timeseries — critical for [DOIN](https://github.com/harveybc/doin-core) verification.

## Programmatic API (Plugin-First)

All plugins have clean programmatic APIs — the CLI is just a wrapper.

```python
# Train
from sdg_plugins.trainer.vae_gan_trainer import VaeGanTrainer
trainer = VaeGanTrainer()
trainer.configure({"window_size": 144, "latent_dim": 16, "epochs": 400, ...})
trainer.train(train_data=["d1.csv", "d2.csv"], save_model="model.keras")

# Generate (e.g. from DOIN evaluator)
from sdg_plugins.generator.typical_price_generator import TypicalPriceGenerator
gen = TypicalPriceGenerator()
gen.load_model("model.keras")
df = gen.generate(seed=42, n_samples=5000)
# → DataFrame with DATE_TIME, typical_price columns

# Evaluate (THE metric — predictive utility from MDSc thesis phase 4)
from sdg_plugins.evaluator.predictive_evaluator import PredictiveEvaluator
ev = PredictiveEvaluator()
ev.configure({"window_size": 144, "eval_epochs": 50})
result = ev.evaluate(
    synthetic=synthetic_df,
    real_train=train_df,     # d4
    real_val=val_df,         # d5
    real_test=test_df,       # d6
)
# result["mae_delta_test"] < 0  → synthetic data HELPS prediction
# result["synthetic_helps_test"] = True/False

# Secondary distribution metrics
from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
dist_ev = DistributionEvaluator()
metrics = dist_ev.evaluate(synthetic=synthetic_df, real=real_df)
```

## CLI Quick Start

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

# Evaluate: does synthetic data improve prediction? (thesis phase 4)
sdg --mode evaluate \
    --synthetic_data synthetic_typical_price.csv \
    --real_train examples/data/d4.csv \
    --real_val examples/data/d5.csv \
    --real_test examples/data/d6.csv \
    --metrics_file metrics.json

# Optional: use external predictor repo for evaluation
sdg --mode evaluate \
    --synthetic_data synthetic_typical_price.csv \
    --real_train examples/data/d4.csv \
    --real_val examples/data/d5.csv \
    --real_test examples/data/d6.csv \
    --predictor_dir /home/openclaw/predictor \
    --metrics_file metrics.json

# Secondary: distribution metrics only
sdg --mode evaluate --evaluator distribution_evaluator \
    --synthetic_data synthetic_typical_price.csv \
    --real_data examples/data/d4.csv \
    --metrics_file dist_metrics.json

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
| **evaluate** | Predictive utility test: does synthetic data improve prediction? |
| **optimize** | GA search for optimal hyper-parameters |

## Output Format

Matches predictor's expected input exactly:

```csv
DATE_TIME,typical_price
2020-01-01 00:00:00,1.3007625
2020-01-01 04:00:00,1.2966883333333332
```

## Evaluation Methodology (MDSc Thesis Phase 4)

The **real test** of synthetic data quality: does it improve prediction?

```
Step 1: Train predictor on real d4          → MAE on d5, d6 (baseline)
Step 2: Prepend synthetic data to d4        → train same predictor
Step 3: Measure MAE on same d5, d6          → (augmented)
Step 4: Compare: delta = augmented - baseline
        If delta < 0 → synthetic data HELPS → good generator
        If delta > 0 → synthetic data HURTS → bad generator
```

This is THE metric. Distribution similarity (KL, Wasserstein) is secondary.

Two evaluator backends:
- **Built-in** (default): lightweight LSTM predictor, fast, good for iteration
- **External**: runs Harvey's full predictor repo as subprocess, authoritative

## Key Design Decisions

- **Single feature**: typical_price only — no OHLC, no indicators
- **Self-contained**: trains AND generates — no dependency on feature-extractor
- **Returns-based**: models log-returns (stationary), reconstructs prices
- **Seed-deterministic**: same model + same seed = identical output
- **Plugin architecture**: all components replaceable via entry_points
- **4h periodicity**: trains on and outputs 4h interval data directly

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

---

## Phase 4 — Project 3 SAC Augmentation Workflow (ETHUSDT 4h)

This repo ships a fully-wired Phase 4 augmentation pipeline that takes
the 8-year ETHUSDT 4h dataset, generates synthetic OHLCV bars, recomputes
the full `tech_stat` feature matrix, and emits a `model_ready.csv`
suitable for direct consumption by [agent-multi](https://github.com/harveybc/agent-multi)'s
`feature_window_preprocessor` → `project3_sac_actor_critic_agent`.

### Data layout

The 8-year reference dataset is provided at:

- [examples/data/ethusdt_4h_full_8yr.csv](examples/data/ethusdt_4h_full_8yr.csv) — `2017-09-28 → 2025-12-31`, 18,086 rows, full `tech_stat` feature matrix.

The agent-multi default split (anchored at `start`) is:

| Window | Span | n_rows |
|---|---|---|
| Train | 2017-09-28 → 2021-09-28 | 8,749 |
| Validation | 2021-09-28 → 2022-09-28 | 2,190 |
| Test | 2022-09-28 → 2023-09-28 | 2,190 |
| **Heldout (Phase 3 firewall)** | **2023-09-28 → 2025-12-31** | **4,957** |

Phase 4 generators may NEVER see the validation, test, or heldout windows
(see [forbidden_paths.txt](forbidden_paths.txt) for enforcement).

### One-command end-to-end augmentation

```bash
python -m examples.scripts.build_augmented_project3_training \
    --method stationary_bootstrap \
    --synthetic_years 1 \
    --output_dir experiments/synthetic_data/project3_eth_4h
```

Pipeline:

1. Slices the 4-year training window from the 8-year file.
2. Trains `stationary_bootstrap_ohlcv_trainer` (Politis-Romano stationary block bootstrap) on the OHLCV columns only.
3. Generates 1 year (≈2,190 bars) of synthetic 4h OHLCV with timestamps **preceding** the real-data start, so the agent-multi temporal split does not need any modification beyond `train_years += synthetic_years`.
4. Validates with `OhlcvAlgebraicEvaluator` (must report 0 violations).
5. Runs `FinancialDistributionEvaluator` + `MemorizationEvaluator` and writes their gate dicts.
6. Concatenates synthetic + real OHLCV and recomputes the full `tech_stat` feature matrix via `TechStatFeatureEngine` — Phase 4 §6 forbids generating indicators directly.
7. Emits `ethusdt_4h_tech_stat_augmented_<method>.csv` plus an audit summary and a row appended to `SYNTHETIC_LEDGER.csv`.

### Wiring into agent-multi

The augmented CSV plugs straight into the agent-multi config — only three keys change vs. the real-only baseline:

```jsonc
{
  "input_data_file": ".../ethusdt_4h_tech_stat_augmented_stationary_bootstrap.csv",
  "train_years": 5,        // was 4 — absorbs the 1 year of synthetic
  "val_years": 1,
  "test_years": 1
  // val + test windows still map to 100% real data
}
```

### Available Phase-4-ready generators

| Generator family | Status | Validity-by-construction | Memorization gates (4yr ETH 4h) |
|---|---|---|---|
| `stationary_bootstrap_ohlcv_generator` | ⚠ Diagnostic-only | ✅ | **FAILS** `duplicate_window_rate=0.025`, `nn_overlap_rate=0.010`, `copied_subseq_ratio=2.469` |
| `regime_residual_bootstrap_ohlcv_generator` | ⚠ Diagnostic-only | ✅ | 6/7 gates pass; **fails only** `duplicate_window_rate=0.015` (vs 0.001 max). Strictly better than stationary on every gate (`copied_subseq_ratio: 2.469 → 0.000`, `nn_overlap_rate: 0.010 → 0.000`, `duplicate_window_rate: 0.025 → 0.015`) |
| `typical_price_generator` (legacy) | ⚠ Not OHLCV | ❌ | n/a |
| `block_bootstrap`, `regime_*` legacy, `grasynda`, `timegan`, `vae_gan` | ❌ Not Phase-4 plugin yet | n/a | n/a |

**Algorithmic note** — `regime_residual_bootstrap_v1` quantile-bins the
training window into K=3 volatility regimes (rolling `|r_close|`),
stores regime-mean-removed residuals, walks a Markov chain over the
regimes, and at each step produces

    Z_syn[t] = mean[regime_t] + residual[idx_t] + N(0, σ · std[regime_t])

The continuous Gaussian jitter is what eliminates the `copied_subseq`
runs and pushes `nn_overlap_rate` to zero; calibrated `σ=0.20` keeps
`KS_returns p > 0.01`. CLI:

```bash
python -m examples.scripts.build_augmented_project3_training \
    --method regime_residual_bootstrap --synthetic_years 1 --seed 42
```

### Quality gates fail closed (2026-05-03 addendum)

Per the SYNTHETIC_DATAGEN_SPECKIT 2026-05-03 addendum, all three gate
families (algebraic, distribution, memorization) are **fatal by default**.
If any gate fails:

- `augmentation_summary.json` records `project3_valid_for_training = false`.
- `SYNTHETIC_LEDGER.csv` gets an `evaluate` row with `valid=false`.
- The augmented `model_ready` CSV is **not written**, and any pre-existing
  augmented CSV at the expected path is renamed with the suffix
  `.invalid_quality_gates`.
- `build_augmented_project3_training.py` exits with code 3.

A failed generator may be re-run with `--allow_diagnostic_output` purely
to produce diagnostic artifacts; that mode must never feed Project 3
SAC/PPO/DQN training. The first ETH 4h `stationary_bootstrap_v1` output
is currently in this state.

### Audit trail

Every fit/generate/evaluate appends one row to
`experiments/synthetic_data/SYNTHETIC_LEDGER.csv`. Every generator family
is recorded in `generator_family_registry.json`. Each family also gets a
`<family>_protocol.md` stub documenting its mathematical assumptions, the
fit windows used, and the §4.2 gate values it produced.
