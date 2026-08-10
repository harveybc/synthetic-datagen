# synthetic-datagen

Plugin-driven synthetic financial time-series generator. The `sdg` CLI trains
generator models on real market data, generates synthetic series, optimizes
generator hyperparameters and evaluates synthetic quality — with an
**OHLCV-first** workflow: the primary generator families are stationary
bootstrap and regime-residual bootstrap over full open/high/low/close/volume
bars, with algebraic-consistency and stylized-facts evaluation. Legacy
VAE/GAN/VAE-GAN trainers and a typical-price generator remain available as
plugins. Built-in guardrails (forbidden-paths, held-out boundary, run ledger,
audit records) keep synthetic data out of evaluation sets and make every run
traceable.

## Status

**Active component** of the harveybc trading stack (package
`synthetic-datagen` 1.0.0). Successor of the legacy
[timeseries-gan](https://github.com/harveybc/timeseries-gan) repository.

## Role and non-responsibilities

`synthetic-datagen` produces synthetic OHLCV / typical-price datasets,
fitted generator artifacts and augmentation manifests for training-data
augmentation experiments.

It does **not**:

- train or serve predictive models — that is
  [predictor](https://github.com/harveybc/predictor) and
  [prediction_provider](https://github.com/harveybc/prediction_provider);
- engineer features or labels for real data — that is
  [feature-eng](https://github.com/harveybc/feature-eng);
- decide trades or run strategies;
- host distributed optimization — that is
  [doin-node](https://github.com/harveybc/doin-node) (which consumes sdg
  artifacts, see below).

## Architecture

[`app/main.py`](app/main.py) dispatches one of four modes — `train`,
`generate`, `optimize`, `evaluate` — over plugins resolved from nine
namespaced entry-point groups declared in [`pyproject.toml`](pyproject.toml):

| Group | Plugins |
|---|---|
| `sdg.trainer` | `stationary_bootstrap_ohlcv_trainer`, `regime_residual_bootstrap_ohlcv_trainer`, `vae_trainer`, `gan_trainer`, `vae_gan_trainer` — [`sdg_plugins/trainer/`](sdg_plugins/trainer) |
| `sdg.generator` | `stationary_bootstrap_ohlcv_generator`, `regime_residual_bootstrap_ohlcv_generator`, `typical_price_generator` — [`sdg_plugins/generator/`](sdg_plugins/generator) |
| `sdg.evaluator` | `ohlcv_algebraic_evaluator`, `financial_stylized_facts_evaluator`, `financial_distribution_evaluator`, `distribution_evaluator`, `predictive_evaluator`, `augmentation_evaluator`, `memorization_evaluator`, `augmentation_manifest_pipeline` — [`sdg_plugins/evaluator/`](sdg_plugins/evaluator) |
| `sdg.optimizer` | `ga_optimizer` (DEAP genetic search) — [`sdg_plugins/optimizer/`](sdg_plugins/optimizer) |
| `sdg.transformer` | `ohlcv_transformer` — [`sdg_plugins/transformer/`](sdg_plugins/transformer) |
| `sdg.reconstructor` | `ohlcv_reconstructor` — [`sdg_plugins/reconstructor/`](sdg_plugins/reconstructor) |
| `sdg.feature_engine` | `minimal_financial_feature_engine`, `tech_stat_feature_engine` — [`sdg_plugins/feature_engine/`](sdg_plugins/feature_engine) |
| `sdg.aggregator` | `ohlcv_timeframe_aggregator` — [`sdg_plugins/aggregator/`](sdg_plugins/aggregator) |
| `sdg.pipeline` | `augmentation_manifest_pipeline` — [`sdg_plugins/pipeline/`](sdg_plugins/pipeline) |

All groups are namespaced under `sdg.*`, so they cannot collide with other
repositories' entry-point groups. The OHLCV column schema lives in
[`sdg_plugins/schema/financial_ohlcv_schema.py`](sdg_plugins/schema/financial_ohlcv_schema.py).

### Guardrails

- [`app/forbidden_paths.py`](app/forbidden_paths.py) — refuses to open any
  input matching the globs in [`forbidden_paths.txt`](forbidden_paths.txt)
  (held-out evaluation data must never feed generator selection).
- [`app/heldout_guard.py`](app/heldout_guard.py) — enforces the rule that the
  generator never sees rows on or after the configured held-out boundary
  (`--heldout_boundary`, `reject_if_input_crosses_heldout`).
- [`app/synthetic_ledger.py`](app/synthetic_ledger.py) — append-only CSV
  ledger of every generator fit and generation run, keyed by
  `(generator_family_id, config_hash, seed, kind)`; default location
  `experiments/synthetic_data/SYNTHETIC_LEDGER.csv` (overridable via config or
  the `SDG_SYNTHETIC_LEDGER` environment variable).
- [`app/audit.py`](app/audit.py) — reproducibility record (config hash, input
  hash, plugin names, seed, git commit) attached to run metadata.

## Requirements

- Python **>= 3.10** (per [`pyproject.toml`](pyproject.toml)).
- Dependencies (from `pyproject.toml`): `numpy>=1.24`, `pandas>=2.0`,
  `tensorflow>=2.14`, `scipy>=1.11`, `scikit-learn>=1.3`, `deap>=1.4`;
  `pytest` via the `dev` extra.

## Installation

Unverified (not executed in a clean environment for this README):

```bash
git clone https://github.com/harveybc/synthetic-datagen.git
cd synthetic-datagen
pip install -e .[dev]
# installs the `sdg` console script
```

Verified in the maintainer environment (Python 3.12.13, 2026-08-10):

- `python -m app.main --help` → prints the full `sdg` usage.
- `python -m app.main --list_plugins` → prints the plugin registry for all
  groups (trainers, generators, evaluators, optimizers, ...).
- `python -c "import app.forbidden_paths, app.heldout_guard, app.synthetic_ledger, app.audit"`
  → `guardrail imports OK`.

## Quickstart

Fit a stationary-bootstrap OHLCV generator and produce a synthetic series
using the repo-owned config and sample data:

```bash
# Train + generate per the example config (writes model .npz, metadata and
# a synthetic CSV under examples/data/)
python -m app.main --load_config examples/config/financial_ohlcv_bootstrap_config.json
```

The config
[`examples/config/financial_ohlcv_bootstrap_config.json`](examples/config/financial_ohlcv_bootstrap_config.json)
trains on
[`examples/data/financial_ohlcv_sample.csv`](examples/data/financial_ohlcv_sample.csv)
and emits the artifacts committed next to it
([`financial_ohlcv_bootstrap.npz`](examples/data/financial_ohlcv_bootstrap.npz),
[`financial_ohlcv_synthetic.csv`](examples/data/financial_ohlcv_synthetic.csv),
metadata JSONs). This run was not re-executed for this README to avoid
overwriting the committed artifacts; the `--help`, `--list_plugins` and import
checks above were executed. A typical-price generation config is at
[`examples/config/generate.json`](examples/config/generate.json), and larger
drivers (generator sweeps, augmentation-manifest builds, protocol packets)
live in [`examples/scripts/`](examples/scripts).

Everything is also callable programmatically — plugins are plain classes
(`configure(...)` / `train(...)` / `generate(...)` / `evaluate(...)`), and the
CLI is a thin wrapper over them.

## Distributed / DOIN usage

`synthetic-datagen` itself runs locally. Its fitted generator artifacts are
consumed by [doin-node](https://github.com/harveybc/doin-node): predictor
node example configurations point at an sdg artifact root (`sdg_root`) and a
fitted generator model file to source augmentation data during distributed
optimization campaigns.

## Tests

```bash
python -m pytest -q --collect-only
```

Observed result (2026-08-10, Python 3.12.13): `71 tests collected, 3 errors` —
most of the suite under [`tests/`](tests) collects cleanly; three modules have
collection errors. Run the suite with `python -m pytest -q`.

## Outputs and reproducibility

- Fitted generator models (`--save_model`, e.g. `.npz` for bootstrap
  families, `.keras` for neural families) with metadata JSON
  (`--metadata_file` / `save_metadata`).
- Synthetic datasets (`--output_file`) with synthetic metadata
  (`synthetic_metadata_file`) carrying the audit record.
- Evaluation metrics JSON (`--metrics_file`).
- Append-only run ledger (default
  `experiments/synthetic_data/SYNTHETIC_LEDGER.csv`); committed experiment
  outputs live under [`experiments/synthetic_data/`](experiments/synthetic_data).
- Runs are seeded (`--seed`) and hashed (config + input hashes in the audit
  record), so a run is reproducible from its config and metadata.

## Safety and security

- No credentials are required or stored; all data paths are local.
- Leakage guardrails are on by default where configured: forbidden paths,
  held-out boundary rejection, and memorization evaluation
  (`memorization_evaluator`) for copy-detection.
- Synthetic data is for research and training augmentation. Nothing in this
  repository is financial advice.

## Limitations

- **No LICENSE file** is committed; [`pyproject.toml`](pyproject.toml)
  declares `license = MIT`, but until a LICENSE file is added the licensing
  is only declared in packaging metadata.
- No `requirements.txt`; dependencies are managed solely through
  `pyproject.toml`.
- Three test modules fail to collect (see Tests).
- Root-level one-off drivers (`run_*.py`, `measure_tolerance*.py`) are
  research scripts kept for reference, not part of the packaged surface.

## Migration notes

This repository supersedes
[timeseries-gan](https://github.com/harveybc/timeseries-gan) (package `tsg`):
the GAN/VAE experiments moved here as `sdg.trainer` plugins, and the current
recommended generators are the OHLCV bootstrap families. New work should
target `synthetic-datagen`.

## Related repositories

- [doin-node](https://github.com/harveybc/doin-node) — distributed optimizer
  runtime that consumes sdg generator artifacts.
- [predictor](https://github.com/harveybc/predictor) — model training that
  augmentation manifests target.
- [preprocessor](https://github.com/harveybc/preprocessor) /
  [feature-eng](https://github.com/harveybc/feature-eng) — the real-data
  pipeline whose datasets sdg augments.
- [timeseries-gan](https://github.com/harveybc/timeseries-gan) — legacy
  predecessor (superseded).

## License

MIT per [`pyproject.toml`](pyproject.toml); no LICENSE file is committed yet
(see Limitations).
