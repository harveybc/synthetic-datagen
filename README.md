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

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart** section end to end: set up the environment, run the smoke test, execute the example bootstrap fit-and-generate run, then tell me the exact file paths where I can see the results and one analysis I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by most coding agents.

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

```bash
git clone https://github.com/harveybc/synthetic-datagen.git
cd synthetic-datagen
pip install -e .[dev]
# installs the `sdg` console script
```

**Plugin discovery needs the package metadata.** Plugins are resolved through
`importlib.metadata` entry points, and `*.egg-info/` is gitignored, so a fresh
clone reports `(none installed)` for every plugin group until the package is
installed. If you cannot install into your environment — `packages.find`
includes `app*`, which publishes the generic top-level name `app` and collides
with the sibling repositories in this stack — generate only the local metadata
instead:

```bash
python -c "import setuptools; setuptools.setup()" egg_info
```

Verified on a clean `git archive` checkout (Python 3.12.13, 2026-08-16):

- Before the `egg_info` command, `python -m app.main --list_plugins` prints
  `(none installed)` for all groups; after it, the full registry appears.
- `python -m app.main --help` → prints the full `sdg` usage.
- `python -c "import app.forbidden_paths, app.heldout_guard, app.synthetic_ledger, app.audit"`
  → `guardrail imports OK`.
- The full train → generate → evaluate loop below runs in about two seconds.

`pip install -e .[dev]` itself was not executed for this README.

## Quickstart

Fit a stationary-bootstrap OHLCV generator and produce a synthetic series
using the repo-owned config and sample data:

```bash
# 1. Fit the generator (the config's mode is "train")
python -m app.main --load_config examples/config/financial_ohlcv_bootstrap_config.json

# 2. Generate the synthetic series from the model just fitted
python -m app.main --load_config examples/config/financial_ohlcv_bootstrap_config.json \
  --mode generate --load_model examples/data/financial_ohlcv_bootstrap.npz

# 3. Check the generated bars are algebraically consistent
python -m app.main --mode evaluate --evaluator ohlcv_algebraic_evaluator \
  --synthetic_data examples/data/financial_ohlcv_synthetic.csv \
  --real_data examples/data/financial_ohlcv_sample.csv \
  --metrics_file examples/data/financial_ohlcv_metrics.json
```

The config
[`examples/config/financial_ohlcv_bootstrap_config.json`](examples/config/financial_ohlcv_bootstrap_config.json)
trains on
[`examples/data/financial_ohlcv_sample.csv`](examples/data/financial_ohlcv_sample.csv)
(600 hourly OHLCV bars) and writes generated artifacts next to it
(`examples/data/financial_ohlcv_bootstrap.npz`,
`examples/data/financial_ohlcv_synthetic.csv`, metadata JSONs — generated
outputs, gitignored, not committed). Note that step 1 alone does **not** write
the synthetic CSV even though the config carries `n_samples` and
`output_file`: the config's `mode` is `train`, and generation is a separate
run. All three steps were executed on a clean checkout; step 3 returned
`valid: true` over 600 rows with zero violations. A typical-price generation
config is at
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
python -m pytest tests/ -q
```

Observed result (2026-08-16, Python 3.12.13): **71 passed** in about 27
seconds. Running `python -m pytest -q` without a path also collects
[`examples/scripts/`](examples/scripts) and reports `71 tests collected, 3
errors`; all three errors are in that directory — two need the unlisted
`hmmlearn` dependency and one is missing a data file. Scope pytest to
[`tests/`](tests).

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
  `pyproject.toml`. `hmmlearn`, needed by two `examples/scripts/` modules, is
  not declared anywhere.
- A fresh clone has no plugins until the package metadata is generated (see
  Installation) — nothing in the repository states this at the point of use.
- Three modules under [`examples/scripts/`](examples/scripts) fail to collect
  (see Tests).
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
