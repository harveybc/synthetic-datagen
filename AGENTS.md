# AGENTS.md — synthetic-datagen

## Project overview

`synthetic-datagen` generates synthetic financial time series. The `sdg` CLI
runs one of four modes — `train`, `generate`, `optimize`, `evaluate` — over
plugins resolved from nine namespaced `sdg.*` entry-point groups. The primary
generator families are stationary bootstrap and regime-residual bootstrap over
full OHLCV bars; legacy VAE/GAN/VAE-GAN trainers and a typical-price generator
are also registered. Evaluators check algebraic consistency of generated bars,
compare stylized facts against real data, and test for memorization.

It does not train or serve predictive models, does not engineer features or
labels for real data, does not decide trades, and does not host distributed
optimization. Guardrails (forbidden paths, held-out boundary, an append-only
run ledger and per-run audit records) exist to keep synthetic data out of
evaluation sets and make each run traceable.

**Runnability status: genuinely runnable.** Every command in the quickstart was
executed against a clean `git archive` checkout and produced the stated output.
The full train → generate → evaluate loop takes about two seconds.

## Agent quickstart (install → run → show the user results)

Verified on Python 3.12.13 from the repository root, in a clean checkout.

### 1. Environment

Requires Python >= 3.10 with `numpy>=1.24`, `pandas>=2.0`, `tensorflow>=2.14`,
`scipy>=1.11`, `scikit-learn>=1.3`, `deap>=1.4` (declared in
`pyproject.toml`; there is no `requirements.txt`).

**In a dedicated virtual environment**, install the package:

```bash
pip install -e .[dev]      # also installs the `sdg` console script and pytest
```

**In an environment shared with other projects, do not run that.**
`[tool.setuptools.packages.find]` includes `app*`, so installing publishes the
generic top-level package name `app`, which collides with the sibling repos in
this stack that also ship an `app` package. Generate only the local package
metadata instead — that is all plugin discovery needs:

```bash
python -c "import setuptools; setuptools.setup()" egg_info
```

This writes `synthetic_datagen.egg-info/` in the repository root. It is
matched by `*.egg-info/` in `.gitignore` and is not committed, so **a fresh
clone has no plugin metadata and every plugin group reports `(none
installed)`** until you run one of the two commands above. Verified: on a clean
checkout `python -m app.main --list_plugins` lists `(none installed)` for all
groups, and lists all plugins after the `egg_info` command.

### 2. Smoke test

```bash
python -m app.main --help
python -m app.main --list_plugins
python -c "import app.forbidden_paths, app.heldout_guard, app.synthetic_ledger, app.audit; print('guardrail imports OK')"
python -m pytest tests/ -q
```

`--list_plugins` prints five trainers, three generators, eight evaluators and
one optimizer. `python -m pytest tests/ -q` gives **71 passed** in about 27
seconds.

Note: `python -m pytest -q` without a path also picks up `examples/scripts/`
and reports `71 tests collected, 3 errors`. All three errors are in
`examples/scripts/` — two need the unlisted `hmmlearn` dependency, one is
missing a data file. Scope pytest to `tests/`.

### 3. Representative run — fit a bootstrap generator and produce a synthetic series

Uses the repo-owned config `examples/config/financial_ohlcv_bootstrap_config.json`,
which trains on `examples/data/financial_ohlcv_sample.csv` (600 hourly OHLCV
bars, columns `DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME`). Seed is 42, so the run
is reproducible.

```bash
# a) Fit — writes the model, its transformer, metadata and a run log
python -m app.main --load_config examples/config/financial_ohlcv_bootstrap_config.json

# b) Generate — the same config in generate mode, loading the model just fitted
python -m app.main --load_config examples/config/financial_ohlcv_bootstrap_config.json \
  --mode generate --load_model examples/data/financial_ohlcv_bootstrap.npz
```

Each step takes well under a second. Note that the config's `mode` is `train`:
step (a) alone does **not** write the synthetic CSV, despite the config
carrying `n_samples` and `output_file`. Step (b) is required.

Outputs, all under `examples/data/` and all gitignored:

| File | Contents |
|---|---|
| `financial_ohlcv_bootstrap.npz` | The fitted stationary-bootstrap model |
| `financial_ohlcv_bootstrap.transformer.json` | Column transform used to fit it |
| `financial_ohlcv_bootstrap.metadata.json` | Model metadata |
| `financial_ohlcv_synthetic.csv` | 600 synthetic OHLCV bars from `2024-01-01 00:00:00`, hourly |
| `financial_ohlcv_synthetic.metadata.json` | Audit record: config hash, input hash, plugin names, seed, git commit |
| `financial_ohlcv_run.log.json` | Debug log for the run |

Every run also appends a row to `experiments/synthetic_data/SYNTHETIC_LEDGER.csv`
(override with the `SDG_SYNTHETIC_LEDGER` environment variable).

### 4. Analytics step — evaluate the synthetic series against the real one

```bash
# Algebraic consistency of the generated bars
python -m app.main --mode evaluate --evaluator ohlcv_algebraic_evaluator \
  --synthetic_data examples/data/financial_ohlcv_synthetic.csv \
  --real_data examples/data/financial_ohlcv_sample.csv \
  --metrics_file examples/data/financial_ohlcv_metrics.json

# Stylized facts, synthetic vs real, side by side
python -m app.main --mode evaluate --evaluator financial_stylized_facts_evaluator \
  --synthetic_data examples/data/financial_ohlcv_synthetic.csv \
  --real_data examples/data/financial_ohlcv_sample.csv \
  --metrics_file examples/data/financial_ohlcv_stylized.json
```

The algebraic evaluator returned, verbatim:

```json
{
  "valid": true,
  "n_rows": 600,
  "violations": {
    "nan": 0, "inf": 0, "non_positive_ohlc": 0, "negative_volume": 0,
    "high_below_max_oc": 0, "low_above_min_oc": 0, "typical_price_mismatch": 0
  }
}
```

The stylized-facts evaluator writes `synthetic` and `real` blocks with matching
keys — `ret_mean`, `ret_std`, `ret_skew`, `ret_kurt`, return quantiles,
`acf_ret_lag1`, `acf_sqret_lag{1,5,20}`, `max_drawdown`, `mean_drawdown` and
volume statistics — so the two can be compared key by key.

### 5. Final message to the user

Report exactly this, with `<repo>` replaced by the absolute repository path:

> Done. The fit → generate → evaluate loop ran on the bundled example
> (`examples/data/financial_ohlcv_sample.csv`, 600 hourly OHLCV bars) in about
> two seconds. Results are in `<repo>/examples/data/` — all of these are
> generated and gitignored:
>
> - `<repo>/examples/data/financial_ohlcv_synthetic.csv` — 600 synthetic OHLCV
>   bars starting 2024-01-01, produced by the stationary-bootstrap generator.
> - `<repo>/examples/data/financial_ohlcv_bootstrap.npz` — the fitted model.
> - `<repo>/examples/data/financial_ohlcv_metrics.json` — algebraic checks:
>   `valid: true`, 600 rows, zero violations across all seven checks.
> - `<repo>/examples/data/financial_ohlcv_stylized.json` — stylized-facts
>   metrics for the synthetic and the real series, under matching keys.
> - `<repo>/examples/data/financial_ohlcv_synthetic.metadata.json` — the audit
>   record (config hash, input hash, seed, git commit).
> - `<repo>/experiments/synthetic_data/SYNTHETIC_LEDGER.csv` — one appended row
>   per run.
>
> There is no web UI; these are files on disk.
>
> **Analysis to try first:** open the stylized-facts JSON and compare the
> `synthetic` and `real` blocks key by key, starting with `ret_std`,
> `ret_kurt` and `acf_sqret_lag1`. A stationary bootstrap should reproduce the
> return standard deviation closely but tends to under-reproduce fat tails and
> volatility clustering, so a synthetic `ret_kurt` well below the real one, or
> a collapsed `acf_sqret_lag1`, tells you the block length needs raising
> (`block_length_mean`, currently 24) or that the regime-residual family is the
> better fit. Re-run step 3b with a different `--block_length_mean` and compare.

## Build, test and lint commands

```bash
# Package metadata (required for plugin discovery)
pip install -e .[dev]                                   # dedicated env
python -c "import setuptools; setuptools.setup()" egg_info   # shared env, no install

# CLI
python -m app.main --help
python -m app.main --list_plugins
sdg --help                       # only after pip install

# Tests
python -m pytest tests/ -q       # 71 passed
python -m pytest -q --collect-only   # 71 collected, 3 errors (all in examples/scripts/)
```

There is no linter, formatter or CI configuration in this repository.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | CLI entry; dispatches `train` / `generate` / `optimize` / `evaluate` |
| `app/plugin_loader.py` | Entry-point plugin resolution over the `sdg.*` groups |
| `app/config_handler.py`, `app/config_merger.py`, `app/cli.py` | Config load/save/merge and argument parsing |
| `app/forbidden_paths.py` | Refuses inputs matching the globs in `forbidden_paths.txt` |
| `app/heldout_guard.py` | Rejects inputs crossing the configured held-out boundary |
| `app/synthetic_ledger.py` | Append-only CSV ledger of every fit and generation run |
| `app/audit.py` | Reproducibility record: config hash, input hash, plugins, seed, git commit |
| `sdg_plugins/trainer/` | 5 trainers: two bootstrap OHLCV families, VAE, GAN, VAE-GAN |
| `sdg_plugins/generator/` | 3 generators: two bootstrap OHLCV families, typical price |
| `sdg_plugins/evaluator/` | 8 evaluators: algebraic, stylized facts, distribution, predictive, augmentation, memorization |
| `sdg_plugins/optimizer/` | `ga_optimizer`, a DEAP genetic search |
| `sdg_plugins/transformer/`, `sdg_plugins/reconstructor/` | OHLCV transform and inverse |
| `sdg_plugins/feature_engine/`, `sdg_plugins/aggregator/`, `sdg_plugins/pipeline/` | Feature engines, timeframe aggregation, augmentation-manifest pipeline |
| `sdg_plugins/schema/financial_ohlcv_schema.py` | The OHLCV column schema |
| `examples/config/` | 3 committed configs: bootstrap OHLCV, typical-price generate, VAE-GAN train |
| `examples/data/` | Committed sample data: `financial_ohlcv_sample.csv` (600 bars), `d1`–`d6.csv` (typical-price splits), `ethusdt_4h_full_8yr.csv` (18,085 ETH/USDT 4h bars with ~90 engineered columns) |
| `examples/scripts/` | Research drivers: generator sweeps, tolerance studies, protocol packets. Not part of the packaged surface, and three fail to import. |
| `experiments/` | Run ledger and experiment outputs. Gitignored. |
| `tests/` | The real test suite — 71 tests, all passing |
| `run_*.py`, `measure_tolerance*.py` at the root | One-off research drivers kept for reference |

## Conventions and constraints

- **Plugin architecture.** All nine entry-point groups are namespaced under
  `sdg.*` (`sdg.trainer`, `sdg.generator`, `sdg.evaluator`, `sdg.optimizer`,
  `sdg.transformer`, `sdg.reconstructor`, `sdg.feature_engine`,
  `sdg.aggregator`, `sdg.pipeline`), declared in `pyproject.toml`. The
  namespacing is deliberate: it prevents the group collisions that the
  predecessor repository suffers from. **Never register a plugin under an
  unqualified group name.**
- **Plugins are plain classes** with `configure(...)` / `train(...)` /
  `generate(...)` / `evaluate(...)`. The CLI is a thin wrapper, so everything
  is callable programmatically.
- **OHLCV data contract.** `DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME`, defined in
  `sdg_plugins/schema/financial_ohlcv_schema.py`. Column names are
  configurable per run (`datetime_column`, `open_col`, …). Bars must satisfy
  `high >= max(open, close)`, `low <= min(open, close)`, positive prices and
  non-negative volume — `ohlcv_algebraic_evaluator` checks exactly this.
- **Config-driven.** Everything is expressible as JSON passed to
  `--load_config`; CLI flags override the file. `--save_config` persists the
  effective configuration.
- **Leakage guardrails are the point of this repository.** `forbidden_paths.txt`
  lists globs (2025 held-out windows, Stage C) that the generator and any
  selection-time evaluator must never load. `--heldout_boundary` plus
  `reject_if_input_crosses_heldout` stops the generator seeing rows on or after
  the boundary. Do not disable, relax or work around either.
- **Reproducibility.** Runs are seeded (`--seed`) and hashed. The audit record
  and the ledger row are the trace; a run should be reproducible from its
  config plus its metadata.
- **Modes are explicit.** `train` fits and saves a model; `generate` requires
  `--load_model`; `evaluate` requires `--synthetic_data`; `optimize` requires
  `--train_data`. A `train` config that also carries `n_samples` and
  `output_file` still will not generate.
- No credentials are required or stored; all data paths are local.

## Do not touch

- **`examples/data/*.csv` that ship with the repository** —
  `financial_ohlcv_sample.csv`, `d1.csv`–`d6.csv` and
  `ethusdt_4h_full_8yr.csv` are committed input fixtures. Read them; never
  overwrite, trim or regenerate them. The generated `financial_ohlcv_*`
  artifacts in the same directory are gitignored and safe to overwrite.
- **`forbidden_paths.txt` and the held-out boundary settings.** Widening either
  one silently contaminates model selection. Treat changes here as a
  correctness bug, not a configuration tweak.
- **`experiments/synthetic_data/SYNTHETIC_LEDGER.csv`** — append-only by
  design. Do not edit or truncate it; redirect it with `SDG_SYNTHETIC_LEDGER`
  if you need a throwaway ledger.
- **`examples/results/`** — committed outputs of past experiments.
- **Other repositories.** Nothing here writes outside this repository.
- **A shared Python environment.** Do not `pip install -e .` into an
  environment shared with the sibling repositories — `setup`'s package
  discovery publishes the generic top-level name `app`. Use the `egg_info`
  command from step 1 instead, or a dedicated virtual environment.
- **`synthetic_datagen.egg-info/`** — generated, gitignored. Regenerate it;
  never edit or commit it.
