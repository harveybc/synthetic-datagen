# Project 3 — Synthetic Pretrain vs Real-Only Compare Plan

**Status:** PENDING_APPROVAL · do **not** launch agent-multi training
runs from this plan until the protocol packet
`regime_residual_bootstrap_v1_anti_mem_protocol.json` is signed and the
corresponding row in `SYNTHETIC_LEDGER.csv` is approved (Stage B).

This document specifies the exact, paired set of agent-multi runs that
will quantify the lift (or regression) introduced by pretraining on
synthetic ETHUSDT 4h data prior to fine-tuning on real data only.

## Scope and invariants

The two arms below differ **only** in the training data presented to
the SAC actor-critic. Every other hyperparameter — environment, action
space, reward, costs, splits, seeds — is held identical. Both arms use:

- env: `gym_fx_env`
- agent: `project3_sac_actor_critic_agent`
- pipeline: `rl_pipeline_with_validation`
- strategy: `direct_atr_sltp` (`k_sl=2.0`, `k_tp=3.0`, `atr_period=14`)
- reward: `pnl_reward`
- preprocessor: `feature_window_preprocessor` (`window_size=32`, `feature_scaling=rolling_zscore`, `feature_scaling_window=256`)
- broker: `default_broker` (`commission=0.0002`, `slippage=0.0`, `leverage=1.0`)
- action space: `continuous`, `continuous_action_threshold=0.1`
- position sizing: `notional`, `position_size=0.01`
- features: identical `tech_stat` preset (78 cols + binary mask)
- val/test windows: 100 % real data, never synthetic
- heldout boundary: 2025-01-01 00:00:00 (Stage C firewall enforced by trainer)
- replicate seeds: `train_seed ∈ {0, 1, 2}`, `eval_seed = train_seed`

## Arm A — real-only baseline

- config: `agent-multi/examples/config/project3_ethusdt_4h_sac_train_val_test_v2.json`
- input_data_file: `predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`
- splits: `train_years=4`, `val_years=1`, `test_years=1`, `split_anchor=start`
- pretraining: none
- artifacts: `agent-multi/examples/results/project3_ethusdt_4h_sac_train_val_test_v2/{policy.zip,summary.json}`

## Arm B — synthetic pretrain → real fine-tune

- config template (PENDING_APPROVAL): `agent-multi/examples/config/project3_ethusdt_4h_sac_synth_anti_mem_v1.json`
- generator family: `regime_residual_bootstrap_v1` (revision `anti_mem_v1`, generator seed=13)
- protocol packet: `experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/regime_residual_bootstrap_v1_anti_mem_protocol.json`

### B.1 — Pretrain on synthetic-augmented panel
- input_data_file: `synthetic-datagen/experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/ethusdt_4h_tech_stat_augmented_regime_residual_bootstrap.csv`
- splits: `train_years=5` (= 4 real + 1 synthetic prepended), `val_years=1`, `test_years=1`, `split_anchor=start`
- save policy: `./examples/results/project3_ethusdt_4h_sac_synth_anti_mem_v1/pretrain/policy.zip`

### B.2 — Fine-tune on real-only
- input_data_file: same file as Arm A
- splits: identical to Arm A
- load_pretrained_policy: artifact from B.1
- save policy: `./examples/results/project3_ethusdt_4h_sac_synth_anti_mem_v1/finetune/policy.zip`
- training budget: same `epoch_timesteps`, `max_epochs`, `l1_patience` as Arm A

## Metrics + decision rule

For each replicate seed, both arms produce a `summary.json` with
`train_*`, `val_*`, `test_*` blocks. The compare driver will report,
on the **test** split (real, post-train-end, pre-2025):

| Metric | Direction | Used for decision |
|---|---|---|
| `composite_score` | higher = better | primary |
| `sharpe` | higher = better | primary |
| `total_return_pct` | higher = better | secondary |
| `max_drawdown_pct` | lower = better | secondary |
| `n_trades` | sanity | reject if zero in either arm |

**Promotion rule:** Arm B is declared an improvement only if the
mean test-set `composite_score` exceeds Arm A by ≥1 standard error
across the 3 replicate seeds **and** the per-seed sign of the lift is
positive in ≥2/3 seeds. Otherwise the synthetic family stays at
diagnostic-only and is **not** propagated to Stage 4.3.

## What this plan does NOT authorize

- Running either Arm A or Arm B before Stage B approval.
- Modifying gate thresholds.
- Using the synthetic CSV in the val or test windows.
- Generating a second synthetic year (the protocol locks `synthetic_years=1`).
- Promoting the family to Stage 4.3 without recording the compare result
  back into `SYNTHETIC_LEDGER.csv`.
