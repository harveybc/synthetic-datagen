# Synthetic Augmentation Protocol — `regime_residual_bootstrap_v1` (anti_mem_v1)

**LOCKED PACKET** — Phase 4 §4.1/§4.2/§4.3. Do not edit by hand;
regenerate via `examples/scripts/build_protocol_packet.py`.

- Asset: **ETHUSDT_4h**
- Schema version: `1.0.0`
- Built (UTC): `2026-05-03T16:37:37Z`
- Stage B status: **PENDING_APPROVAL**

## Generator
- family_id: `regime_residual_bootstrap_v1`
- family_revision: `anti_mem_v1`
- trainer plugin: `regime_residual_bootstrap_ohlcv_trainer`
- generator plugin: `regime_residual_bootstrap_ohlcv_generator`
- seed: `13`

### Anti-memorization params
- `anti_memorization` = `True`
- `anti_mem_window` = `32`
- `anti_mem_max_real_windows` = `4000`
- `anti_mem_dup_eps_quantile` = `0.001`
- `anti_mem_safety_margin` = `1.5`
- `anti_mem_boost_factor` = `1.0`
- `anti_mem_max_passes` = `16`

## Windows
- train_start: `2017-09-28 04:00:00`
- train_end: `2021-09-28 00:00:00`
- real_train_n_rows: `8749`
- synthetic_n_rows: `2190`
- heldout_boundary: **`2025-01-01 00:00:00`** (Stage C firewall)
- pre_stage_c_real_end: `2024-12-31 20:00:00`

### Downstream split template
- train_years: `5`
- val_years: `1`
- test_years: `1`
- split_anchor: `start`
- note: val + test windows must remain entirely real-data

## Gate table (Phase 4 §4.2)

| Gate | Threshold | Value | Pass |
|---|---|---|---|
| algebraic_violations | == 0 | 0 | ✅ |
| ks_returns_pvalue | > 0.01 | 0.325095 | ✅ |
| wasserstein_returns_ratio | < 1.5 | 0.0141817 | ✅ |
| classifier_auc_window_std | < 0.70 | 0.581525 | ✅ |
| nn_overlap_rate | < 1e-3 | 0 | ✅ |
| copied_subseq_ratio | < 0.50 | 0 | ✅ |
| duplicate_window_rate | < 1e-3 | 0 | ✅ |

`project3_valid_for_training` = **True**

## Input files
- `real_input_csv` — `examples/data/ethusdt_4h_full_8yr.csv`
  - sha256: `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`

## Output files
- `synthetic_ohlcv_csv` — `experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/synthetic_ohlcv.csv`
  - sha256: `f28ebbbe6f748a7496667382871cdba13fb949854a57a887eafdd8bc241cfdec`
- `augmented_tech_stat_csv` — `experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/ethusdt_4h_tech_stat_augmented_regime_residual_bootstrap.csv`
  - sha256: `377cb8f11b252def25729893f7eca9b19aa11363be64c457ec60e5c00c9e4946`
- `trainer_npz` — `experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/bootstrap.npz`
  - sha256: `fbe2ae82996f2cb7139b2ac325764a7fb36712196707913d1bdb88ceceb89351`

## Downstream training
- agent-multi config template: `examples/config/project3_ethusdt_4h_sac_synth_anti_mem_v1.json`
- do_not_run_until: **Stage B approval recorded in SYNTHETIC_LEDGER.csv**
- compare plan: `experiments/synthetic_data/project3_eth_4h/COMPARE_PLAN.md`
