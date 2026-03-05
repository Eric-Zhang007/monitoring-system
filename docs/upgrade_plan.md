# Top50 Panel Upgrade Plan

## 1) Pipeline Architecture

```text
market_bars + feature_matrix_main
  -> update_universe_top50.py (stable top50 snapshot + hysteresis)
  -> build_training_cache.py (npz panel cache + global index + data_audit)
  -> LiquidPanelCacheDataset (cache-only, fail-fast if missing)
  -> LiquidModel (backbone + symbol embedding + horizon pooling + MoE + Student-t)
  -> compose_liquid_loss (Student-t NLL + quantile + direction + calibration + MoE regularizers)
  -> eval_liquid_top50.py (global + stratified + calibration + router report + gates)
  -> pack_artifact.py (manifest + weights + schema snapshots + cost snapshot + hash list)
  -> liquid_model_service / inference CLI
  -> v2_router decision scoring (edge/risk/band) + DecisionTrace
  -> FinancialAnalysisAgent (explain/evaluate/suggest, optional router hint)
```

## 1.1) RepoScout Mapping

| Topic | Existing touchpoint | Upgrade touchpoint |
|---|---|---|
| universe snapshot resolve/upsert | `backend/v2_repository.py` (`resolve_asset_universe_asof`, `upsert_asset_universe_snapshot`) | `training/universe/top50.py`, `scripts/update_universe_top50.py`, `scripts/update_universe_snapshot.py` |
| feature matrix source | `feature_matrix_main` (`scripts/merge_feature_views.py`) | `scripts/audit_offline_training_data.py`, `training/cache/panel_cache.py` |
| panel cache build/read | n/a (legacy direct DB samples) | `scripts/build_training_cache.py`, `training/cache/panel_cache.py`, `training/datasets/liquid_panel_cache_dataset.py` |
| training entry | `training/train_liquid.py` | `training/train_liquid.py` (readiness gating), `scripts/train_top50.py` (cache-first top50) |
| model backbone/head | `models/liquid_model.py`, `models/heads/dist_head.py` | same files + `models/symbol_embedding.py` |
| loss | `training/losses/trading_losses.py` | same file (Student-t + quantile + direction + MoE regs) |
| artifact/manifest loader | `mlops_artifacts/validate.py`, `backend/liquid_model_service.py` | `scripts/pack_artifact.py`, `backend/liquid_model_service.py` strict checks |
| inference/decision | `backend/liquid_model_service.py`, `backend/v2_router.py` | same files (distributional score + analyst output + DecisionTrace fields) |
| cost unification | `cost/cost_profile.py` | same file consumed by labels/eval/decision |
| execution safety | existing kill-switch/risk checks in router | `backend/execution_safety.py` + router preflight integration |
| audit trail | `backend/decision_trace/models.py` | same file + analyst/cost/router fields carried through |

## 2) Regime Features (As-Of Safe)

| feature | formula (as-of t) | mask rule |
|---|---|---|
| `realized_vol` | rolling std of `ret_1` over 24 bars | `ret_1` missing -> masked |
| `vol_of_vol` | rolling std of `vol_12` over 24 bars | `vol_12` missing -> masked |
| `return_skew_proxy` | rolling skew proxy of `ret_1` over 48 bars | `ret_1` missing -> masked |
| `spread_proxy` | `orderbook_spread_bps` | feature mask |
| `depth_proxy` | `log1p(orderbook_depth_total)` | feature mask |
| `imbalance_proxy` | `orderbook_imbalance` | feature mask |
| `funding_rate` | `funding_rate` | feature mask |
| `funding_zscore` | rolling zscore of funding over 96 bars | funding mask |
| `basis_proxy` | `basis_rate` | feature mask |
| `open_interest` | `open_interest` | feature mask |
| `oi_change` | first diff of `open_interest` | OI mask |
| `liquidation_proxy` | `max(0,-oi_change)*(1+spread_proxy)*(1+realized_vol)` | union mask |
| `trend_strength` | `0.6*ret_48 + 0.4*ret_288` | union mask |
| `text_coverage` | feature `text_coverage` | feature mask |
| `text_disagreement` | feature `text_disagreement` | feature mask |
| `event_density` | feature `event_count_1h` | feature mask |

As-of constraint: every training sample uses window `[t-lookback+1, t]`; labels use `t + horizon_step` only; cache builder hard-fails on any violation.

## 3) Step Order (Enforced)

1. Step1 throughput: cache-first panel dataset, no per-sample DB reads.
2. Step2 model thickening: symbol embedding + horizon pooling + heavy-tail output.
3. Step3 regime/MoE routing: 4 experts + top2 gating + collapse gate.

## 4) Cost & Execution Contract

Single module: `cost/cost_profile.py`

- shared by labels/training eval/decision
- outputs structured breakdown:
  - `fee_bps`
  - `slippage_bps`
  - `impact_bps`
  - `funding_bps`
  - `infra_bps`
  - `total_bps`
- infra baseline: `1.88 CNY/h` (configurable), converted to bps via notional or fallback hourly bps

Decision rule now uses:

`edge = E[ret] - cost`

`risk = max(sigma, q90-q10, eps)`

`score = edge / risk`

action by dynamic no-trade band (runtime configurable).

## 5) FinancialAnalysisAgent Constraints

- default mode: explain/evaluate/suggest only
- strict JSON schema (`FinancialAnalysisOutput`)
- optional router hint (`AGENT_ROUTER_HINT_ENABLED=1`):
  - normalized 0..1, sum=1
  - staleness check on `ts`
  - prompt/output hash stored in agent output
- no direct order generation

## 6) Runtime Config Policy

Hot reload (no restart):

- `SIGNAL_SCORE_ENTRY_BY_HORIZON`
- `SIGNAL_CONFIDENCE_MIN_BY_HORIZON`
- `SIGNAL_NO_TRADE_BAND_MULT`
- cost profile multipliers / lambdas (env + runtime config path)
- analyst/risk thresholds

Restart required (security-critical):

- API keys / secrets (`bitget` keys)
- primary DB connection string
- withdraw prohibition rules / exchange credentials

## 7) One-Click Commands

```bash
# 0) init (existing)
python3 scripts/build_feature_store.py --symbols "$LIQUID_SYMBOLS"

# 1) top50 universe snapshot
python3 scripts/update_universe_top50.py --as-of 2026-03-01T00:00:00Z
# alias
python3 scripts/update_universe_snapshot.py --as-of 2026-03-01T00:00:00Z --top-n 50 --rank-by volume_usd_30d

# 2) build training cache
python3 scripts/build_training_cache.py \
  --universe-snapshot artifacts/universe/liquid_top50_snapshot.json \
  --start 2025-01-01T00:00:00Z \
  --end 2026-02-28T23:55:00Z \
  --bar-size 5m \
  --lookback-len 96 \
  --output-dir artifacts/cache/liquid_top50

# 3) audit cache
python3 scripts/check_training_cache_audit.py --cache-dir artifacts/cache/liquid_top50

# 4) train panel model
python3 scripts/train_top50.py \
  --cache-dir artifacts/cache/liquid_top50 \
  --out-dir artifacts/models/liquid_main

# 5) evaluate + gates
python3 scripts/eval_top50.py \
  --artifact-dir artifacts/models/liquid_main \
  --cache-dir artifacts/cache/liquid_top50 \
  --universe-snapshot artifacts/universe/liquid_top50_snapshot.json \
  --out-dir artifacts/eval/top50_latest

# 6) pack strict artifact
python3 scripts/pack_artifact.py \
  --artifact-dir artifacts/models/liquid_main \
  --universe-snapshot artifacts/universe/liquid_top50_snapshot.json \
  --eval-dir artifacts/eval/top50_latest

# 7) inference + decision smoke
python3 scripts/run_inference_smoke.py \
  --artifact-dir artifacts/models/liquid_main \
  --cache-dir artifacts/cache/liquid_top50
python3 scripts/run_decision_smoke.py

# 8) short smoke bundle (Top50, 1 epoch)
bash scripts/smoke_train_top50.sh
```

## 8) Execution Safety (Following-style semantics)

- startup capability probe:
  - if plan order unsupported and stoploss mode is `trigger/plan`, switch to `local_guard`
  - optionally set `safe_mode`
- price feed degradation:
  - WS healthy -> normal mode
  - WS down + REST up while `local_guard` -> configurable action (`safe_mode` by default)
  - WS down + REST down -> force `safe_mode`
- local guard:
  - trigger hit -> protective close callback + `safe_mode`
- kill switch channels:
  - file + env + sqlite-state path, unified preflight
- unknown exchange positions:
  - any non-zero unknown symbol -> `safe_mode` + reason
