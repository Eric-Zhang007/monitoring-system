# AUDIT_REPORT

Generated at: 2026-03-01 (UTC)
Scope: current workspace state (`monitoring-system`) after strict static audit + targeted test spot checks.
Method: LeadAgent + multi-agent audit (Repo/Data/Model/Exec/Agent/QA).

## 1) Chain Mapping (file -> function -> role)

| Chain | Primary touchpoints |
|---|---|
| Schema / contract | `schema/liquid_feature_schema.yaml`, `schema/codegen_feature_contract.py::render_contract/main`, `features/feature_contract.py` |
| Feature build | `scripts/build_feature_store.py::main` -> `feature_snapshots_main` |
| View merge | `scripts/merge_feature_views.py::main` -> `feature_matrix_main` |
| Top universe | `training/universe/top50.py::build_top_universe_snapshot`, `scripts/update_universe_top50.py::main`, `backend/v2_repository.py::resolve_asset_universe_asof/upsert_asset_universe_snapshot` |
| Offline readiness audit | `scripts/audit_offline_training_data.py::run_audit/main` |
| Cache build (panel) | `training/cache/panel_cache.py::build_training_cache_from_db`, `scripts/build_training_cache.py::main` |
| Train (legacy/main) | `training/train_liquid.py::main` + `training/datasets/liquid_sequence_dataset.py::load_training_samples` |
| Train (top50 panel) | `scripts/train_top50.py::main` + `training/datasets/liquid_panel_cache_dataset.py` |
| Loss | `training/losses/trading_losses.py::compose_liquid_loss` |
| Splits | `training/splits/walkforward_purged.py::build_walkforward_purged_splits` |
| Artifact pack/validate | `mlops_artifacts/pack.py::pack_model_artifact`, `mlops_artifacts/validate.py::validate_manifest_dir`, `scripts/pack_artifact.py` |
| Inference CLI | `inference/main.py::main`, `inference/feature_reader.py::fetch_sequence` |
| Runtime model service | `backend/liquid_model_service.py::predict_with_context/_predict_sequence` |
| Router/decision | `backend/v2_router.py::_build_liquid_prediction`, `_liquid_multi_horizon_signal`, `generate_signal`, `run_execution` |
| Cost unified module | `cost/cost_profile.py` + `training/labels/liquid_labels.py` + `backend/v2_router.py` |
| Execution/risk/account | `backend/execution_engine.py`, `backend/risk_manager.py`, `backend/execution_style_selector.py`, `backend/account_state/aggregator.py`, `monitoring/reconciliation_daemon.py` |
| Analyst agent | `backend/financial_analysis_agent.py`, `backend/v2_router.py::generate_signal`, `backend/decision_trace/models.py` |

## 2) Requirement Matrix (Present / Partial / Missing)

### 2.1 User checklist (a-j)

| Item | Status | Evidence | Gap summary |
|---|---|---|---|
| (a) Top50 universe stable + reproducible snapshot + configurable | **Partial** | `training/universe/top50.py`, `scripts/update_universe_top50.py`, `asset_universe_snapshots` APIs | Snapshot machinery exists; hysteresis behavior currently weak (selection order issue) and replay determinism needs tighter ordering/constraints. |
| (b) Cross-asset features | **Missing** | schema/build pipeline mostly single-symbol fields | No explicit market-index/relative-strength/beta/correlation factors in main schema/build path. |
| (c) Cross-asset shared model params | **Partial** | top50 path uses shared model + `symbol_id`; legacy path mixes symbols but often without `symbol_id`/regime inputs | Main training entry still underuses symbol/regime context. |
| (d) Walk-forward/purged | **Partial** | `training/train_liquid.py` uses walk-forward purged; top50 pipeline still simple split | Top50 production chain should also use walk-forward/purged gate by default. |
| (e) Trading-friendly loss (fat-tail+direction+quantile+uncertainty+smoothness) | **Partial** | Student-t + quantile + direction + calibration + router regs present | Missing explicit temporal smoothness/turnover surrogate penalties in composite loss. |
| (f) Unified cost for label/runtime/backtest | **Present** | `cost/cost_profile.py` used by labels, eval, router decision | Need preserve hash-level gate in all eval/pack flows (mostly present). |
| (g) Stable decision output (anti-churn) | **Partial** | dynamic no-trade band + position sizing penalty + turnover limits | Missing persisted prediction smoother (stateful EWMA/Kalman) in common online/offline path. |
| (h) Execution guards (SL/ws->rest/kill switch/health/reconcile) | **Partial** | kill switch + reconcile + risk events present; safety controller implemented | `execution_safety` hooks not fully wired into production run_execution/startup; healthz/readyz gate incomplete. |
| (i) Multi-process/multi-account isolation | **Partial** | process manager and account control APIs exist | execution/account write path still has weak `account_id` isolation in core order lifecycle. |
| (j) Financial analyst agent with optional risk-only overlay | **Partial** | deterministic agent + router hint constraints + trace fields present | schema strictness (`extra=forbid`) and replay assertions/tests incomplete; default path is on-chain explain path (not fully off-path). |

### 2.2 Hard constraints compliance snapshot

| Constraint | Status | Notes |
|---|---|---|
| strict-only, no silent fallback | **Partial** | core model/artifact/schema checks fail-fast; some non-critical paths still carry compatibility/degraded patterns. |
| single source of truth = `schema/liquid_feature_schema.yaml` | **Partial** | core build/merge/infer uses contract; additional legacy scripts still contain local list logic. |
| no pad/truncate; missing via mask only | **Partial** | primary sequence uses mask; some legacy/aux paths may still rely on implicit zero semantics. |
| inference reads full `feature_matrix_main` sequence only | **Present** (primary liquid path) | `features/sequence.py` + `inference/feature_reader.py` fetch full sequence. |
| synthetic default excluded from main train | **Partial** | upstream build has `allow-synthetic=false`; downstream hard gate/filter coverage incomplete. |
| core logic changes always with tests/gates | **Partial** | good baseline tests exist; missing tests around some new safety/agent strictness edges. |

## 3) RepoScout touchpoint mapping (for modification)

| Area | File(s) | Function(s) | Planned action |
|---|---|---|---|
| Universe hysteresis | `training/universe/top50.py` | `build_top_universe_snapshot` | Fix stable retention ordering/selection semantics. |
| Offline audit snapshot/event logic | `scripts/audit_offline_training_data.py` | `_resolve_symbols_from_snapshot`, `run_audit` | Support `symbols_json` object/list formats; fix event coverage query by schema-safe join. |
| Top50 cache gate | `scripts/build_training_cache.py`, `scripts/run_top50_upgrade_pipeline.sh`, `scripts/smoke_train_top50.sh` | `main` | Add readiness input and BLOCKED exclusion handshake. |
| Model smoothness loss | `training/losses/trading_losses.py`, `scripts/train_top50.py`, `training/train_liquid.py` | `compose_liquid_loss` + train calls | Add temporal/horizon smoothness regularization and wire weights. |
| Top50 calibration | `scripts/train_top50.py` | `main` | Fit and persist calibration bundle to ckpt. |
| Inference parity for symbol/regime | `inference/main.py` | `_predict`/`main` | Pass `symbol_id` and regime features when available. |
| Analyst strict schema/tests | `backend/financial_analysis_agent.py`, `tests/test_analyst_agent_schema.py`, new tests | model validators | enforce strict schema + add exception/isolation/reduce-only tests. |
| Execution safety wiring | `backend/v2_router.py` | `run_execution` (and startup path if safe) | Hook preflight/unknown-position/safe-mode blocking in main execution path. |

## 4) Minimal change strategy (non-rewrite)

1. **Do not rewrite Present items**: keep current strict artifact validation, cost unification, walk-forward module, model output schema.
2. **Patch Partial items in-place**: universe hysteresis, readiness gating, top50 training calibration, smoothness penalty, execution safety wiring, analyst schema strictness.
3. **Add tests first-class with each patch**: unit + at least one smoke/e2e gate update.
4. **Fail-fast over fallback**: any missing critical artifact/schema/cache/config should raise explicit runtime error.

## 5) Current blockers / environment notes

- Full `scripts/run_strict_e2e_acceptance.sh` in this environment may fail at DB auth (`DATABASE_URL` credentials) unless provided.
- Code-level tests for targeted modules are runnable and will be used as acceptance for incremental patches.

## 6) Implemented In This Round (delta)

### 6.1 Fixed / Enhanced

- Universe hysteresis retention bug fixed in `training/universe/top50.py` (prior members within keep-rank now retained before fill).
- Offline readiness audit improved in `scripts/audit_offline_training_data.py`:
  - supports snapshot `symbols_json` as list/object;
  - event coverage SQL now schema-aware (`events.symbol` or `events + event_links + entities`).
- Cache build now enforces readiness handshake and BLOCKED exclusion in `scripts/build_training_cache.py` (fail-fast when readiness file missing and exclusion enabled).
- Training losses upgraded with smoothness regularizers in `training/losses/trading_losses.py`:
  - `horizon_smoothness_regularizer`
  - `vol_monotonic_regularizer`
  and wired into `training/train_liquid.py` + `scripts/train_top50.py`.
- Main training/inference paths now pass symbol/regime context into model:
  - `training/datasets/liquid_sequence_dataset.py`
  - `training/train_liquid.py`
  - `inference/main.py`
- FinancialAnalysisAgent schema hardening and risk-only bound:
  - strict extra-field rejection (`extra=forbid`)
  - `pos_scale` constrained to `[0,1]`
  in `backend/financial_analysis_agent.py`.
- `backend/v2_router.py` execution safety path strengthened:
  - connectivity-safe-mode guard for live adapter path,
  - unknown-position guard before execution.
- One-click scripts hardened for import stability:
  - `scripts/run_strict_e2e_acceptance.sh`
  - `scripts/smoke_train_top50.sh`
  - `scripts/run_top50_upgrade_pipeline.sh`
  now inject repo-root `PYTHONPATH`.
- Direct CLI entry stability improved via repo-root bootstrap in:
  - `scripts/update_universe_top50.py`
  - `scripts/build_training_cache.py`
  - `scripts/train_top50.py`
  - `scripts/eval_liquid_top50.py`
  - `scripts/pack_artifact.py`
  - `scripts/run_inference_smoke.py`
  - `scripts/run_decision_smoke.py`
  - `training/train_liquid.py`
  - `inference/main.py`

### 6.2 Tests Added / Updated

- Added:
  - `tests/test_universe_hysteresis_effective.py`
  - `tests/test_build_training_cache_blocked_filter.py`
  - `tests/test_loss_smoothness_regularizers.py`
- Updated:
  - `tests/test_training_loss_outputs_calibratable.py`
  - `tests/test_analyst_agent_schema.py`
  - `backend/tests/test_multi_horizon_signal_logic.py`

### 6.3 Verification Results

- Targeted pytest batches: **23 passed**.
- `scripts/smoke_train_top50.sh`: import-path issue resolved; current first blocker is external DB auth (`psycopg2 OperationalError: fe_sendauth: no password supplied`).
- `scripts/run_strict_e2e_acceptance.sh`: first blocker remains external DB auth in current environment.
