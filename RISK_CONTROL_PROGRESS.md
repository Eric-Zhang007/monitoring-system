# Risk Control Upgrade Progress

Date: 2026-02-23

## Scope
- Execute `风控.txt` plan in strict mode.
- Add account-state-driven risk pipeline and ensure end-to-end runnable path.
- Maintain completion checklist and unresolved issues.

## Checklist
- [x] Read `风控.txt` and map requirements to current codebase.
- [x] Commit 1: contracts for `AccountState` / `RiskState` / `DecisionTrace` + contract tests.
- [x] Commit 2: DB tables + repository store for account/risk/decision trace.
- [x] Commit 3: `AccountStateAggregator` + account-state daemon.
- [x] Commit 4: reconciliation RED trigger test coverage.
- [x] Commit 5: `RiskManager` hard/soft evaluation + tests.
- [x] Commit 6: `PositionSizer` deterministic target sizing + monotonic tests.
- [x] Commit 7: execution style selector + volatility switch tests.
- [x] Commit 8: wire signal pipeline to account/risk/sizer/style + decision trace persistence.
- [x] Commit 9: training loss module alignment (`trading_losses`) and calibration compatibility checks.
- [x] Commit 10: e2e scripts for account-state and ablation matrix.
- [x] Commit 11: env switches and defaults.
- [x] Commit 12: run end-to-end tests and verify pass.
- [ ] Push all local changes to remote.

## Open Issues
- Live DB credentials are not available in this environment; `scripts/run_account_state_e2e.sh` now prints an explicit skip message for live-db daemon checks and still runs strict pytest e2e subset.
- Existing `backend/tests/test_v2_router_core.py::test_default_model_dir_prefers_repo_path_in_runtime` fails because `_default_model_dir` is absent (pre-existing, outside this risk-control scope).

## Verification
- `pytest -q tests/test_account_state_contract.py tests/test_decision_trace_contract.py tests/test_account_state_ttl_enforced.py tests/test_risk_manager_soft_penalties.py tests/test_soft_penalty_does_not_freeze.py tests/test_position_sizer_monotonic.py tests/test_execution_style_selector.py tests/test_vol_spike_switches_to_marketable.py tests/test_signal_pipeline_uses_account_state.py tests/test_reconciliation_triggers_red.py tests/test_recon_failure_triggers_kill_switch.py tests/test_training_loss_outputs_calibratable.py tests/test_execution_adapter_contract.py tests/test_execution_e2e_paper.py` -> `23 passed`.
- `./scripts/run_risk_ablation_matrix.sh` -> pass for A/B/C matrix.
- `./scripts/run_account_state_e2e.sh` -> pass (with explicit DB-skip message in current env).
