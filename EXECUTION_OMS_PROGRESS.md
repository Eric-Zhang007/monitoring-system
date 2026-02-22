# Execution OMS Upgrade Progress

## Scope
- One-shot production OMS refactor (strict path only; no gray, no fallback).
- Unified adapter contract and OMS-driven execution lifecycle.
- Full DB traceability for decision/parent/child/fill/reconciliation/positions.

## Progress
- [x] Read `执行.txt` and mapped current gaps.
- [x] Audited existing execution router/engine/repository/tests.
- [x] Add execution domain models + FSM + contract tests.
- [x] Add DB schema for execution_decisions/child_orders/fills/reconciliation/positions_live.
- [x] Add OMS repository module and wire `execution_store`/`v2_repository`.
- [x] Implement execution planner (marketable_limit + passive_twap).
- [x] Refactor ExecutionEngine to OMS `run_decision` pipeline.
- [x] Add adapter base interface and conform paper/coinbase/bitget implementations.
- [x] Upgrade `/execution/orders` + `/execution/run` to full OMS outputs.
- [x] Add reconciliation daemon + position accounting.
- [x] Add adapter contract + paper E2E integration tests.
- [x] Run pytest and fix regressions.
- [x] Re-check `执行.txt` item by item against current code.
- [x] Re-run OMS + strict-model combined E2E validation.

## Active Issues
- No open execution blocker in current pass.
- Residual warning: strict model tests emit one upstream torch nested-tensor warning (non-failing).

## Validation Results (Latest)
- `pytest -q tests/test_execution_fsm.py tests/test_execution_models_schema.py tests/test_execution_adapter_contract.py tests/test_execution_e2e_paper.py backend/tests/test_execution_engine_paths.py backend/tests/test_execution_reject_realism.py backend/tests/test_bitget_adapter.py backend/tests/test_execution_policy_context.py backend/tests/test_execution_request_validation.py backend/tests/test_paper_trading_execution_events.py`
  - `22 passed`
- Combined OMS + strict-model chain:
  - `pytest -q tests/test_execution_fsm.py tests/test_execution_models_schema.py tests/test_execution_adapter_contract.py tests/test_execution_e2e_paper.py backend/tests/test_execution_engine_paths.py backend/tests/test_execution_reject_realism.py backend/tests/test_bitget_adapter.py backend/tests/test_execution_policy_context.py backend/tests/test_execution_request_validation.py backend/tests/test_paper_trading_execution_events.py tests/test_model_output_contract.py tests/test_backbone_registry.py tests/test_dist_head_shapes.py tests/test_patchtst_patchify.py tests/test_itransformer_shapes.py tests/test_tft_minimal.py tests/test_gate_behavior.py tests/test_walkforward_purged_split.py tests/test_cost_profile_parity.py tests/test_inference_no_heuristic_confidence.py tests/test_nim_failure_results_in_missing.py tests/test_strict_train_artifact_can_infer.py tests/test_vc_train_infer_parity.py backend/tests/test_model_artifact_manifest_validation.py backend/tests/test_model_router_core.py backend/tests/test_model_router_residual_fusion.py backend/tests/test_liquid_model_service_artifact_fallback.py`
  - `46 passed, 1 warning`
