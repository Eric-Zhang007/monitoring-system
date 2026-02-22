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

## Active Issues
- Local env lacks `torch`, so `backend/tests/test_v2_router_core.py` was not re-run in this pass.
- Added a root `metrics.py` shim and backend metrics fallback to avoid `training.metrics` import shadowing in tests.
