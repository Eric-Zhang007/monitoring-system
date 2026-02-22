# Production Gap Debug Progress

Date: 2026-02-22

## Scope
- Validate and fix reported gaps:
  - `risk_events` schema gap (P0)
  - runtime import path dependency on test-only `sys.path` injection (P1)
  - `merge_feature_views.py` dim-mismatch silent-rebuild behavior (P1)
  - PatchTST fixed positional capacity (P2)

## Status
- [x] Verify `risk_events` table gap is real in `scripts/init_db.sql`.
- [x] Verify import-path dependency is real (`import backend.v2_router` fails without injected backend path).
- [x] Fix P0: add `risk_events` DDL into `scripts/init_db.sql`.
- [x] Fix P1: add explicit bootstrap for runtime import path normalization.
- [x] Fix P1: add dim-mismatch gate + audit output in `scripts/merge_feature_views.py`.
- [x] Fix P2: PatchTST positional capacity fallback (no hard fail at >1024 patches).
- [x] Run targeted pytest and import checks.

## Notes
- Confirmed: `backend/repository_modules/execution_store.py::save_risk_event()` directly writes to `risk_events`.
- Confirmed: `scripts/init_db.sql` currently does not define `risk_events`.
- Confirmed: with only repo root on `sys.path`, `import backend.v2_router` failed (`ModuleNotFoundError: schemas_v2`).
- Implemented:
  - `scripts/init_db.sql`: added `risk_events` table + indexes.
  - `backend/bootstrap.py` + `backend/__init__.py`: runtime path bootstrap for root/backend/inference/training.
  - `scripts/merge_feature_views.py`:
    - added `--dim-mismatch-max-ratio` gate (env: `MERGE_DIM_MISMATCH_MAX_RATIO`, default `0.02`);
    - added `--audit-path` (env: `MERGE_AUDIT_PATH`, default `artifacts/audit/merge_feature_views_audit.json`);
    - emits mismatch metrics in output json and fails hard when ratio exceeds threshold.
  - `models/backbones/patchtst.py`: >1024 patch windows now use sinusoidal positional fallback instead of raising `patchtst_positional_capacity_exceeded`.
- Added tests:
  - `tests/test_backend_bootstrap_imports.py`
  - `tests/test_merge_feature_views_dim_gate.py`
  - `tests/test_patchtst_positional_fallback.py`
- Validation:
  - targeted new tests: `3 passed`
  - execution+model regression set: `50 passed, 1 warning`
