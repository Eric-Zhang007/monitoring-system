# Strict Model Upgrade Progress

## Scope
- Single-path strict model system upgrade for liquid stack.
- Unified backbone interface with `patchtst` / `itransformer` / `tft`.
- Unified strict artifact chain and validation.
- Remove legacy/toy fallback branches in model pipeline.

## Progress Log
- [x] Read `模型.txt` and mapped required changes to current codebase.
- [x] Audited current train/infer/action/cost/NIM/test paths.
- [x] Implemented strict model architecture modules.
- [x] Rebuilt strict training pipeline (walk-forward + purged + dist losses + calibration).
- [x] Rebuilt strict inference service outputs (mu/sigma/quantiles + calibrated confidence).
- [x] Unified `cost_profile` across labels/oos/action.
- [x] Removed/disabled legacy trainer/router/json baseline paths.
- [x] Enforced NIM failure => missing features (no pseudo-embedding).
- [x] Added strict contract and integration tests.
- [x] Ran pytest suites and fixed regressions.

## Implemented Items (High-level)
- New strict model stack:
  - `models/outputs.py`
  - `models/backbones/{base,registry,patchtst,itransformer,tft}.py`
  - `models/{text_tower,quality_encoder,liquid_model}.py`
  - `models/heads/dist_head.py`
- New training stack:
  - `training/losses/liquid_losses.py`
  - `training/splits/walkforward_purged.py`
  - `training/metrics/liquid_metrics.py`
  - `training/calibration/calibrate.py`
  - Rewritten `training/train_liquid.py`
- Unified cost source:
  - `cost/cost_profile.py`
  - Updated `training/labels/liquid_labels.py`, `training/validation.py`, `backend/v2_router.py`
- Strict inference/artifact path:
  - Rewritten `backend/liquid_model_service.py`
  - Rewritten `inference/main.py`
  - Updated `backend/v2_repository.py` strict directory-manifest validation
- Legacy/toy branch cleanup:
  - `inference/model_router.py` disabled in strict-only mode
  - `training/liquid_model_trainer.py` disabled
  - `training/vc_model_trainer.py` disabled
  - `training/backbone_experiments.py` disabled
  - Removed `backend/models/*.json` old baseline/ensemble artifacts
- NIM behavior:
  - `backend/nim_integration.py` now returns missing on endpoint failure (no pseudo-embedding)
  - `scripts/merge_feature_views.py` updated to preserve text missing-mask semantics
- VC parity cleanup:
  - Added `vc/feature_spec.py`
  - Rewritten `training/train_vc.py` and `backend/vc_model_service.py`

## Tests Added/Updated
- Added:
  - `tests/test_model_output_contract.py`
  - `tests/test_backbone_registry.py`
  - `tests/test_dist_head_shapes.py`
  - `tests/test_patchtst_patchify.py`
  - `tests/test_itransformer_shapes.py`
  - `tests/test_tft_minimal.py`
  - `tests/test_gate_behavior.py`
  - `tests/test_walkforward_purged_split.py`
  - `tests/test_cost_profile_parity.py`
  - `tests/test_inference_no_heuristic_confidence.py`
  - `tests/test_nim_failure_results_in_missing.py`
  - `tests/test_strict_train_artifact_can_infer.py`
  - `tests/test_vc_train_infer_parity.py`
- Updated strict-only backend tests:
  - `backend/tests/test_model_router_core.py`
  - `backend/tests/test_model_router_residual_fusion.py`
  - `backend/tests/test_model_artifact_manifest_validation.py`
  - `backend/tests/test_liquid_model_service_artifact_fallback.py`

## Validation Results
- `pytest -q tests`
  - `13 passed, 8 skipped`
- `pytest -q backend/tests/test_model_artifact_manifest_validation.py backend/tests/test_model_router_core.py backend/tests/test_model_router_residual_fusion.py backend/tests/test_liquid_model_service_artifact_fallback.py`
  - `4 passed, 1 skipped`

## Active Issues
- Runtime environment currently has no `torch` installed for tests; torch-dependent tests are guarded with `importorskip` and show as skipped.
- Full backend test suite was not run in this pass; strict-path and touched tests were run.
