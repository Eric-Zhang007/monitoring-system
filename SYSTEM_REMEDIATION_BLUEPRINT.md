# Quant System Implementation Spec (Real-Data First, Model Upgrade Path Included)

## 1. Scope and Objective

It defines exact engineering changes required to solve the current system issues:

1. `prod` vs `maintenance` data contamination in gating
2. Sharpe metric distortion and poor statistical comparability
3. Overfitting risk from short-window tuning
4. Weak data-source breadth and source reliability issues
5. Rule-heavy strategy path with insufficient model-driven alpha
6. Inefficient hyperparameter search (serial grid)
7. Missing hard linkage from information quality to model decision value

Target state:
- Gate decisions are based on real data only (`run_source=prod`, `score_source=model`)
- Offline/online dual-track is explicit and auditable
- Model path is executable: linear/tree baseline -> neural network -> ensemble -> staged rollout
- HPO uses parallel Bayesian optimization (Optuna), not serial brute-force grid by default

---

## 2. Non-Negotiable Principles (Must Not Be Changed)

1. Real-data gate principle
- Release gate can only use `prod + model` runs.
- `maintenance/smoke/async_test` are monitor-only.

2. Reproducibility principle
- Same data + same config + same code => same outcome (within numerical tolerance).
- No hidden env-only behavior in core backtest logic.

3. Anti-overfit principle
- No single-window optimize-and-claim.
- Must include Search/Validation/Forward-OOS segmentation.

4. Metric transparency principle
- Sharpe method must be disclosed (`daily_agg` vs `step_raw`).
- Do not report only favorable metric variants.

5. Data lineage principle
- Every training and inference sample must be traceable to source, timestamp, mapping method, and quality score.

6. Risk-first principle
- Profit improvement cannot break hard risk constraints (max drawdown, reject rate, execution constraints).

7. Production safety principle
- New model/strategy requires shadow/canary and rollback path.

8. Compliance principle
- Public information only. No insider data.

---

## 3. Current System Diagnosis (Repository-Verified)

## 3.1 Real-data vs maintenance mixing
Observed:
- `backtest_runs` has substantial `maintenance` samples in recent windows.
- Gate scripts historically defaulted to `prod,maintenance` mixed filtering.

Impact:
- Gate outcome not purely representative of live market regime.

## 3.2 Sharpe distortion
Observed:
- Backtest engine computes step-level Sharpe with high-frequency annualization (`sqrt(24*365)`).

Impact:
- Numerical magnitude is unstable and easily exaggerated.

## 3.3 Rule-heavy strategy path
Observed:
- Core execution of signal -> sizing -> turnover/cost is still rule-centric in `backend/v2_router.py`.
- Current liquid artifact path often resolves to lightweight tabular/json model.

Impact:
- Model contribution is not isolated cleanly from heuristic policy behavior.

## 3.4 Data-source weakness
Observed:
- GDELT connector frequently fails with 429.
- Macro FRED feed is often empty.
- Source concentration remains high on limited channels.

Impact:
- Feature diversity and freshness robustness are insufficient for stable alpha generation.

## 3.5 Tuning inefficiency
Observed:
- Default tuning scripts are mostly serial grid loops.

Impact:
- Poor exploration efficiency, high wall-clock, higher chance of regime-fitting.

---

## 4. Architecture Target: Offline/Online Dual Track

## 4.1 Offline track (research and training)
Responsibilities:
- Data curation and quality filtering
- Feature generation and schema enforcement
- Model training and OOS validation
- Hyperparameter optimization

Outputs:
- Versioned artifacts (`model`, `normalization`, `manifest`)
- Validation reports with fixed windows and lineage IDs

## 4.2 Online track (inference and execution)
Responsibilities:
- Real-time feature computation with strict schema
- Model inference with fallback accounting
- Risk-constrained order generation
- Monitoring and drift detection

Outputs:
- Predictions, decisions, execution records, quality metrics

## 4.3 Offline/online alignment contract
Required shared fields:
- `feature_payload_schema_version`
- `feature_version`
- `data_version`
- `model_name`, `model_version`
- `lineage_id`

No promotion allowed if schema/version mismatch exists.

---

## 5. Concrete Implementation Plan

## Track A: Real-Data Gate Enforcement

### A1. Default filters (already changed, keep locked)
Files:
- `backend/v2_router.py`
- `scripts/evaluate_hard_metrics.py`
- `scripts/check_backtest_paper_parity.py`
- `docker-compose.yml`
- `scripts/daily_phase63_maintenance.sh`

Behavior:
- Include sources default: `prod`
- Exclude sources default: `smoke,async_test,maintenance`

### A2. Add `data_regime` to run config/metrics
Files:
- `backend/schemas_v2.py`
- `backend/v2_router.py`

Changes:
- Extend backtest config with `data_regime` enum:
  - `prod_live`
  - `maintenance_replay`
  - `mixed`
- Write this field into `backtest_runs.config` and reflected in gate scripts.

Acceptance:
- Gate command output includes `data_regime` and rejects non-`prod_live` by default.

---

## Track B: Metric Method Hardening (Sharpe and observation adequacy)

### B1. Introduce dual Sharpe outputs
Files:
- `backend/v2_router.py`
- `scripts/evaluate_hard_metrics.py`

Changes:
- In backtest metrics output, add:
  - `sharpe_step_raw`
  - `sharpe_daily`
  - `sharpe_method="daily_agg_v1"`
  - `observation_days`
  - `vol_floor_applied`
- Gate script must evaluate `sharpe_daily` only.

### B2. Daily aggregation implementation
Implementation rule:
- Aggregate per-step pnl into UTC day buckets.
- Compute Sharpe on daily returns with annual factor `sqrt(365)`.
- `step_raw` kept only for diagnostics.

### B3. Observation adequacy check
Gate conditions become:
- `completed_runs >= min_completed_runs`
- `observation_days >= min_observation_days`
- `data_regime == prod_live`

Acceptance:
- Insufficient days/runs must return `status=insufficient_observation`.

---

## Track C: Data Source Expansion and Reliability

### C1. Connector reliability framework
Files:
- `collector/connectors/gdelt.py`
- `collector/connectors/macro_fred.py`
- `collector/collector.py`
- `backend/metrics.py`

Changes:
- Add connector-level retry with exponential backoff + jitter.
- Add source health counters:
  - success/failure
  - empty_result_count
  - rate_limit_count
  - fetch_latency
- Add cooldown/circuit breaker per source on repeated failures.

### C2. Source tiering and quality gating
Files:
- `collector/collector.py`
- `training/feature_pipeline.py`
- `inference/main.py`

Changes:
- Attach source quality tier and confidence in event payload.
- Feature pipeline adds weighted event aggregation by tier.
- Low-quality events can be excluded from training via config flag.

### C3. Latency-first ingestion objective
SLO targets:
- `P95(source_publish_to_ingest) < 120s` for primary crypto feeds
- Source success rate > 95% for primary connectors

Acceptance:
- Dashboard must show per-source freshness and error profile.

---

## Track D: Strategy-to-Model Upgrade Path (Executable, staged)

## D0. Baseline freeze (current state)
Baseline model classes:
- Tabular linear/LightGBM artifacts (`liquid_*_lgbm_baseline_v2.json`)
- Optional TSMixer checkpoints (`liquid_*_tsmixer_v2.pt`)

Requirement:
- Freeze baseline as control arm for all future comparisons.

## D1. Strong tabular stage (linear/tree done correctly)
Files:
- `training/liquid_model_trainer.py`
- `training/validation.py`
- `inference/model_router.py`

Implementation:
- Replace ad-hoc fallback weighting with explicit model registry entries.
- Train LightGBM with purged K-fold + walk-forward reporting.
- Persist full training metadata:
  - data range
  - feature schema
  - fold metrics
  - seed

Promotion rule:
- Tree model must outperform frozen baseline across validation and forward OOS.

## D2. Neural stage (TSMixer) with teacher-student distillation
Files:
- `training/liquid_model_trainer.py`
- `inference/model_router.py`
- `backend/models/*` manifest format

Implementation:
- Teacher: validated tabular model predictions.
- Student: TSMixer on sequence features.
- Distillation objective: `MSE(student, true)` + `lambda * MSE(student, teacher)`.
- Save checkpoint with mandatory fields:
  - `type=tsmixer_liquid`
  - `normalization`
  - `n_tokens`, `n_channels`
  - `ensemble_alpha`
  - `train_report_hash`

Promotion rule:
- NN-only and Ensemble both evaluated.
- Accept ensemble only if it improves net pnl-after-cost and does not degrade max drawdown constraint.

## D3. Ensemble stage and rollout
Files:
- `inference/model_router.py`
- `backend/v2_router.py`
- `backend/v2_repository.py`

Implementation:
- Explicit ensemble policy versions:
  - `tabular_only`
  - `nn_only`
  - `blend_alpha_x`
- Store chosen policy in backtest config and prediction explanations.
- Rollout strategy:
  - shadow -> 10% canary -> staged increases

Rollback triggers:
- Consecutive gate failures
- Drift threshold exceeded
- Execution reject/slippage anomalies

---

## Track E: Hyperparameter Optimization Upgrade (Parallel + Optuna)

### E1. New HPO script
Add file:
- `scripts/optuna_liquid_hpo.py`

Key design:
- Sampler: `TPESampler`
- Pruner: `MedianPruner` (or ASHA equivalent)
- Storage: SQLite (`artifacts/hpo/optuna_liquid.db`)
- Parallel workers: default `max(1, cpu_count()-2)`

Objective (multi-term scalarized):
- maximize `pnl_after_cost`
- penalize `max_drawdown`
- penalize `reject_rate`
- include `turnover` penalty

### E2. Three-stage optimization protocol
1. Stage 1: coarse search (single symbol, short window)
2. Stage 2: candidate refinement (multi-symbol, medium window)
3. Stage 3: forward OOS confirmation (long window, prod-only)

Only stage-3 winners can become gate candidates.

### E3. Incremental persistence
- Each trial writes one JSONL line immediately.
- Long runs must be resumable after interruption.

Acceptance:
- HPO runs saturate `n-2` cores.
- Time-to-top-k significantly better than serial grid baseline.

---

## Track F: Information Processing Pipeline (Not just ingestion)

### F1. Event processing layers
1. Ingest (raw)
2. Normalize (type/time/source)
3. Entity-link (symbol/company/macro index)
4. Dedup cluster
5. Quality scoring
6. Feature projection
7. Decision attribution

### F2. Quality-to-decision accountability
Required outputs per prediction:
- top contributing events
- source tiers
- feature contribution summary
- missingness flags

### F3. LLM integration boundary
Allowed:
- extraction, classification, summarization, entity normalization
Not allowed:
- direct order decisioning

LLM output must be converted into deterministic structured features before model use.

---

## 6. Macro/Micro Error Register and Fixes

## Macro errors
1. Mixed regime gating -> fixed by prod-only defaults and `data_regime`
2. Metric method mismatch -> fix via dual Sharpe + daily gate
3. Search/validation leakage -> fix via strict 3-stage windows
4. Source concentration risk -> fix via connector reliability + source tiering

## Micro errors
1. Sizing override inconsistency
- Issue: `_score_to_size()` reads env directly; may ignore request-specific overrides.
- Fix: pass effective sizing config object through full backtest stack.

2. Synthetic event link contamination
- Issue: broad symbol fallback can inflate coverage with weak semantics.
- Fix: mark synthetic links and exclude from training by default.

3. Artifact validity too weak
- Issue: file-parsable != scientifically valid.
- Fix: mandatory artifact manifest fields + signed eval metadata.

4. Tuning script robustness
- Issue: serial loops, delayed output, interruption losses.
- Fix: incremental persistence + resume + Optuna parallelization.

---

## 7. Deliverables Checklist (Implementation)

1. Code changes
- [ ] Add `data_regime` fields and filters
- [ ] Add `sharpe_daily` pipeline and gate switch
- [ ] Add connector health metrics and circuit-breakers
- [ ] Add strict artifact manifest validation
- [ ] Add Optuna HPO script and runner docs
- [ ] Refactor sizing config propagation to remove hidden env dependency

2. Data/ops changes
- [ ] Prod-only gate dashboard
- [ ] Source-health dashboard
- [ ] OOS-only promotion report

3. Tests
- [ ] Unit tests for Sharpe daily aggregation
- [ ] Unit tests for source filter strictness (`prod` only)
- [ ] Unit tests for sizing override determinism
- [ ] Integration tests for HPO resume and parallel workers
- [ ] Regression tests for artifact validation rules

---

## 8. Acceptance Criteria

The upgrade is accepted only if all are true:

1. Gate purity
- Gate statistics contain only `run_source=prod` and `score_source=model`.

2. Metric reliability
- `sharpe_daily` replaces raw-step Sharpe as gate input.
- Insufficient observation returns explicit status.

3. Model upgrade path execution
- Tabular baseline -> NN -> Ensemble completed with tracked OOS comparisons.
- Ensemble promoted only with measurable net improvement.

4. Search efficiency
- HPO parallel runtime materially outperforms serial grid.
- Search pipeline is resumable and interruption-safe.

5. Data value proof
- Feature variants with added information demonstrate positive OOS contribution versus baseline.

---

## 9. Immediate Next Sprint (Execution Order)

Sprint 1 (high priority):
1. Implement Sharpe daily gate method + observation adequacy checks
2. Refactor sizing config propagation deterministically
3. Add strict artifact manifest validation

Sprint 2:
1. Connector reliability/circuit-breaker + source health metrics
2. Optuna parallel HPO implementation
3. Stage-structured (Search/Validation/OOS) promotion logic

Sprint 3:
1. Distillation-based TSMixer training path hardening
2. Ensemble policy rollout with shadow/canary and rollback automation
3. Information-quality contribution reporting into decision pipeline

