# Production Readiness Checklist (No-Docker + No-GPU Runtime)

This checklist is for production deployments that run non-training paths only on target nodes where Docker is unavailable.

WSL constraint: run all commands from WSL Bash (Linux shell). PowerShell and `cmd.exe` paths are out of scope.

## 1. Environment Baseline

- `python3`, `git`, `screen`, `awk`, `sed`, `df`, `curl` are available on target host.
- Disk capacity meets runbook minimum (`MIN_DISK_GB`, default `20`).
- Memory capacity meets runbook minimum (`MIN_MEM_GB`, default `8`).
- `REQUIRE_GPU` is **not** set to `1` for no-GPU nodes (`MIN_GPU_COUNT=0` by default).
- `DATABASE_URL` and `REDIS_URL` point to reachable production services.
- Alembic migration can reach `head` on the target database.
- `.env` is created from `.env.example` and kept out of version control.

## 2. Security Baseline (No-Docker)

- No placeholder secrets are used (`change_me_please`, `REPLACE_WITH_*`, default `admin` credentials).
- Backend CORS uses explicit allowlist via `CORS_ALLOW_ORIGINS`.
- `CORS_ALLOW_CREDENTIALS=1` is only used with explicit origins (never wildcard `*`).
- Sensitive runtime values are loaded from `.env` for `server_nodocker_up.sh`.
- Security static validation passes:

```bash
bash scripts/validate_security_hardening.sh
```

## 3. Data + Control Plane Availability

- PostgreSQL connectivity is healthy (`monitoring/health_check.py`).
- Redis connectivity is healthy (`monitoring/health_check.py`).
- Backend `/health` endpoint is reachable.
- Backend `/metrics` endpoint is reachable (recommended).
- Backend `/api/v2/risk/limits` endpoint is reachable.
- Optional analytics tables (`predictions_v2`, `semantic_features`) are not required for basic health to pass; warnings are acceptable if intentionally absent.

## 4. Background Workers + Schedulers

- `screen` sessions exist: `backend`, `collector`, `task_worker`, `model_ops`.
- `task_worker` status transitions are observed: `queued -> running -> completed|failed` via async task API.
- `model_ops_scheduler` process is running on schedule.
- Task worker logs show no persistent loop crashes/restarts.
- `TASK_WORKER_ERROR_BACKOFF_BASE_SEC` / `TASK_WORKER_ERROR_BACKOFF_MAX_SEC` are set appropriately for Redis/network turbulence.

## 5. API and Functional Smoke

- Core non-training API paths execute successfully (`/api/v2/*` risk/signal/execution/backtest endpoints as applicable).
- At least one async task (`/api/v2/tasks/pnl-attribution`) completes end-to-end.
- Kill-switch and risk hard-block paths behave as expected under threshold breaches.
- Metrics for API, execution, and queue latency are emitted.

## 6. Alerting + Observability

- Prometheus scrape status is healthy for backend + collector.
- Alertmanager receives firing alerts in test drill.
- Grafana dashboards load and display current metrics.
- Health-check SLO output includes latency + availability status.

## 7. Rollout and Ops Controls

- Rollout state is initialized for active tracks (`liquid`/`vc` as used).
- Rollback checks are enabled and auditable.
- `scripts/continuous_remediation_loop.py` gates are configured to block on hard failures.
- Incident runbook + oncall ownership for this service is documented.

## 8. Preflight Command (No-GPU)

Run before release cutover:

```bash
bash scripts/server_preflight_nodocker.sh
```

Optional strict mode (requires live backend probe):

```bash
REQUIRE_LIVE_API=1 bash scripts/server_preflight_nodocker.sh
```

## 9. Runtime Readiness Gate (No-Docker)

Run after `server_nodocker_up.sh`:

```bash
bash scripts/server_readiness_nodocker.sh
```

Optional strict collector metrics check:

```bash
REQUIRE_COLLECTOR_METRICS=1 bash scripts/server_readiness_nodocker.sh
```

Long-running async jobs can use stall-aware readiness polling knobs:

```bash
TASK_MAX_WAIT_SEC=1800 TASK_STALL_TIMEOUT_SEC=300 bash scripts/server_readiness_nodocker.sh
```
