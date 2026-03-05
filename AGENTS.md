# Agent Runbook (Repository Local)

This file defines local execution conventions for long-running and API-heavy tasks.

## Long-running jobs

- Always use a stable background launch mode for multi-hour tasks.
- Preferred method: `scripts/run_bg_task.sh start <name> -- <command ...>`.
- Every background task must have:
  - persistent log file under `artifacts/runtime/bg/logs/`
  - pid file under `artifacts/runtime/bg/pids/`
  - explicit status check before launching a duplicate task
- Never rely on transient interactive sessions as the only runtime holder for long jobs.

## Concurrency and rate limits

- For API-heavy ingestion/backfill, parallelism must be bounded by configuration (for example `--workers`, `INGEST_WORKERS`).
- Do not increase worker count automatically after failures.
- Respect upstream limits: keep retries with backoff and avoid bursty re-requests.
- If two heavy jobs contend for the same resources (DB/API/CPU), run them serially unless explicitly requested otherwise.

## Operational defaults

- Run one heavy data backfill at a time by default.
- Keep model training/e2e validation out of the critical data-backfill window unless the user asks for overlap.
- On any critical dependency failure, fail fast with a clear error; do not silently downgrade behavior.
