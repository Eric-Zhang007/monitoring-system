# Ingestion Progress (Single Source of Truth)

Last updated: **2026-03-05T02:52:00Z**

## Active Jobs
- Running merge: `scripts/merge_feature_views_chunked.sh`
  - current child window: `2021-05-31T00:00:00Z -> 2021-06-07T00:00:00Z`
  - progress over target window `2018-01-01 -> 2026-03-04`: **41.76%**
- Live awaiter (persistent session): `session_id=79254`
  - log file: `artifacts/runtime/bg/logs/readiness_after_merge_active_live_20260305T024406Z.log`
  - behavior: wait merge done -> run readiness audit -> write `artifacts/audit/top50_data_readiness_latest.json`

## Verified Facts (No Guessing)
- `feature_snapshots_main` is complete for top50 panel:
  - rows: `14,704,788`
  - distinct symbols: `50`
  - range: `2019-07-15 09:00:00+08` to `2026-03-02 11:10:00+08`
- `market_bars` has all top50 symbols across main timeframes:
  - `1m / 3m / 5m / 15m / 30m / 1h / 4h / 1d`
  - missing symbols per timeframe: `0`

## In Progress
- `feature_matrix_main` is being rebuilt by chunked merge.
- Until merge reaches latest horizon, distinct symbol count inside `feature_matrix_main` will rise progressively (expected behavior because many symbols are listed later in history).

## Next Automatic Step
1. Merge process exits.
2. Awaiter auto-runs:
   - `python scripts/audit_offline_training_data.py --track liquid --as-of 2026-03-04T00:00:00Z --top-n 50 --start 2018-01-01T00:00:00Z --end 2026-03-04T00:00:00Z --output artifacts/audit/top50_data_readiness_latest.json`
3. Audit report becomes the training gate input.
