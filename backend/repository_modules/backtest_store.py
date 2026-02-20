from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def create_backtest_run(connector, run_name: str, track: str, config: Dict[str, Any], run_source: str = "prod") -> int:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO backtest_runs (run_name, track, run_source, started_at, metrics, config, created_at)
                VALUES (%s, %s, %s, NOW(), %s, %s, NOW())
                RETURNING id
                """,
                (run_name, track, run_source, json.dumps({"status": "running"}), json.dumps(config)),
            )
            return cur.fetchone()["id"]


def finish_backtest_run(connector, run_id: int, metrics: Dict[str, Any]) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE backtest_runs
                SET ended_at = NOW(), metrics = %s
                WHERE id = %s
                """,
                (json.dumps(metrics), run_id),
            )


def list_recent_backtest_runs(
    connector,
    track: str,
    limit: int = 10,
    include_sources: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
    data_regimes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    with connector() as conn:
        with conn.cursor() as cur:
            cond = ["track = %s"]
            params: List[Any] = [track]
            if include_sources:
                cond.append("COALESCE(run_source, 'prod') = ANY(%s)")
                params.append([s.strip() for s in include_sources if s and s.strip()])
            if exclude_sources:
                cond.append("COALESCE(run_source, 'prod') <> ALL(%s)")
                params.append([s.strip() for s in exclude_sources if s and s.strip()])
            if data_regimes:
                cond.append("COALESCE(NULLIF(config->>'data_regime',''),'missing') = ANY(%s)")
                params.append([s.strip() for s in data_regimes if s and s.strip()])
            params.append(limit)
            cur.execute(
                f"""
                SELECT * FROM backtest_runs
                WHERE {' AND '.join(cond)}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            return [dict(r) for r in cur.fetchall()]
