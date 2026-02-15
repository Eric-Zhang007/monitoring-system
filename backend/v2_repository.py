from __future__ import annotations

import hashlib
import json
import os
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool

from schemas_v2 import Event


class V2Repository:
    def __init__(self, db_url: str):
        self.db_url = db_url
        min_conn = int(os.getenv("DB_POOL_MIN_CONN", "1"))
        max_conn = int(os.getenv("DB_POOL_MAX_CONN", "8"))
        self._pool = ThreadedConnectionPool(minconn=min_conn, maxconn=max_conn, dsn=self.db_url)

    class _PooledConnection(AbstractContextManager):
        class _ConnProxy:
            def __init__(self, conn):
                self._conn = conn

            def cursor(self, *args, **kwargs):
                if "cursor_factory" not in kwargs:
                    kwargs["cursor_factory"] = RealDictCursor
                return self._conn.cursor(*args, **kwargs)

            def __getattr__(self, item):
                return getattr(self._conn, item)

        def __init__(self, pool: ThreadedConnectionPool):
            self.pool = pool
            self.conn = None

        def __enter__(self):
            self.conn = self.pool.getconn()
            return self._ConnProxy(self.conn)

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.conn:
                return False
            try:
                if exc_type:
                    self.conn.rollback()
                else:
                    self.conn.commit()
            finally:
                self.pool.putconn(self.conn)
            return False

    def _connect(self):
        return self._PooledConnection(self._pool)

    @staticmethod
    def _fingerprint(event: Event) -> str:
        key = f"{event.event_type}|{event.title.strip().lower()}|{event.source_url or ''}|{event.occurred_at.isoformat()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def upsert_entity(self, cur, entity: Dict[str, Any]) -> int:
        cur.execute(
            """
            INSERT INTO entities (entity_type, name, symbol, country, sector, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (entity_type, name)
            DO UPDATE SET symbol = EXCLUDED.symbol,
                          country = COALESCE(EXCLUDED.country, entities.country),
                          sector = COALESCE(EXCLUDED.sector, entities.sector),
                          metadata = entities.metadata || EXCLUDED.metadata,
                          updated_at = NOW()
            RETURNING id
            """,
            (
                entity["entity_type"],
                entity["name"],
                entity.get("symbol"),
                entity.get("country"),
                entity.get("sector"),
                json.dumps(entity.get("metadata", {})),
            ),
        )
        return cur.fetchone()["id"]

    def ingest_events(self, events: List[Event]) -> Tuple[int, int, int, List[int]]:
        accepted = len(events)
        inserted = 0
        deduplicated = 0
        event_ids: List[int] = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                for event in events:
                    fp = self._fingerprint(event)
                    cur.execute(
                        """
                        INSERT INTO events (
                            event_type, title, occurred_at, source_url, source_name,
                            source_timezone, source_tier, confidence_score, event_importance,
                            novelty_score, entity_confidence, latency_ms, dedup_cluster_id,
                            payload, fingerprint, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (fingerprint) DO NOTHING
                        RETURNING id
                        """,
                        (
                            event.event_type,
                            event.title,
                            event.occurred_at,
                            event.source_url,
                            event.source_name,
                            event.source_timezone,
                            event.source_tier,
                            event.confidence_score,
                            event.event_importance,
                            event.novelty_score,
                            event.entity_confidence,
                            event.latency_ms,
                            event.dedup_cluster_id,
                            json.dumps(event.payload),
                            fp,
                        ),
                    )
                    row = cur.fetchone()
                    if row:
                        event_id = row["id"]
                        inserted += 1
                    else:
                        cur.execute("SELECT id FROM events WHERE fingerprint = %s", (fp,))
                        existing = cur.fetchone()
                        if not existing:
                            continue
                        event_id = existing["id"]
                        deduplicated += 1
                    event_ids.append(event_id)

                    links_payload = []
                    for entity in event.entities:
                        entity_id = self.upsert_entity(cur, entity.model_dump())
                        links_payload.append((event_id, entity_id, "mentioned"))
                    if links_payload:
                        execute_values(
                            cur,
                            """
                            INSERT INTO event_links (event_id, entity_id, role, created_at)
                            VALUES %s
                            ON CONFLICT (event_id, entity_id, role) DO NOTHING
                            """,
                            links_payload,
                            template="(%s, %s, %s, NOW())",
                        )
                    self._insert_data_quality_audit(cur, event_id, event)

        return accepted, inserted, deduplicated, event_ids

    def _insert_data_quality_audit(self, cur, event_id: int, event: Event) -> None:
        freshness_sec = int(max(0.0, (datetime.utcnow() - event.occurred_at.replace(tzinfo=None)).total_seconds()))
        entity_coverage = float(min(1.0, len(event.entities) / 4.0))
        quality_score = (
            0.3 * float(event.confidence_score)
            + 0.25 * float(event.event_importance)
            + 0.25 * float(event.novelty_score)
            + 0.2 * entity_coverage
        )
        try:
            cur.execute(
                """
                INSERT INTO data_quality_audit (
                    event_id, source_name, freshness_sec, entity_coverage,
                    quality_score, notes, audited_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    event_id,
                    event.source_name,
                    freshness_sec,
                    entity_coverage,
                    quality_score,
                    "auto_audit_v1",
                ),
            )
        except Exception:
            # Keep ingestion backward-compatible before migration is applied.
            return

    def get_entity(self, entity_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
                entity = cur.fetchone()
                if not entity:
                    return None
                cur.execute(
                    """
                    SELECT e.id, e.event_type, e.title, e.occurred_at, e.source_url, el.role
                    FROM event_links el
                    JOIN events e ON e.id = el.event_id
                    WHERE el.entity_id = %s
                    ORDER BY e.occurred_at DESC
                    LIMIT 20
                    """,
                    (entity_id,),
                )
                entity["recent_events"] = [dict(r) for r in cur.fetchall()]
                return dict(entity)

    def insert_prediction(
        self,
        track: str,
        target: str,
        score: float,
        confidence: float,
        outputs: Dict[str, Any],
        explanation: Dict[str, Any],
        horizon: Optional[str] = None,
        model_id: Optional[int] = None,
        feature_set_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions_v2 (
                        track, target, score, confidence, outputs,
                        horizon, model_id, feature_set_id, decision_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()) RETURNING id
                    """,
                    (
                        track,
                        target,
                        score,
                        confidence,
                        json.dumps(outputs),
                        horizon,
                        model_id,
                        feature_set_id,
                        decision_id,
                    ),
                )
                prediction_id = cur.fetchone()["id"]
                cur.execute(
                    """
                    INSERT INTO prediction_explanations (
                        prediction_id, top_event_contributors, top_feature_contributors,
                        evidence_links, model_version, feature_version, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        prediction_id,
                        json.dumps(explanation.get("top_event_contributors", [])),
                        json.dumps(explanation.get("top_feature_contributors", [])),
                        json.dumps(explanation.get("evidence_links", [])),
                        explanation.get("model_version", "v2-baseline"),
                        explanation.get("feature_version", "v2-default"),
                    ),
                )
                return prediction_id

    def get_prediction_explanation(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM predictions_v2 WHERE id = %s", (prediction_id,))
                pred = cur.fetchone()
                if not pred:
                    return None
                cur.execute(
                    "SELECT * FROM prediction_explanations WHERE prediction_id = %s",
                    (prediction_id,),
                )
                exp = cur.fetchone()
                result = dict(pred)
                result["outputs"] = result.get("outputs", {})
                if exp:
                    expd = dict(exp)
                    result["explanation"] = {
                        "top_event_contributors": expd.get("top_event_contributors", []),
                        "top_feature_contributors": expd.get("top_feature_contributors", []),
                        "evidence_links": expd.get("evidence_links", []),
                        "model_version": expd.get("model_version"),
                        "feature_version": expd.get("feature_version"),
                    }
                return result

    def recent_event_context(self, target: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.event_type, e.title, e.source_url, e.occurred_at, e.confidence_score
                    FROM events e
                    LEFT JOIN event_links el ON el.event_id = e.id
                    LEFT JOIN entities en ON en.id = el.entity_id
                    WHERE en.name = %s OR en.symbol = UPPER(%s)
                    ORDER BY e.occurred_at DESC
                    LIMIT %s
                    """,
                    (target, target, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def latest_price_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, price, volume, timestamp
                    FROM prices
                    WHERE symbol = UPPER(%s)
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (symbol,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_target_profiles(self, targets: List[str]) -> Dict[str, Dict[str, Any]]:
        keys = [t.strip().upper() for t in targets if t and t.strip()]
        if not keys:
            return {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (k)
                        k,
                        sector,
                        metadata
                    FROM (
                        SELECT
                            UPPER(COALESCE(NULLIF(symbol, ''), name)) AS k,
                            sector,
                            metadata,
                            updated_at
                        FROM entities
                        WHERE UPPER(COALESCE(NULLIF(symbol, ''), name)) = ANY(%s)
                    ) t
                    ORDER BY k, updated_at DESC NULLS LAST
                    """,
                    (keys,),
                )
                out: Dict[str, Dict[str, Any]] = {}
                for r in cur.fetchall():
                    meta = r.get("metadata") or {}
                    style_bucket = None
                    if isinstance(meta, dict):
                        style_bucket = meta.get("style_bucket") or meta.get("style")
                    out[str(r["k"]).upper()] = {
                        "sector": r.get("sector"),
                        "style_bucket": style_bucket,
                    }
                return out

    def create_backtest_run(self, run_name: str, track: str, config: Dict[str, Any]) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO backtest_runs (run_name, track, started_at, metrics, config, created_at)
                    VALUES (%s, %s, NOW(), %s, %s, NOW())
                    RETURNING id
                    """,
                    (run_name, track, json.dumps({"status": "running"}), json.dumps(config)),
                )
                return cur.fetchone()["id"]

    def finish_backtest_run(self, run_id: int, metrics: Dict[str, Any]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE backtest_runs
                    SET ended_at = NOW(), metrics = %s
                    WHERE id = %s
                    """,
                    (json.dumps(metrics), run_id),
                )

    def get_backtest_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM backtest_runs WHERE id = %s", (run_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def list_recent_backtest_runs(self, track: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM backtest_runs
                    WHERE track = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (track, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def load_price_history(self, symbol: str, lookback_days: int = 90) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, price::float AS price, volume::float AS volume, timestamp
                    FROM prices
                    WHERE symbol = UPPER(%s)
                      AND timestamp > NOW() - make_interval(days => %s)
                    ORDER BY timestamp ASC
                    """,
                    (symbol, lookback_days),
                )
                return [dict(r) for r in cur.fetchall()]

    def latest_prediction(self, track: str, target: str, horizon: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if horizon:
                    cur.execute(
                        """
                        SELECT * FROM predictions_v2
                        WHERE track = %s AND target = %s AND (horizon = %s OR horizon IS NULL)
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (track, target, horizon),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM predictions_v2
                        WHERE track = %s AND target = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (track, target),
                    )
                row = cur.fetchone()
                return dict(row) if row else None

    def insert_signal_candidate(
        self,
        track: str,
        target: str,
        horizon: str,
        score: float,
        confidence: float,
        action: str,
        policy: str,
        decision_id: str,
        metadata: Dict[str, Any],
    ) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO signal_candidates (
                        track, target, horizon, score, confidence,
                        action, policy, decision_id, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (
                        track,
                        target,
                        horizon,
                        score,
                        confidence,
                        action,
                        policy,
                        decision_id,
                        json.dumps(metadata),
                    ),
                )
                return cur.fetchone()["id"]

    def save_positions_snapshot(self, decision_id: str, positions: List[Dict[str, Any]], reason: str = "rebalance") -> None:
        if not positions:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                rows = [(decision_id, pos["target"], pos["track"], pos["weight"], reason) for pos in positions]
                execute_values(
                    cur,
                    """
                    INSERT INTO positions_snapshots (
                        decision_id, target, track, weight, reason, created_at
                    ) VALUES %s
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s, NOW())",
                )

    def save_orders_sim(self, decision_id: str, orders: List[Dict[str, Any]]) -> None:
        if not orders:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                payload = [
                    (
                        decision_id,
                        order["target"],
                        order["track"],
                        order["side"],
                        order.get("quantity", 0.0),
                        order.get("est_price"),
                        order.get("est_cost_bps", 0.0),
                        order.get("status", "simulated"),
                        order.get("adapter", "paper"),
                        order.get("venue", "coinbase"),
                        order.get("time_in_force", "IOC"),
                        order.get("max_slippage_bps", 20.0),
                        order.get("strategy_id", "default-liquid-v1"),
                        json.dumps(order.get("metadata", {})),
                    )
                    for order in orders
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO orders_sim (
                        decision_id, target, track, side, quantity,
                        est_price, est_cost_bps, status, adapter, venue, time_in_force,
                        max_slippage_bps, strategy_id, metadata, created_at
                    ) VALUES %s
                    """,
                    payload,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                )

    def create_execution_orders(
        self,
        decision_id: str,
        adapter: str,
        venue: str,
        time_in_force: str,
        max_slippage_bps: float,
        orders: List[Dict[str, Any]],
    ) -> List[int]:
        if not orders:
            return []
        with self._connect() as conn:
            with conn.cursor() as cur:
                rows = [
                    (
                        decision_id,
                        o["target"],
                        o.get("track", "liquid"),
                        o["side"],
                        o.get("quantity", 0.0),
                        o.get("est_price"),
                        float(max_slippage_bps),
                        "submitted",
                        adapter,
                        venue,
                        time_in_force,
                        float(max_slippage_bps),
                        o.get("strategy_id", "default-liquid-v1"),
                        json.dumps(o.get("metadata", {})),
                    )
                    for o in orders
                ]
                inserted = execute_values(
                    cur,
                    """
                    INSERT INTO orders_sim (
                        decision_id, target, track, side, quantity, est_price, est_cost_bps,
                        status, adapter, venue, time_in_force, max_slippage_bps, strategy_id, metadata, created_at
                    ) VALUES %s
                    RETURNING id
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                    fetch=True,
                )
                return [int(r["id"]) for r in inserted]

    def fetch_orders_for_decision(self, decision_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM orders_sim
                    WHERE decision_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (decision_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def get_order_by_id(self, order_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM orders_sim WHERE id = %s", (order_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def update_order_execution(self, order_id: int, status: str, metadata: Dict[str, Any]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE orders_sim
                    SET status = %s, metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                    WHERE id = %s
                    """,
                    (status, json.dumps(metadata), order_id),
                )

    def save_risk_event(self, decision_id: str, severity: str, code: str, message: str, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO risk_events (
                        decision_id, severity, code, message, payload, created_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (decision_id, severity, code, message, json.dumps(payload)),
                )

    def promote_model(
        self,
        track: str,
        model_name: str,
        model_version: str,
        passed: bool,
        metrics: Dict[str, Any],
        gate_reason: str,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_promotions (
                        track, model_name, model_version, passed, metrics, gate_reason, promoted_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (track, model_name, model_version, passed, json.dumps(metrics), gate_reason),
                )

    def sample_data_quality(self, limit: int = 200, min_quality_score: float = 0.0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dqa.*, e.title, e.event_type, e.source_url, e.occurred_at
                    FROM data_quality_audit dqa
                    LEFT JOIN events e ON e.id = dqa.event_id
                    WHERE COALESCE(dqa.quality_score, 0.0) >= %s
                    ORDER BY dqa.audited_at DESC
                    LIMIT %s
                    """,
                    (min_quality_score, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def update_data_quality_audit(self, audit_id: int, reviewer: str, verdict: str, note: Optional[str]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE data_quality_audit
                    SET notes = COALESCE(notes, '') || %s,
                        reviewer = %s,
                        verdict = %s,
                        reviewed_at = NOW(),
                        audited_at = NOW()
                    WHERE id = %s
                    """,
                    (f"[{reviewer}] verdict={verdict}; note={note or ''}", reviewer, verdict, audit_id),
                )

    def get_data_quality_stats(self, lookback_days: int = 7) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)::double precision AS total,
                        COALESCE(SUM(CASE WHEN verdict IS NOT NULL THEN 1 ELSE 0 END), 0)::double precision AS reviewed,
                        COALESCE(SUM(CASE WHEN verdict = 'correct' THEN 1 ELSE 0 END), 0)::double precision AS correct_cnt,
                        COALESCE(SUM(CASE WHEN verdict = 'incorrect' THEN 1 ELSE 0 END), 0)::double precision AS incorrect_cnt,
                        COALESCE(SUM(CASE WHEN verdict = 'uncertain' THEN 1 ELSE 0 END), 0)::double precision AS uncertain_cnt,
                        COALESCE(AVG(quality_score), 0.0)::double precision AS avg_quality_score,
                        COALESCE(AVG(entity_coverage), 0.0)::double precision AS avg_entity_coverage,
                        COALESCE(AVG(freshness_sec), 0.0)::double precision AS avg_freshness_sec
                    FROM data_quality_audit
                    WHERE audited_at > NOW() - make_interval(days => %s)
                    """,
                    (lookback_days,),
                )
                total_row = dict(cur.fetchone() or {})

                cur.execute(
                    """
                    SELECT
                        COALESCE(source_name, 'unknown') AS source_name,
                        COUNT(*)::double precision AS samples,
                        COALESCE(SUM(CASE WHEN verdict IS NOT NULL THEN 1 ELSE 0 END), 0)::double precision AS reviewed,
                        COALESCE(SUM(CASE WHEN verdict = 'correct' THEN 1 ELSE 0 END), 0)::double precision AS correct_cnt,
                        COALESCE(SUM(CASE WHEN verdict = 'incorrect' THEN 1 ELSE 0 END), 0)::double precision AS incorrect_cnt,
                        COALESCE(AVG(quality_score), 0.0)::double precision AS avg_quality_score,
                        COALESCE(AVG(freshness_sec), 0.0)::double precision AS avg_freshness_sec
                    FROM data_quality_audit
                    WHERE audited_at > NOW() - make_interval(days => %s)
                    GROUP BY COALESCE(source_name, 'unknown')
                    ORDER BY samples DESC, source_name ASC
                    LIMIT 50
                    """,
                    (lookback_days,),
                )
                source_rows = [dict(r) for r in cur.fetchall()]

        total = float(total_row.get("total") or 0.0)
        reviewed = float(total_row.get("reviewed") or 0.0)
        correct_cnt = float(total_row.get("correct_cnt") or 0.0)
        incorrect_cnt = float(total_row.get("incorrect_cnt") or 0.0)
        uncertain_cnt = float(total_row.get("uncertain_cnt") or 0.0)

        precision = correct_cnt / max(1.0, reviewed)
        review_coverage = reviewed / max(1.0, total)

        totals = {
            "samples": round(total, 6),
            "reviewed_samples": round(reviewed, 6),
            "review_coverage": round(review_coverage, 6),
            "precision": round(precision, 6),
            "incorrect_rate": round(incorrect_cnt / max(1.0, reviewed), 6),
            "uncertain_rate": round(uncertain_cnt / max(1.0, reviewed), 6),
            "avg_quality_score": round(float(total_row.get("avg_quality_score") or 0.0), 6),
            "avg_entity_coverage": round(float(total_row.get("avg_entity_coverage") or 0.0), 6),
            "avg_freshness_sec": round(float(total_row.get("avg_freshness_sec") or 0.0), 6),
        }

        by_source: List[Dict[str, Any]] = []
        for row in source_rows:
            s_reviewed = float(row.get("reviewed") or 0.0)
            s_correct = float(row.get("correct_cnt") or 0.0)
            by_source.append(
                {
                    "source_name": row.get("source_name"),
                    "samples": round(float(row.get("samples") or 0.0), 6),
                    "reviewed_samples": round(s_reviewed, 6),
                    "precision": round(s_correct / max(1.0, s_reviewed), 6),
                    "incorrect_rate": round(float(row.get("incorrect_cnt") or 0.0) / max(1.0, s_reviewed), 6),
                    "avg_quality_score": round(float(row.get("avg_quality_score") or 0.0), 6),
                    "avg_freshness_sec": round(float(row.get("avg_freshness_sec") or 0.0), 6),
                }
            )

        return {"totals": totals, "by_source": by_source}

    def get_active_model_state(self, track: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM active_model_state WHERE track = %s", (track,))
                row = cur.fetchone()
                return dict(row) if row else None

    def upsert_active_model_state(
        self,
        track: str,
        active_model_name: str,
        active_model_version: str,
        previous_model_name: Optional[str],
        previous_model_version: Optional[str],
        status: str,
        metadata: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO active_model_state (
                        track, active_model_name, active_model_version,
                        previous_model_name, previous_model_version,
                        status, metadata, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (track) DO UPDATE SET
                        active_model_name = EXCLUDED.active_model_name,
                        active_model_version = EXCLUDED.active_model_version,
                        previous_model_name = EXCLUDED.previous_model_name,
                        previous_model_version = EXCLUDED.previous_model_version,
                        status = EXCLUDED.status,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        track,
                        active_model_name,
                        active_model_version,
                        previous_model_name,
                        previous_model_version,
                        status,
                        json.dumps(metadata),
                    ),
                )

    def get_recent_prediction_scores(self, track: str, lookback_hours: int) -> List[float]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT score
                    FROM predictions_v2
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                    ORDER BY created_at DESC
                    """,
                    (track, lookback_hours),
                )
                return [float(r["score"]) for r in cur.fetchall()]

    def get_prediction_scores_window(self, track: str, offset_hours: int, window_hours: int) -> List[float]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT score
                    FROM predictions_v2
                    WHERE track = %s
                      AND created_at <= NOW() - make_interval(hours => %s)
                      AND created_at > NOW() - make_interval(hours => %s)
                    ORDER BY created_at DESC
                    """,
                    (track, offset_hours, offset_hours + window_hours),
                )
                return [float(r["score"]) for r in cur.fetchall()]

    def get_execution_slippage_samples(self, track: str, lookback_hours: int) -> List[float]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        CASE
                            WHEN est_price IS NULL OR est_price = 0 THEN NULL
                            WHEN side = 'buy' THEN (COALESCE((metadata->'execution'->>'avg_fill_price')::double precision, est_price) - est_price) / est_price
                            ELSE (est_price - COALESCE((metadata->'execution'->>'avg_fill_price')::double precision, est_price)) / est_price
                        END AS slippage
                    FROM orders_sim
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                      AND status IN ('filled', 'submitted')
                    """,
                    (track, lookback_hours),
                )
                vals = []
                for r in cur.fetchall():
                    v = r.get("slippage")
                    if v is not None:
                        vals.append(float(v))
                return vals

    def build_pnl_attribution(self, track: str, lookback_hours: int) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        target,
                        COALESCE(SUM(
                            CASE
                              WHEN side = 'buy' THEN -1.0
                              WHEN side = 'sell' THEN 1.0
                              ELSE 0.0
                            END
                            * quantity
                            * COALESCE((metadata->'execution'->>'avg_fill_price')::double precision, est_price, 0.0)
                        ), 0.0) AS notional_signed,
                        COALESCE(SUM(quantity * COALESCE(est_cost_bps, 0.0) / 10000.0), 0.0) AS est_cost,
                        COUNT(*) AS order_count,
                        COALESCE(SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END), 0) AS rejected_count
                    FROM orders_sim
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                    GROUP BY target
                    ORDER BY ABS(COALESCE(SUM(quantity * COALESCE(est_price, 0.0)), 0.0)) DESC
                    LIMIT 100
                    """,
                    (track, lookback_hours),
                )
                rows = [dict(r) for r in cur.fetchall()]

        by_target = []
        total_notional = 0.0
        total_cost = 0.0
        total_rejected = 0
        total_orders = 0
        for r in rows:
            n = float(r.get("notional_signed") or 0.0)
            c = float(r.get("est_cost") or 0.0)
            oc = int(r.get("order_count") or 0)
            rc = int(r.get("rejected_count") or 0)
            total_notional += n
            total_cost += c
            total_orders += oc
            total_rejected += rc
            by_target.append(
                {
                    "target": r.get("target"),
                    "selection_pnl": round(n * 0.0008, 8),
                    "timing_pnl": round(n * 0.0004, 8),
                    "execution_pnl": round(-abs(n) * 0.0002, 8),
                    "cost_pnl": round(-c, 8),
                    "risk_constraint_pnl": round(-abs(n) * 0.0001, 8),
                    "orders": oc,
                    "rejected": rc,
                }
            )

        reject_rate = float(total_rejected / max(1, total_orders))
        totals = {
            "selection_pnl": round(sum(float(x["selection_pnl"]) for x in by_target), 8),
            "timing_pnl": round(sum(float(x["timing_pnl"]) for x in by_target), 8),
            "execution_pnl": round(sum(float(x["execution_pnl"]) for x in by_target), 8),
            "cost_pnl": round(sum(float(x["cost_pnl"]) for x in by_target), 8),
            "risk_constraint_pnl": round(sum(float(x["risk_constraint_pnl"]) for x in by_target), 8),
            "net_pnl": round(sum(float(x["selection_pnl"]) + float(x["timing_pnl"]) + float(x["execution_pnl"]) + float(x["cost_pnl"]) + float(x["risk_constraint_pnl"]) for x in by_target), 8),
            "gross_notional_signed": round(total_notional, 8),
            "reject_rate": round(reject_rate, 8),
        }
        return {"totals": totals, "by_target": by_target}
