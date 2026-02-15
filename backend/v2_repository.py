from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from numbers import Number
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
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
                    published_at = event.published_at or event.occurred_at
                    ingested_at = event.ingested_at or datetime.utcnow()
                    latency_ms = int(event.latency_ms or event.source_latency_ms or 0)
                    available_at = event.available_at or (published_at + timedelta(milliseconds=max(0, latency_ms)))
                    effective_at = event.effective_at or available_at
                    payload = dict(event.payload or {})
                    if "source_confidence" not in payload:
                        payload["source_confidence"] = float(event.confidence_score)
                    if "source_tier_weight" not in payload:
                        payload["source_tier_weight"] = max(0.1, min(1.0, (6.0 - float(event.source_tier)) / 5.0))
                    try:
                        cur.execute(
                            """
                            INSERT INTO events (
                                event_type, title, occurred_at, published_at, ingested_at, available_at, effective_at,
                                source_url, source_name, source_timezone, source_tier, confidence_score, event_importance,
                                novelty_score, entity_confidence, latency_ms, source_latency_ms, dedup_cluster_id,
                                market_scope, payload, fingerprint, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (fingerprint) DO NOTHING
                            RETURNING id
                            """,
                            (
                                event.event_type,
                                event.title,
                                event.occurred_at,
                                published_at,
                                ingested_at,
                                available_at,
                                effective_at,
                                event.source_url,
                                event.source_name,
                                event.source_timezone,
                                event.source_tier,
                                event.confidence_score,
                                event.event_importance,
                                event.novelty_score,
                                event.entity_confidence,
                                event.latency_ms,
                                event.source_latency_ms,
                                event.dedup_cluster_id,
                                event.market_scope,
                                json.dumps(payload),
                                fp,
                            ),
                        )
                    except Exception:
                        conn.rollback()
                        cur.execute(
                            """
                            INSERT INTO events (
                                event_type, title, occurred_at, source_url, source_name,
                                source_timezone, source_tier, confidence_score, event_importance,
                                novelty_score, entity_confidence, latency_ms, dedup_cluster_id,
                                payload, fingerprint, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
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
                                json.dumps(payload),
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
                try:
                    cur.execute(
                        """
                        SELECT
                            e.id, e.event_type, e.title, e.source_url, e.occurred_at, e.confidence_score,
                            e.available_at, e.effective_at, e.market_scope
                        FROM events e
                        LEFT JOIN event_links el ON el.event_id = e.id
                        LEFT JOIN entities en ON en.id = el.entity_id
                        WHERE en.name = %s OR en.symbol = UPPER(%s)
                        ORDER BY COALESCE(e.available_at, e.occurred_at) DESC
                        LIMIT %s
                        """,
                        (target, target, limit),
                    )
                except Exception:
                    conn.rollback()
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

    def resolve_asset_universe_asof(
        self,
        track: str,
        as_of: datetime,
        fallback_targets: List[str],
    ) -> Dict[str, Any]:
        fallback = []
        seen = set()
        for t in fallback_targets:
            sym = str(t or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                fallback.append(sym)
        out: Dict[str, Any] = {
            "track": str(track).strip().lower(),
            "as_of": as_of.isoformat(),
            "symbols": fallback,
            "source": "env_default",
            "universe_version": "env_default",
            "snapshot_at": None,
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT track, as_of, universe_version, source, symbols_json
                        FROM asset_universe_snapshots
                        WHERE track = %s
                          AND as_of <= %s
                        ORDER BY as_of DESC
                        LIMIT 1
                        """,
                        (str(track).strip().lower(), as_of),
                    )
                except Exception:
                    conn.rollback()
                    return out
                row = cur.fetchone()
                if not row:
                    return out
                payload = row.get("symbols_json")
                raw_symbols: List[str] = []
                if isinstance(payload, list):
                    raw_symbols = [str(x) for x in payload]
                elif isinstance(payload, dict) and isinstance(payload.get("symbols"), list):
                    raw_symbols = [str(x) for x in payload.get("symbols")]
                symbols: List[str] = []
                seen_symbols = set()
                for v in raw_symbols:
                    sym = str(v or "").strip().upper()
                    if sym and sym not in seen_symbols:
                        seen_symbols.add(sym)
                        symbols.append(sym)
                if not symbols:
                    return out
                out["symbols"] = symbols
                out["source"] = str(row.get("source") or "snapshot")
                out["universe_version"] = str(row.get("universe_version") or "unknown")
                snap_as_of = row.get("as_of")
                out["snapshot_at"] = snap_as_of.isoformat() if isinstance(snap_as_of, datetime) else None
                return out

    def create_backtest_run(self, run_name: str, track: str, config: Dict[str, Any], run_source: str = "prod") -> int:
        with self._connect() as conn:
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

    def mark_backtest_run_superseded(self, run_id: int, superseded_by_run_id: int, reason: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE backtest_runs
                    SET superseded_by_run_id = %s,
                        supersede_reason = %s,
                        superseded_at = NOW()
                    WHERE id = %s
                    """,
                    (superseded_by_run_id, reason, run_id),
                )

    def get_backtest_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM backtest_runs WHERE id = %s", (run_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def list_recent_backtest_runs(
        self,
        track: str,
        limit: int = 10,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        data_regimes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
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

    def list_failed_backtest_runs(
        self,
        track: str,
        reason: Optional[str] = None,
        unsuperseded_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cond = ["track = %s", "COALESCE(metrics->>'status','') = 'failed'"]
                params: List[Any] = [track]
                if reason:
                    cond.append("COALESCE(metrics->>'reason','') = %s")
                    params.append(reason)
                if unsuperseded_only:
                    cond.append("superseded_by_run_id IS NULL")
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT *
                    FROM backtest_runs
                    WHERE {' AND '.join(cond)}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                return [dict(r) for r in cur.fetchall()]

    def model_artifact_exists(self, model_name: str, track: str, model_version: str) -> bool:
        def _has_required_fields(payload: Dict[str, Any], required: List[str]) -> bool:
            for k in required:
                v = payload.get(k)
                if v is None:
                    return False
                if isinstance(v, str) and not v.strip():
                    return False
            return True

        def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            return data if isinstance(data, dict) else None

        def _find_checkpoint_manifest(path: Path) -> Optional[Path]:
            candidates = [
                path.with_suffix(".manifest.json"),
                path.with_suffix(path.suffix + ".manifest.json"),
                path.with_suffix(path.suffix + ".json"),
            ]
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    return cand
            return None

        base_required = [
            "model_name",
            "model_version",
            "track",
            "type",
            "created_at",
            "feature_version",
            "data_version",
        ]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT artifact_path
                    FROM model_registry
                    WHERE model_name = %s AND track = %s AND model_version = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (model_name, track, model_version),
                )
                row = cur.fetchone()
                if not row:
                    return False
                p = str(row.get("artifact_path") or "")
                if not p:
                    return False
                path = Path(p)
                if not path.exists() or not path.is_file():
                    return False
                if path.suffix.lower() == ".json":
                    payload = _load_json_file(path)
                    if payload is None:
                        return False
                    if str(payload.get("type") or "").strip().lower() == "bootstrap_placeholder":
                        return False
                    if not _has_required_fields(payload, base_required):
                        return False
                    if str(payload.get("track") or "").strip().lower() != str(track).strip().lower():
                        return False
                    if str(payload.get("model_name") or "").strip() != str(model_name).strip():
                        return False
                    if str(payload.get("model_version") or "").strip() != str(model_version).strip():
                        return False
                    return True
                if path.suffix.lower() in {".pt", ".pth"}:
                    try:
                        if path.stat().st_size <= 1024:
                            return False
                    except Exception:
                        return False
                    manifest_path = _find_checkpoint_manifest(path)
                    if manifest_path is None:
                        return False
                    manifest = _load_json_file(manifest_path)
                    if manifest is None:
                        return False
                    if not _has_required_fields(manifest, base_required):
                        return False
                    if str(manifest.get("track") or "").strip().lower() != str(track).strip().lower():
                        return False
                    if str(manifest.get("model_name") or "").strip() != str(model_name).strip():
                        return False
                    if str(manifest.get("model_version") or "").strip() != str(model_version).strip():
                        return False
                    # NN checkpoint must carry deterministic replay metadata via sidecar manifest.
                    if not _has_required_fields(manifest, ["normalization", "train_report_hash", "feature_payload_schema_version"]):
                        return False
                    return True
                return True

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

    def get_trade_audit_chain(self, decision_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM signal_candidates
                    WHERE decision_id = %s
                    ORDER BY created_at ASC
                    """,
                    (decision_id,),
                )
                signals = [dict(r) for r in cur.fetchall()]
                cur.execute(
                    """
                    SELECT *
                    FROM orders_sim
                    WHERE decision_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (decision_id,),
                )
                orders = [dict(r) for r in cur.fetchall()]
                cur.execute(
                    """
                    SELECT *
                    FROM positions_snapshots
                    WHERE decision_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (decision_id,),
                )
                positions = [dict(r) for r in cur.fetchall()]
        return {"signals": signals, "orders": orders, "positions": positions}

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

    def save_scheduler_audit_log(
        self,
        track: str,
        action: str,
        window: Dict[str, Any],
        thresholds: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> None:
        self.save_risk_event(
            decision_id=f"scheduler-{track}-{action}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            severity="info",
            code="scheduler_audit_log",
            message=f"{track}:{action}",
            payload={
                "who": "system",
                "source": "scheduler",
                "track": track,
                "action": action,
                "window": window,
                "thresholds": thresholds,
                "decision": decision,
            },
        )

    def upsert_kill_switch_state(
        self,
        track: str,
        strategy_id: str,
        state: str,
        reason: str,
        duration_minutes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        expires_at = None
        if duration_minutes and duration_minutes > 0:
            expires_at = datetime.utcnow() + timedelta(minutes=int(duration_minutes))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO risk_control_state (
                        track, strategy_id, state, reason, metadata, triggered_at, expires_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW(), %s, NOW())
                    ON CONFLICT (track, strategy_id)
                    DO UPDATE SET
                        state = EXCLUDED.state,
                        reason = EXCLUDED.reason,
                        metadata = EXCLUDED.metadata,
                        triggered_at = EXCLUDED.triggered_at,
                        expires_at = EXCLUDED.expires_at,
                        updated_at = NOW()
                    RETURNING *
                    """,
                    (
                        track,
                        strategy_id,
                        state,
                        reason,
                        json.dumps(metadata or {}),
                        expires_at,
                    ),
                )
                row = cur.fetchone()
                return dict(row) if row else {}

    def get_kill_switch_state(self, track: str, strategy_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM risk_control_state
                    WHERE track = %s AND strategy_id = %s
                    LIMIT 1
                    """,
                    (track, strategy_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
        row = self.get_kill_switch_state(track, strategy_id)
        if not row:
            return False
        if str(row.get("state") or "armed") != "triggered":
            return False
        expires_at = row.get("expires_at")
        if expires_at is None:
            return True
        now = datetime.now(tz=expires_at.tzinfo) if getattr(expires_at, "tzinfo", None) else datetime.utcnow()
        return expires_at > now

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
                    ON CONFLICT (track, model_name, model_version)
                    DO UPDATE SET
                        passed = EXCLUDED.passed,
                        metrics = EXCLUDED.metrics,
                        gate_reason = EXCLUDED.gate_reason,
                        promoted_at = NOW()
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
                cur.execute("SELECT event_id FROM data_quality_audit WHERE id = %s", (audit_id,))
                row = cur.fetchone()
                event_id = int(row["event_id"]) if row and row.get("event_id") is not None else None
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
                cur.execute(
                    """
                    INSERT INTO data_quality_review_logs (
                        audit_id, event_id, reviewer, verdict, note, created_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (audit_id, event_id, reviewer, verdict, note),
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

    def get_data_quality_consistency(self, lookback_days: int = 30) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reviewer, verdict, event_id
                    FROM data_quality_review_logs
                    WHERE created_at > NOW() - make_interval(days => %s)
                      AND event_id IS NOT NULL
                    ORDER BY event_id, created_at
                    """,
                    (lookback_days,),
                )
                rows = [dict(r) for r in cur.fetchall()]
        total_logs = len(rows)
        by_event: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            eid = int(r["event_id"])
            by_event.setdefault(eid, []).append(r)
        multi_events = 0
        agree = 0
        total_pairs = 0
        pair_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
        for _, reviews in by_event.items():
            reviewers_seen = set()
            uniq = []
            for rev in reviews:
                rv = str(rev.get("reviewer") or "")
                if rv in reviewers_seen:
                    continue
                reviewers_seen.add(rv)
                uniq.append(rev)
            if len(uniq) < 2:
                continue
            multi_events += 1
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a = uniq[i]
                    b = uniq[j]
                    ra = str(a.get("reviewer") or "")
                    rb = str(b.get("reviewer") or "")
                    key = tuple(sorted((ra, rb)))
                    st = pair_stats.setdefault(key, {"pairs": 0, "agree": 0})
                    st["pairs"] += 1
                    total_pairs += 1
                    if a.get("verdict") == b.get("verdict"):
                        st["agree"] += 1
                        agree += 1
        reviewer_pairs = [
            {
                "reviewer_a": k[0],
                "reviewer_b": k[1],
                "pairs": int(v["pairs"]),
                "agreement": round(float(v["agree"] / max(1, v["pairs"])), 6),
            }
            for k, v in sorted(pair_stats.items(), key=lambda item: item[1]["pairs"], reverse=True)
        ]
        return {
            "total_review_logs": total_logs,
            "multi_review_events": multi_events,
            "pairwise_agreement": round(float(agree / max(1, total_pairs)), 6),
            "reviewer_pairs": reviewer_pairs[:50],
        }

    @staticmethod
    def _feature_payload_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return 0.0, 0.0
        keys = set(a.keys()) | set(b.keys())
        diffs: List[float] = []
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if isinstance(av, Number) and isinstance(bv, Number):
                diffs.append(abs(float(av) - float(bv)))
            else:
                if av != bv:
                    diffs.append(1.0)
        if not diffs:
            return 0.0, 0.0
        return float(max(diffs)), float(sum(diffs) / len(diffs))

    def check_feature_lineage_consistency(
        self,
        track: str,
        lineage_id: str,
        target: Optional[str] = None,
        data_version: Optional[str] = None,
        strict: bool = True,
        max_mismatch_keys: int = 20,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if target:
                    if data_version:
                        cur.execute(
                            """
                            SELECT id, target, track, lineage_id, data_version, feature_payload, created_at
                            FROM feature_snapshots
                            WHERE track = %s AND lineage_id = %s AND target = %s AND data_version = %s
                            ORDER BY created_at DESC
                            LIMIT 2
                            """,
                            (track, lineage_id, target, data_version),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, target, track, lineage_id, data_version, feature_payload, created_at
                            FROM feature_snapshots
                            WHERE track = %s AND lineage_id = %s AND target = %s
                            ORDER BY created_at DESC
                            LIMIT 2
                            """,
                            (track, lineage_id, target),
                        )
                    rows = [dict(r) for r in cur.fetchall()]
                else:
                    if data_version:
                        cur.execute(
                            """
                            WITH ranked AS (
                                SELECT
                                    id, target, track, lineage_id, data_version, feature_payload, created_at,
                                    ROW_NUMBER() OVER (PARTITION BY target ORDER BY created_at DESC) AS rn
                                FROM feature_snapshots
                                WHERE track = %s AND lineage_id = %s AND data_version = %s
                            )
                            SELECT id, target, track, lineage_id, data_version, feature_payload, created_at
                            FROM ranked
                            WHERE rn <= 2
                            ORDER BY target ASC, created_at DESC
                            """,
                            (track, lineage_id, data_version),
                        )
                    else:
                        cur.execute(
                            """
                            WITH ranked AS (
                                SELECT
                                    id, target, track, lineage_id, data_version, feature_payload, created_at,
                                    ROW_NUMBER() OVER (PARTITION BY target ORDER BY created_at DESC) AS rn
                                FROM feature_snapshots
                                WHERE track = %s AND lineage_id = %s
                            )
                            SELECT id, target, track, lineage_id, data_version, feature_payload, created_at
                            FROM ranked
                            WHERE rn <= 2
                            ORDER BY target ASC, created_at DESC
                            """,
                            (track, lineage_id),
                        )
                    rows = [dict(r) for r in cur.fetchall()]

        if len(rows) < 2:
            return {
                "passed": False,
                "compared_snapshots": len(rows),
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
                "reason": "insufficient_snapshots",
            }

        if target:
            a = rows[0].get("feature_payload") or {}
            b = rows[1].get("feature_payload") or {}
            max_abs, mean_abs = self._feature_payload_diff(a, b)
            mismatch_keys = self._feature_mismatch_keys(a, b, tolerance=tolerance, max_keys=max_mismatch_keys)
            compared_snapshots = len(rows)
        else:
            by_target: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                k = str(row.get("target") or "")
                by_target.setdefault(k, []).append(row)
            pair_max: List[float] = []
            pair_mean: List[float] = []
            compared_snapshots = 0
            mismatch_keys: List[str] = []
            for rs in by_target.values():
                if len(rs) < 2:
                    continue
                a = rs[0].get("feature_payload") or {}
                b = rs[1].get("feature_payload") or {}
                mx, mn = self._feature_payload_diff(a, b)
                pair_max.append(mx)
                pair_mean.append(mn)
                if strict:
                    mismatch_keys.extend(self._feature_mismatch_keys(a, b, tolerance=tolerance, max_keys=max_mismatch_keys))
                compared_snapshots += 2
            if not pair_max:
                return {
                    "passed": False,
                    "compared_snapshots": 0,
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                    "reason": "insufficient_snapshots",
                }
            max_abs = float(max(pair_max))
            mean_abs = float(sum(pair_mean) / len(pair_mean))
            mismatch_keys = mismatch_keys[:max_mismatch_keys]

        passed = max_abs <= tolerance if strict else mean_abs <= tolerance
        return {
            "passed": passed,
            "compared_snapshots": compared_snapshots,
            "max_abs_diff": round(max_abs, 10),
            "mean_abs_diff": round(mean_abs, 10),
            "mismatch_keys": mismatch_keys,
            "reason": "ok" if passed else "payload_mismatch",
        }

    @staticmethod
    def _feature_mismatch_keys(a: Dict[str, Any], b: Dict[str, Any], tolerance: float, max_keys: int) -> List[str]:
        out: List[str] = []
        for k in sorted(set(a.keys()) | set(b.keys())):
            av = a.get(k)
            bv = b.get(k)
            if isinstance(av, Number) and isinstance(bv, Number):
                if abs(float(av) - float(bv)) > tolerance:
                    out.append(str(k))
            elif av != bv:
                out.append(str(k))
            if len(out) >= max_keys:
                break
        return out

    def load_feature_history(
        self,
        target: str,
        track: str,
        lookback_days: int,
        data_version: Optional[str] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    if data_version:
                        cur.execute(
                            """
                            SELECT
                                target,
                                track,
                                COALESCE(as_of_ts, as_of) AS as_of_ts,
                                data_version,
                                lineage_id,
                                feature_payload,
                                COALESCE(feature_available_at, as_of_ts, as_of) AS feature_available_at
                            FROM feature_snapshots
                            WHERE target = %s AND track = %s
                              AND data_version = %s
                              AND COALESCE(as_of_ts, as_of) > NOW() - make_interval(days => %s)
                            ORDER BY COALESCE(as_of_ts, as_of) ASC
                            LIMIT %s
                            """,
                            (target, track, data_version, lookback_days, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                target,
                                track,
                                COALESCE(as_of_ts, as_of) AS as_of_ts,
                                data_version,
                                lineage_id,
                                feature_payload,
                                COALESCE(feature_available_at, as_of_ts, as_of) AS feature_available_at
                            FROM feature_snapshots
                            WHERE target = %s AND track = %s
                              AND COALESCE(as_of_ts, as_of) > NOW() - make_interval(days => %s)
                            ORDER BY COALESCE(as_of_ts, as_of) ASC
                            LIMIT %s
                            """,
                            (target, track, lookback_days, limit),
                        )
                except Exception:
                    conn.rollback()
                    if data_version:
                        cur.execute(
                            """
                            SELECT target, track, as_of_ts, data_version, lineage_id, feature_payload
                            FROM feature_snapshots
                            WHERE target = %s AND track = %s
                              AND data_version = %s
                              AND as_of_ts > NOW() - make_interval(days => %s)
                            ORDER BY as_of_ts ASC
                            LIMIT %s
                            """,
                            (target, track, data_version, lookback_days, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT target, track, as_of_ts, data_version, lineage_id, feature_payload
                            FROM feature_snapshots
                            WHERE target = %s AND track = %s
                              AND as_of_ts > NOW() - make_interval(days => %s)
                            ORDER BY as_of_ts ASC
                            LIMIT %s
                            """,
                            (target, track, lookback_days, limit),
                        )
                return [dict(r) for r in cur.fetchall()]

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

    def get_feature_payloads_window(self, track: str, offset_hours: int, window_hours: int, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT feature_payload
                    FROM feature_snapshots
                    WHERE track = %s
                      AND created_at <= NOW() - make_interval(hours => %s)
                      AND created_at > NOW() - make_interval(hours => %s)
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (track, offset_hours, offset_hours + window_hours, limit),
                )
                return [dict(r.get("feature_payload") or {}) for r in cur.fetchall()]

    def get_recent_feature_payloads(self, track: str, lookback_hours: int, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT feature_payload
                    FROM feature_snapshots
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (track, lookback_hours, limit),
                )
                return [dict(r.get("feature_payload") or {}) for r in cur.fetchall()]

    def get_backtest_metric_window(self, track: str, metric_key: str, offset_hours: int, window_hours: int) -> List[float]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT (metrics->>%s)::double precision AS metric
                    FROM backtest_runs
                    WHERE track = %s
                      AND created_at <= NOW() - make_interval(hours => %s)
                      AND created_at > NOW() - make_interval(hours => %s)
                      AND metrics ? %s
                    ORDER BY created_at DESC
                    """,
                    (metric_key, track, offset_hours, offset_hours + window_hours, metric_key),
                )
                return [float(r["metric"]) for r in cur.fetchall() if r.get("metric") is not None]

    def get_recent_backtest_metric(self, track: str, metric_key: str, lookback_hours: int) -> List[float]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT (metrics->>%s)::double precision AS metric
                    FROM backtest_runs
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                      AND metrics ? %s
                    ORDER BY created_at DESC
                    """,
                    (metric_key, track, lookback_hours, metric_key),
                )
                return [float(r["metric"]) for r in cur.fetchall() if r.get("metric") is not None]

    def get_backtest_target_pnl_window(
        self,
        track: str,
        window_hours: int,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        score_source: Optional[str] = None,
        data_regimes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cond = [
                    "track = %s",
                    "created_at > NOW() - make_interval(hours => %s)",
                    "COALESCE(metrics->>'status','') = 'completed'",
                    "superseded_by_run_id IS NULL",
                ]
                params: List[Any] = [track, window_hours]
                if include_sources:
                    cond.append("COALESCE(run_source, 'prod') = ANY(%s)")
                    params.append([s.strip() for s in include_sources if s and s.strip()])
                if exclude_sources:
                    cond.append("COALESCE(run_source, 'prod') <> ALL(%s)")
                    params.append([s.strip() for s in exclude_sources if s and s.strip()])
                if score_source:
                    cond.append("COALESCE(config->>'score_source','heuristic') = %s")
                    params.append(str(score_source).strip().lower())
                if data_regimes:
                    cond.append("COALESCE(NULLIF(config->>'data_regime',''),'missing') = ANY(%s)")
                    params.append([s.strip() for s in data_regimes if s and s.strip()])
                cur.execute(
                    f"""
                    SELECT config, metrics
                    FROM backtest_runs
                    WHERE {' AND '.join(cond)}
                    ORDER BY created_at DESC
                    """,
                    tuple(params),
                )
                out: Dict[str, Dict[str, float]] = {}
                for r in cur.fetchall():
                    cfg = r.get("config") if isinstance(r.get("config"), dict) else {}
                    metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                    targets = cfg.get("targets") if isinstance(cfg.get("targets"), list) else []
                    if not targets:
                        continue
                    per_target = metrics.get("per_target") if isinstance(metrics.get("per_target"), dict) else {}
                    for t in targets:
                        target = str(t).upper()
                        if target not in out:
                            out[target] = {"sum": 0.0, "count": 0.0}
                        pnl = 0.0
                        if isinstance(per_target.get(target), dict):
                            pnl = float((per_target.get(target) or {}).get("pnl_after_cost", 0.0) or 0.0)
                        elif target in per_target:
                            pnl = float(per_target.get(target) or 0.0)
                        else:
                            pnl = float(metrics.get("pnl_after_cost", 0.0) or 0.0) / max(1.0, float(len(targets)))
                        out[target]["sum"] += pnl
                        out[target]["count"] += 1.0
                return out

    def get_execution_target_realized_window(
        self, track: str, window_hours: int
    ) -> Dict[str, Dict[str, float]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        UPPER(target) AS target,
                        side,
                        quantity::double precision AS quantity,
                        est_price::double precision AS est_price,
                        COALESCE((metadata->'execution'->>'filled_qty')::double precision, quantity::double precision, 0.0) AS filled_qty,
                        COALESCE((metadata->'execution'->>'avg_fill_price')::double precision, est_price::double precision, 0.0) AS avg_fill_price
                    FROM orders_sim
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                      AND status IN ('filled', 'partially_filled')
                    """,
                    (track, window_hours),
                )
                out: Dict[str, Dict[str, float]] = {}
                for r in cur.fetchall():
                    target = str(r.get("target") or "").upper()
                    if not target:
                        continue
                    side = str(r.get("side") or "buy").lower()
                    filled_qty = float(r.get("filled_qty") or 0.0)
                    if filled_qty <= 0:
                        continue
                    est_price = float(r.get("est_price") or 0.0)
                    avg_fill_price = float(r.get("avg_fill_price") or 0.0)
                    if est_price <= 0:
                        continue
                    signed = 1.0 if side == "buy" else -1.0
                    realized = signed * (avg_fill_price - est_price) / max(abs(est_price), 1e-12)
                    notional = abs(filled_qty * est_price)
                    if target not in out:
                        out[target] = {"sum_weighted": 0.0, "sum_notional": 0.0, "orders": 0.0}
                    out[target]["sum_weighted"] += realized * notional
                    out[target]["sum_notional"] += notional
                    out[target]["orders"] += 1.0
                return out

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
                      AND status IN ('filled', 'partially_filled')
                    """,
                    (track, lookback_hours),
                )
                vals = []
                for r in cur.fetchall():
                    v = r.get("slippage")
                    if v is not None:
                        vals.append(float(v))
                return vals

    def get_execution_reject_rate(self, track: str, lookback_hours: int) -> float:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)::double precision AS total,
                        COALESCE(SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END), 0)::double precision AS rejected
                    FROM orders_sim
                    WHERE track = %s
                      AND created_at > NOW() - make_interval(hours => %s)
                    """,
                    (track, lookback_hours),
                )
                row = dict(cur.fetchone() or {})
        total = float(row.get("total") or 0.0)
        rejected = float(row.get("rejected") or 0.0)
        return float(rejected / max(1.0, total))

    def get_execution_reject_rate_window(self, track: str, offset_hours: int, window_hours: int) -> float:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)::double precision AS total,
                        COALESCE(SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END), 0)::double precision AS rejected
                    FROM orders_sim
                    WHERE track = %s
                      AND created_at <= NOW() - make_interval(hours => %s)
                      AND created_at > NOW() - make_interval(hours => %s)
                    """,
                    (track, offset_hours, offset_hours + window_hours),
                )
                row = dict(cur.fetchone() or {})
        total = float(row.get("total") or 0.0)
        rejected = float(row.get("rejected") or 0.0)
        return float(rejected / max(1.0, total))

    def get_execution_edge_pnls(
        self,
        track: str,
        lookback_hours: int,
        limit: int = 500,
        strategy_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if strategy_id:
                    cur.execute(
                        """
                        SELECT
                            created_at,
                            side,
                            quantity::double precision AS quantity,
                            est_price::double precision AS est_price,
                            (metadata->'execution'->>'avg_fill_price')::double precision AS avg_fill_price
                        FROM orders_sim
                        WHERE track = %s
                          AND strategy_id = %s
                          AND created_at > NOW() - make_interval(hours => %s)
                          AND status IN ('filled', 'partially_filled')
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                        """,
                        (track, strategy_id, lookback_hours, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            created_at,
                            side,
                            quantity::double precision AS quantity,
                            est_price::double precision AS est_price,
                            (metadata->'execution'->>'avg_fill_price')::double precision AS avg_fill_price
                        FROM orders_sim
                        WHERE track = %s
                          AND created_at > NOW() - make_interval(hours => %s)
                          AND status IN ('filled', 'partially_filled')
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                        """,
                        (track, lookback_hours, limit),
                    )
                rows = [dict(r) for r in cur.fetchall()]
        out: List[Dict[str, Any]] = []
        for r in rows:
            qty = float(r.get("quantity") or 0.0)
            est = r.get("est_price")
            avg = r.get("avg_fill_price")
            if qty <= 0 or est is None or avg is None:
                edge_pnl = 0.0
                notional = 0.0
            else:
                est_f = float(est)
                avg_f = float(avg)
                if est_f <= 0 or avg_f <= 0:
                    edge_pnl = 0.0
                    notional = 0.0
                else:
                    side = str(r.get("side") or "").lower()
                    sign = 1.0 if side == "buy" else -1.0
                    edge_pnl = (est_f - avg_f) * qty * sign
                    notional = abs(est_f * qty)
            out.append(
                {
                    "created_at": r.get("created_at"),
                    "edge_pnl": float(edge_pnl),
                    "notional": float(notional),
                }
            )
        return out

    def get_execution_daily_loss_ratio(self, track: str, lookback_hours: int = 24, strategy_id: Optional[str] = None) -> float:
        rows = self.get_execution_edge_pnls(track=track, lookback_hours=lookback_hours, limit=5000, strategy_id=strategy_id)
        if not rows:
            return 0.0
        net_edge = float(sum(float(r.get("edge_pnl") or 0.0) for r in rows))
        gross = float(sum(float(r.get("notional") or 0.0) for r in rows))
        if net_edge >= 0 or gross <= 1e-9:
            return 0.0
        return float(max(0.0, min(5.0, (-net_edge) / gross)))

    def get_execution_consecutive_losses(
        self,
        track: str,
        lookback_hours: int = 24,
        limit: int = 200,
        strategy_id: Optional[str] = None,
    ) -> int:
        rows = self.get_execution_edge_pnls(track=track, lookback_hours=lookback_hours, limit=limit, strategy_id=strategy_id)
        streak = 0
        for r in rows:
            pnl = float(r.get("edge_pnl") or 0.0)
            if pnl < -1e-12:
                streak += 1
            else:
                break
        return int(streak)

    def get_model_rollout_state(self, track: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM model_rollout_state WHERE track = %s LIMIT 1", (track,))
                row = cur.fetchone()
                return dict(row) if row else None

    def upsert_model_rollout_state(
        self,
        track: str,
        model_name: str,
        model_version: str,
        stage_pct: int,
        status: str,
        hard_limits: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_rollout_state (
                        track, model_name, model_version, stage_pct, status, hard_limits, metrics, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (track)
                    DO UPDATE SET
                        model_name = EXCLUDED.model_name,
                        model_version = EXCLUDED.model_version,
                        stage_pct = EXCLUDED.stage_pct,
                        status = EXCLUDED.status,
                        hard_limits = EXCLUDED.hard_limits,
                        metrics = EXCLUDED.metrics,
                        updated_at = NOW()
                    RETURNING *
                    """,
                    (track, model_name, model_version, stage_pct, status, json.dumps(hard_limits), json.dumps(metrics)),
                )
                row = cur.fetchone()
                return dict(row) if row else {}

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
