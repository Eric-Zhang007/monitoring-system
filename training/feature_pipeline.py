from __future__ import annotations

import json
import os
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")


@dataclass
class SampleBatch:
    X: np.ndarray
    y: np.ndarray
    meta: List[Dict]
    extra_labels: Dict[str, np.ndarray] | None = None


class FeaturePipeline:
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url

    def _connect(self):
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    @staticmethod
    def _dq_thresholds() -> Dict[str, float]:
        return {
            "missing_rate_max": float(os.getenv("DQ_MAX_MISSING_RATE", "0.02")),
            "invalid_price_rate_max": float(os.getenv("DQ_MAX_INVALID_PRICE_RATE", "0.005")),
            "duplicate_rate_max": float(os.getenv("DQ_MAX_DUPLICATE_RATE", "0.02")),
            "stale_ratio_max": float(os.getenv("DQ_MAX_STALE_RATIO", "0.1")),
            "min_rows": float(os.getenv("DQ_MIN_ROWS", "200")),
        }

    @staticmethod
    def _source_tier_weights() -> Dict[int, float]:
        raw = os.getenv("SOURCE_TIER_WEIGHTS", "1=1.0,2=0.85,3=0.65,4=0.4,5=0.2")
        out: Dict[int, float] = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.4, 5: 0.2}
        for part in raw.split(","):
            piece = part.strip()
            if not piece or "=" not in piece:
                continue
            k_raw, v_raw = piece.split("=", 1)
            try:
                k = int(k_raw.strip())
                v = float(v_raw.strip())
            except Exception:
                continue
            if 1 <= k <= 5 and v >= 0:
                out[k] = v
        return out

    def check_data_quality(self, symbol: str, timeframe: str = "5m", lookback_hours: int = 48) -> Dict[str, float]:
        thresholds = self._dq_thresholds()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COUNT(*) AS total_rows,
                      COALESCE(SUM(CASE WHEN close IS NULL OR volume IS NULL THEN 1 ELSE 0 END), 0) AS missing_rows,
                      COALESCE(SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END), 0) AS invalid_price_rows,
                      COALESCE(SUM(
                        CASE
                          WHEN prev_ts IS NOT NULL AND ts = prev_ts THEN 1
                          ELSE 0
                        END
                      ), 0) AS duplicate_rows,
                      COALESCE(SUM(
                        CASE
                          WHEN prev_ts IS NOT NULL AND EXTRACT(EPOCH FROM (ts - prev_ts)) > 900 THEN 1
                          ELSE 0
                        END
                      ), 0) AS stale_gap_rows
                    FROM (
                      SELECT
                        ts, close, volume,
                        LAG(ts) OVER (ORDER BY ts ASC) AS prev_ts
                      FROM market_bars
                      WHERE symbol = UPPER(%s)
                        AND timeframe = %s
                        AND ts > NOW() - make_interval(hours => %s)
                    ) s
                    """,
                    (symbol, timeframe, lookback_hours),
                )
                row = dict(cur.fetchone() or {})
        total = int(row.get("total_rows") or 0)
        missing = int(row.get("missing_rows") or 0)
        invalid = int(row.get("invalid_price_rows") or 0)
        dup = int(row.get("duplicate_rows") or 0)
        stale = int(row.get("stale_gap_rows") or 0)
        missing_rate = float(missing / max(1, total))
        invalid_rate = float(invalid / max(1, total))
        dup_rate = float(dup / max(1, total))
        stale_ratio = float(stale / max(1, total))
        passed = (
            total >= int(thresholds["min_rows"])
            and missing_rate <= thresholds["missing_rate_max"]
            and invalid_rate <= thresholds["invalid_price_rate_max"]
            and dup_rate <= thresholds["duplicate_rate_max"]
            and stale_ratio <= thresholds["stale_ratio_max"]
        )
        return {
            "total_rows": float(total),
            "missing_rate": missing_rate,
            "invalid_price_rate": invalid_rate,
            "duplicate_rate": dup_rate,
            "stale_ratio": stale_ratio,
            "quality_passed": float(1.0 if passed else 0.0),
        }

    def load_vc_training_batch(self, limit: int = 2000) -> SampleBatch:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.event_type, e.source_tier, e.confidence_score,
                           EXTRACT(EPOCH FROM (NOW() - e.occurred_at))/86400.0 AS recency_days
                    FROM events e
                    ORDER BY e.occurred_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return SampleBatch(X=np.zeros((0, 5), dtype=np.float32), y=np.zeros((0,), dtype=np.float32), meta=[])

        event_map = {"funding": 1.0, "product": 0.7, "mna": 0.9, "regulatory": -0.8, "market": 0.2}
        feats = []
        labels = []
        for r in rows:
            et = r.get("event_type", "market")
            recency = float(r.get("recency_days") or 0.0)
            source_tier = float(r.get("source_tier") or 3)
            confidence = float(r.get("confidence_score") or 0.5)
            f = [event_map.get(et, 0.0), source_tier / 5.0, confidence, max(0.0, 1.0 - recency / 365.0), recency / 365.0]
            feats.append(f)
            labels.append(1.0 if et in {"funding", "mna"} else 0.0)

        return SampleBatch(
            X=np.array(feats, dtype=np.float32),
            y=np.array(labels, dtype=np.float32),
            meta=rows,
        )

    def load_liquid_training_batch(self, symbol: str, limit: int = 2000) -> SampleBatch:
        rows: List[Dict] = []
        orderbook_rows: List[Dict] = []
        funding_rows: List[Dict] = []
        onchain_rows: List[Dict] = []
        event_rows: List[Dict] = []
        global_event_rows: List[Dict] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT symbol, close::float AS price, volume::float AS volume, ts AS timestamp
                        FROM market_bars
                        WHERE symbol = UPPER(%s)
                          AND timeframe = '5m'
                        ORDER BY ts DESC
                        LIMIT %s
                        """,
                        (symbol, limit),
                    )
                    rows = [dict(r) for r in cur.fetchall()]
                    cur.execute(
                        """
                        SELECT ts AS timestamp, imbalance::float AS imbalance
                        FROM orderbook_l2
                        WHERE symbol = UPPER(%s)
                        ORDER BY ts ASC
                        LIMIT %s
                        """,
                        (symbol, limit * 2),
                    )
                    orderbook_rows = [dict(r) for r in cur.fetchall()]
                    cur.execute(
                        """
                        SELECT ts AS timestamp, funding_rate::float AS funding_rate
                        FROM funding_rates
                        WHERE symbol = UPPER(%s)
                        ORDER BY ts ASC
                        LIMIT %s
                        """,
                        (symbol, max(200, limit // 4)),
                    )
                    funding_rows = [dict(r) for r in cur.fetchall()]
                    cur.execute(
                        """
                        SELECT ts AS timestamp, metric_value::float AS metric_value
                        FROM onchain_signals
                        WHERE asset_symbol = UPPER(%s)
                          AND metric_name IN ('netflow','exchange_netflow','net_inflow')
                        ORDER BY ts ASC
                        LIMIT %s
                        """,
                        (symbol, limit * 2),
                    )
                    onchain_rows = [dict(r) for r in cur.fetchall()]
                    try:
                        cur.execute(
                            """
                            SELECT COALESCE(e.available_at, e.occurred_at) AS timestamp, e.source_tier, e.confidence_score, 1.0::double precision AS scope_weight
                            FROM events e
                            JOIN event_links el ON el.event_id = e.id
                            JOIN entities en ON en.id = el.entity_id
                            WHERE UPPER(en.symbol) = UPPER(%s)
                              AND COALESCE(en.metadata->>'synthetic_link', 'false') <> 'true'
                            ORDER BY COALESCE(e.available_at, e.occurred_at) ASC
                            LIMIT %s
                            """,
                            (symbol, limit * 2),
                        )
                    except Exception:
                        cur.execute(
                            """
                            SELECT e.occurred_at AS timestamp, e.source_tier, e.confidence_score, 1.0::double precision AS scope_weight
                            FROM events e
                            JOIN event_links el ON el.event_id = e.id
                            JOIN entities en ON en.id = el.entity_id
                            WHERE UPPER(en.symbol) = UPPER(%s)
                              AND COALESCE(en.metadata->>'synthetic_link', 'false') <> 'true'
                            ORDER BY e.occurred_at ASC
                            LIMIT %s
                            """,
                            (symbol, limit * 2),
                        )
                    event_rows = [dict(r) for r in cur.fetchall()]
                    try:
                        cur.execute(
                            """
                            SELECT
                                COALESCE(e.available_at, e.occurred_at) AS timestamp,
                                e.source_tier,
                                e.confidence_score,
                                0.7::double precision AS scope_weight
                            FROM events e
                            WHERE e.market_scope = 'macro'
                               OR COALESCE(e.payload->>'global_impact', 'false') = 'true'
                            ORDER BY COALESCE(e.available_at, e.occurred_at) ASC
                            LIMIT %s
                            """,
                            (max(50, limit // 8),),
                        )
                        global_event_rows = [dict(r) for r in cur.fetchall()]
                    except Exception:
                        global_event_rows = []
                except Exception:
                    rows = []
                if not rows:
                    cur.execute(
                        """
                        SELECT symbol, price::float AS price, volume::float AS volume, timestamp
                        FROM prices
                        WHERE symbol = UPPER(%s)
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (symbol, limit),
                    )
                    rows = [dict(r) for r in cur.fetchall()]

        rows = sorted(rows, key=lambda r: r["timestamp"])

        if len(rows) < 10:
            return SampleBatch(X=np.zeros((0, 18), dtype=np.float32), y=np.zeros((0,), dtype=np.float32), meta=[], extra_labels={})

        ob_ts = [r["timestamp"] for r in orderbook_rows]
        ob_vals = [float(r.get("imbalance") or 0.0) for r in orderbook_rows]
        fr_ts = [r["timestamp"] for r in funding_rows]
        fr_vals = [float(r.get("funding_rate") or 0.0) for r in funding_rows]
        oc_ts = [r["timestamp"] for r in onchain_rows]
        oc_vals = [float(r.get("metric_value") or 0.0) for r in onchain_rows]
        min_event_conf = float(os.getenv("EVENT_MIN_CONFIDENCE", "0.0"))
        max_event_tier = int(os.getenv("EVENT_MAX_SOURCE_TIER", "5"))
        tier_weights = self._source_tier_weights()
        ev_rows = []
        for r in event_rows + global_event_rows:
            tier = int(r.get("source_tier") or 5)
            conf = float(r.get("confidence_score") or 0.0)
            if conf < min_event_conf or tier > max_event_tier:
                continue
            ev_rows.append(
                {
                    "timestamp": r["timestamp"],
                    "tier": tier,
                    "confidence": conf * float(r.get("scope_weight") or 1.0),
                    "tier_weight": float(tier_weights.get(tier, 0.1)),
                }
            )
        ev_ts = [r["timestamp"] for r in ev_rows]

        def latest_before(ts_list, values, ts):
            if not ts_list:
                return 0.0
            idx = bisect_right(ts_list, ts) - 1
            if idx < 0:
                return 0.0
            return float(values[idx])

        feats = []
        labels = []
        labels_1h = []
        labels_4h = []
        labels_cost = []
        for i in range(96, len(rows) - 49):
            price = float(rows[i].get("price") or 0.0)
            if price <= 0:
                continue
            prev_1 = float(rows[i - 1].get("price") or price)
            prev_3 = float(rows[i - 3].get("price") or price)
            prev_12 = float(rows[i - 12].get("price") or price)
            prev_48 = float(rows[i - 48].get("price") or price)
            ret_1 = (price - prev_1) / max(prev_1, 1e-12)
            ret_3 = (price - prev_3) / max(prev_3, 1e-12)
            ret_12 = (price - prev_12) / max(prev_12, 1e-12)
            ret_48 = (price - prev_48) / max(prev_48, 1e-12)
            w3 = np.array([float(rows[j].get("price") or price) for j in range(i - 3, i)], dtype=np.float64)
            w12 = np.array([float(rows[j].get("price") or price) for j in range(i - 12, i)], dtype=np.float64)
            w48 = np.array([float(rows[j].get("price") or price) for j in range(i - 48, i)], dtype=np.float64)
            w96 = np.array([float(rows[j].get("price") or price) for j in range(i - 96, i)], dtype=np.float64)
            vol_3 = float(np.std(np.diff(np.log(np.clip(w3, 1e-12, None)))))
            vol_12 = float(np.std(np.diff(np.log(np.clip(w12, 1e-12, None)))))
            vol_48 = float(np.std(np.diff(np.log(np.clip(w48, 1e-12, None)))))
            vol_96 = float(np.std(np.diff(np.log(np.clip(w96, 1e-12, None)))))
            vol = float(rows[i].get("volume") or 0.0)
            vol_hist = np.array([float(rows[j].get("volume") or 0.0) for j in range(i - 12, i)], dtype=np.float64)
            vol_z = float((vol - np.mean(vol_hist)) / max(np.std(vol_hist), 1e-6))
            volume_impact = float(abs(ret_1) / max(np.sqrt(max(vol, 1.0)), 1e-6))
            ts = rows[i]["timestamp"]
            orderbook_imbalance = latest_before(ob_ts, ob_vals, ts)
            funding = latest_before(fr_ts, fr_vals, ts)
            onchain_flow = latest_before(oc_ts, oc_vals, ts)
            onchain_norm = float(np.tanh(onchain_flow / 1e6))
            evt_idx = bisect_right(ev_ts, ts) - 1 if ev_ts else -1
            if evt_idx >= 0:
                left_idx = bisect_right(ev_ts, ts - timedelta(hours=24))
                num = 0.0
                den = 0.0
                for j in range(left_idx, evt_idx + 1):
                    evt = ev_rows[j]
                    evt_ts = evt["timestamp"]
                    age_hours = max(0.0, float((ts - evt_ts).total_seconds()) / 3600.0)
                    decay = float(np.exp(-age_hours / 12.0))
                    ew = float(evt["tier_weight"]) * float(evt["confidence"])
                    num += ew * decay
                    den += ew
                event_decay = float(num / max(1e-9, den))
            else:
                event_decay = 0.0
            orderbook_missing_flag = 0.0 if ob_ts else 1.0
            funding_missing_flag = 0.0 if fr_ts else 1.0
            onchain_missing_flag = 0.0 if oc_ts else 1.0
            feats.append(
                [
                    ret_1,
                    ret_3,
                    ret_12,
                    ret_48,
                    vol_3,
                    vol_12,
                    vol_48,
                    vol_96,
                    np.log1p(max(vol, 0.0)),
                    vol_z,
                    volume_impact,
                    orderbook_imbalance,
                    funding,
                    onchain_norm,
                    event_decay,
                    orderbook_missing_flag,
                    funding_missing_flag,
                    onchain_missing_flag,
                ]
            )

            fwd_1h = (float(rows[i + 12].get("price") or price) - price) / max(price, 1e-12)
            fwd_4h = (float(rows[i + 48].get("price") or price) - price) / max(price, 1e-12)
            est_cost = (5.0 + 3.0) / 10000.0 + min(0.002, 0.5 * abs(orderbook_imbalance) / 1000.0)
            labels.append(fwd_1h - est_cost)
            labels_1h.append(fwd_1h - est_cost)
            labels_4h.append(fwd_4h - est_cost * 2.0)
            labels_cost.append(est_cost)

        return SampleBatch(
            X=np.array(feats, dtype=np.float32),
            y=np.array(labels, dtype=np.float32),
            meta=rows,
            extra_labels={
                "fwd_ret_1h": np.array(labels_1h, dtype=np.float32),
                "fwd_ret_4h": np.array(labels_4h, dtype=np.float32),
                "est_cost": np.array(labels_cost, dtype=np.float32),
            },
        )

    def save_feature_snapshot(
        self,
        target: str,
        track: str,
        features: Dict[str, float],
        version: str = "feature-store-v2.0",
        lineage_id: Optional[str] = None,
        data_version: str = "v1",
        event_time: Optional[datetime] = None,
    ) -> str:
        lineage_id = lineage_id or uuid.uuid4().hex[:24]
        now = datetime.utcnow()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feature_snapshots (
                        target, track, as_of, as_of_ts, event_time, feature_available_at, feature_version,
                        feature_payload, data_version, lineage_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (target, track, now, now, event_time or now, now, version, json.dumps(features), data_version, lineage_id),
                )
        return lineage_id

    def save_feature_snapshots_bulk(
        self,
        target: str,
        track: str,
        feature_rows: List[Dict[str, float]],
        version: str = "feature-store-v2.0",
        lineage_id: Optional[str] = None,
        data_version: str = "v1",
        event_time: Optional[datetime] = None,
    ) -> str:
        lineage_id = lineage_id or uuid.uuid4().hex[:24]
        if not feature_rows:
            return lineage_id
        now = datetime.utcnow()
        rows = [
            (
                target,
                track,
                now,
                now,
                event_time or now,
                now,
                version,
                json.dumps(f),
                data_version,
                lineage_id,
            )
            for f in feature_rows
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    execute_values(
                        cur,
                        """
                        INSERT INTO feature_snapshots (
                            target, track, as_of, as_of_ts, event_time, feature_available_at, feature_version,
                            feature_payload, data_version, lineage_id, created_at
                        ) VALUES %s
                        """,
                        rows,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                    )
                except Exception:
                    rows_old = [r[:5] + r[6:] for r in rows]
                    execute_values(
                        cur,
                        """
                        INSERT INTO feature_snapshots (
                            target, track, as_of, as_of_ts, event_time, feature_version,
                            feature_payload, data_version, lineage_id, created_at
                        ) VALUES %s
                        """,
                        rows_old,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                    )
        return lineage_id
