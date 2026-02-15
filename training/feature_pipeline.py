from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

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

    def check_data_quality(self, symbol: str, timeframe: str = "5m", lookback_hours: int = 48) -> Dict[str, float]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COUNT(*) AS total_rows,
                      COALESCE(SUM(CASE WHEN close IS NULL OR volume IS NULL THEN 1 ELSE 0 END), 0) AS missing_rows,
                      COALESCE(SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END), 0) AS invalid_price_rows
                    FROM market_bars
                    WHERE symbol = UPPER(%s)
                      AND timeframe = %s
                      AND ts > NOW() - make_interval(hours => %s)
                    """,
                    (symbol, timeframe, lookback_hours),
                )
                row = dict(cur.fetchone() or {})
        total = int(row.get("total_rows") or 0)
        missing = int(row.get("missing_rows") or 0)
        invalid = int(row.get("invalid_price_rows") or 0)
        return {
            "total_rows": float(total),
            "missing_rate": float(missing / max(1, total)),
            "invalid_price_rate": float(invalid / max(1, total)),
            "quality_passed": float(total > 200 and missing / max(1, total) < 0.02 and invalid / max(1, total) < 0.005),
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
            return SampleBatch(X=np.zeros((0, 8), dtype=np.float32), y=np.zeros((0,), dtype=np.float32), meta=[], extra_labels={})

        feats = []
        labels = []
        labels_1h = []
        labels_4h = []
        for i in range(12, len(rows) - 49):
            price = float(rows[i].get("price") or 0.0)
            if price <= 0:
                continue
            prev_1 = float(rows[i - 1].get("price") or price)
            prev_3 = float(rows[i - 3].get("price") or price)
            prev_12 = float(rows[i - 12].get("price") or price)
            ret_1 = (price - prev_1) / max(prev_1, 1e-12)
            ret_3 = (price - prev_3) / max(prev_3, 1e-12)
            ret_12 = (price - prev_12) / max(prev_12, 1e-12)
            window = [float(rows[j].get("price") or price) for j in range(i - 12, i)]
            vol_12 = float(np.std(np.diff(np.log(np.clip(np.array(window, dtype=np.float64), 1e-12, None)))))
            vol = float(rows[i].get("volume") or 0.0)
            vol_hist = np.array([float(rows[j].get("volume") or 0.0) for j in range(i - 12, i)], dtype=np.float64)
            vol_z = float((vol - np.mean(vol_hist)) / max(np.std(vol_hist), 1e-6))
            funding = 0.0
            onchain_flow = 0.0
            event_decay = float(np.exp(-min(48.0, i / 48.0)))
            feats.append([ret_1, ret_3, ret_12, vol_12, np.log1p(max(vol, 0.0)), vol_z, funding + onchain_flow, event_decay])

            fwd_1h = (float(rows[i + 12].get("price") or price) - price) / max(price, 1e-12)
            fwd_4h = (float(rows[i + 48].get("price") or price) - price) / max(price, 1e-12)
            est_cost = (5.0 + 3.0) / 10000.0
            labels.append(fwd_1h - est_cost)
            labels_1h.append(fwd_1h - est_cost)
            labels_4h.append(fwd_4h - est_cost * 2.0)

        return SampleBatch(
            X=np.array(feats, dtype=np.float32),
            y=np.array(labels, dtype=np.float32),
            meta=rows,
            extra_labels={
                "fwd_ret_1h": np.array(labels_1h, dtype=np.float32),
                "fwd_ret_4h": np.array(labels_4h, dtype=np.float32),
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
    ):
        lineage_id = lineage_id or uuid.uuid4().hex[:24]
        now = datetime.utcnow()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feature_snapshots (
                        target, track, as_of, as_of_ts, event_time, feature_version,
                        feature_payload, data_version, lineage_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (target, track, now, now, event_time or now, version, json.dumps(features), data_version, lineage_id),
                )
