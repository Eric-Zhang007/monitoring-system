"""
V2 Data Collector
- Plugin-based connectors (GDELT / RSS / SEC)
- Canonical event schema
- Redis Streams as real-time buffer
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, List

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from connectors import (
    EarningsAlphaVantageConnector,
    GDELTConnector,
    MacroFREDConnector,
    OnChainCoinGeckoConnector,
    RSSConnector,
    SECSubmissionsConnector,
)
from connectors.base import RateLimitError
from entity_linking import extract_entities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

COLLECTOR_CONNECTOR_FETCH_TOTAL = Counter(
    "ms_collector_connector_fetch_total",
    "Collector connector fetch outcomes by connector and status.",
    ["connector", "status"],
)
COLLECTOR_CONNECTOR_EMPTY_RESULTS_TOTAL = Counter(
    "ms_collector_connector_empty_results_total",
    "Collector connector empty-result count by connector.",
    ["connector"],
)
COLLECTOR_CONNECTOR_RATE_LIMIT_TOTAL = Counter(
    "ms_collector_connector_rate_limit_total",
    "Collector connector rate-limit count by connector.",
    ["connector"],
)
COLLECTOR_CONNECTOR_FETCH_LATENCY_SECONDS = Histogram(
    "ms_collector_connector_fetch_latency_seconds",
    "Collector connector fetch latency by connector.",
    ["connector"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30),
)
COLLECTOR_CONNECTOR_COOLDOWN_SKIPS_TOTAL = Counter(
    "ms_collector_connector_cooldown_skips_total",
    "Collector connector fetch skips due to cooldown by connector.",
    ["connector"],
)
COLLECTOR_CONNECTOR_CONSECUTIVE_FAILURES = Gauge(
    "ms_collector_connector_consecutive_failures",
    "Collector connector current consecutive failures.",
    ["connector"],
)
COLLECTOR_EVENT_PUBLISH_TOTAL = Counter(
    "ms_collector_event_publish_total",
    "Collector event publish outcomes by connector and status.",
    ["connector", "status"],
)
COLLECTOR_SOURCE_PUBLISH_TO_INGEST_SECONDS = Histogram(
    "ms_collector_source_publish_to_ingest_seconds",
    "Latency between source publish timestamp and collector ingest timestamp.",
    ["connector"],
    buckets=(1, 5, 10, 30, 60, 120, 180, 300, 600, 1200),
)


class DataCollectorV2:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.api_base = os.getenv("API_BASE", "http://backend:8000").rstrip("/")
        self.ingest_endpoint = f"{self.api_base}/api/v2/ingest/events"
        self.ingest_timeout_sec = float(os.getenv("INGEST_TIMEOUT_SEC", "8"))
        self.ingest_max_retries = int(os.getenv("INGEST_MAX_RETRIES", "3"))
        self.fetch_max_retries = int(os.getenv("CONNECTOR_MAX_RETRIES", "3"))
        self.fetch_backoff_base_sec = float(os.getenv("CONNECTOR_BACKOFF_BASE_SEC", "0.5"))
        self.fetch_backoff_max_sec = float(os.getenv("CONNECTOR_BACKOFF_MAX_SEC", "10.0"))
        self.fetch_failure_threshold = int(os.getenv("CONNECTOR_FAILURE_THRESHOLD", "3"))
        self.fetch_cooldown_sec = float(os.getenv("CONNECTOR_COOLDOWN_SEC", "120.0"))
        self.metrics_port = int(os.getenv("COLLECTOR_METRICS_PORT", "9101"))
        self.liquid_symbols = {s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK").split(",") if s.strip()}
        self.connector_state: Dict[str, Dict[str, float]] = {}

        rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss",
        ]
        sec_ciks = ["320193", "1045810", "789019"]  # Apple/NVDA/MSFT
        alpha_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        coingecko_ids = [
            s.strip().lower()
            for s in os.getenv(
                "ONCHAIN_IDS",
                "bitcoin,ethereum,solana,binancecoin,ripple,cardano,dogecoin,tron,avalanche-2,chainlink",
            ).split(",")
            if s.strip()
        ]

        self.connectors = [
            GDELTConnector(query="venture capital startup funding OR IPO", max_records=25),
            RSSConnector(feeds=rss_feeds),
            SECSubmissionsConnector(cik_list=sec_ciks),
            MacroFREDConnector(),
            OnChainCoinGeckoConnector(ids=coingecko_ids),
        ]
        if alpha_key:
            self.connectors.append(EarningsAlphaVantageConnector(api_key=alpha_key, horizon=os.getenv("EARNINGS_HORIZON", "3month")))
        try:
            start_http_server(self.metrics_port)
            logger.info("collector metrics server started on :%s", self.metrics_port)
        except Exception as exc:
            logger.warning("collector metrics server not started: %s", exc)

    def _state(self, connector_name: str) -> Dict[str, float]:
        if connector_name not in self.connector_state:
            self.connector_state[connector_name] = {
                "success": 0.0,
                "failure": 0.0,
                "empty_result_count": 0.0,
                "rate_limit_count": 0.0,
                "cooldown_skips": 0.0,
                "consecutive_failures": 0.0,
                "cooldown_until_ts": 0.0,
                "last_fetch_latency_sec": 0.0,
                "last_fetch_status": 0.0,  # 1=ok,0=failed
            }
        return self.connector_state[connector_name]

    @staticmethod
    def _tier_weight(source_tier: int) -> float:
        return max(0.1, min(1.0, (6.0 - float(source_tier)) / 5.0))

    def _fetch_connector_rows(self, connector) -> tuple[List[Dict], str]:
        state = self._state(connector.name)
        now_ts = time.time()
        if state["cooldown_until_ts"] > now_ts:
            state["cooldown_skips"] += 1.0
            COLLECTOR_CONNECTOR_COOLDOWN_SKIPS_TOTAL.labels(connector=connector.name).inc()
            COLLECTOR_CONNECTOR_FETCH_TOTAL.labels(connector=connector.name, status="cooldown").inc()
            return [], "cooldown"
        delay = max(0.05, self.fetch_backoff_base_sec)
        last_exc = None
        for attempt in range(1, self.fetch_max_retries + 1):
            started = time.perf_counter()
            try:
                rows = connector.fetch()
                latency = float(max(0.0, time.perf_counter() - started))
                state["last_fetch_latency_sec"] = latency
                state["last_fetch_status"] = 1.0
                state["success"] += 1.0
                state["consecutive_failures"] = 0.0
                COLLECTOR_CONNECTOR_CONSECUTIVE_FAILURES.labels(connector=connector.name).set(0.0)
                COLLECTOR_CONNECTOR_FETCH_LATENCY_SECONDS.labels(connector=connector.name).observe(latency)
                if not rows:
                    state["empty_result_count"] += 1.0
                    COLLECTOR_CONNECTOR_EMPTY_RESULTS_TOTAL.labels(connector=connector.name).inc()
                    COLLECTOR_CONNECTOR_FETCH_TOTAL.labels(connector=connector.name, status="empty").inc()
                    return [], "empty"
                COLLECTOR_CONNECTOR_FETCH_TOTAL.labels(connector=connector.name, status="success").inc()
                return rows, "success"
            except RateLimitError as exc:
                last_exc = exc
                state["rate_limit_count"] += 1.0
                state["failure"] += 1.0
                state["last_fetch_status"] = 0.0
                state["consecutive_failures"] += 1.0
                COLLECTOR_CONNECTOR_RATE_LIMIT_TOTAL.labels(connector=connector.name).inc()
                COLLECTOR_CONNECTOR_FETCH_TOTAL.labels(connector=connector.name, status="failure").inc()
                COLLECTOR_CONNECTOR_CONSECUTIVE_FAILURES.labels(connector=connector.name).set(state["consecutive_failures"])
            except Exception as exc:
                last_exc = exc
                state["failure"] += 1.0
                state["last_fetch_status"] = 0.0
                state["consecutive_failures"] += 1.0
                COLLECTOR_CONNECTOR_FETCH_TOTAL.labels(connector=connector.name, status="failure").inc()
                COLLECTOR_CONNECTOR_CONSECUTIVE_FAILURES.labels(connector=connector.name).set(state["consecutive_failures"])
            if state["consecutive_failures"] >= float(self.fetch_failure_threshold):
                state["cooldown_until_ts"] = time.time() + self.fetch_cooldown_sec
                break
            if attempt < self.fetch_max_retries:
                jitter = 1.0 + random.uniform(0.0, 0.2)
                time.sleep(min(self.fetch_backoff_max_sec, delay * jitter))
                delay = min(self.fetch_backoff_max_sec, delay * 2.0)
        if last_exc is not None:
            logger.warning("connector=%s fetch failed after retries err=%s", connector.name, last_exc)
        return [], "failure"

    def _normalize_entities(self, entities: List[Dict], market_scope: str) -> List[Dict]:
        out: List[Dict] = []
        seen_symbols = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            symbol = str(entity.get("symbol") or "").strip().upper()
            if str(entity.get("entity_type") or "") == "asset" and not symbol:
                continue
            if market_scope == "crypto" and symbol and symbol not in self.liquid_symbols:
                continue
            e = dict(entity)
            if symbol:
                e["symbol"] = symbol
                seen_symbols.add(symbol)
            out.append(e)
        if market_scope == "crypto" and not seen_symbols:
            # Wide fallback mapping to keep liquid feature context populated.
            for sym in sorted(self.liquid_symbols):
                out.append(
                    {
                        "entity_type": "asset",
                        "name": sym,
                        "symbol": sym,
                        "country": None,
                        "sector": "crypto",
                        "metadata": {
                            "source": "broad_crypto_fallback",
                            "reason": "no_symbol_detected",
                            "synthetic_link": True,
                        },
                    }
                )
        return out

    def publish_event(self, event: Dict, connector_name: str, fetch_status: str):
        state = self._state(connector_name)
        occurred_at = event["occurred_at"]
        now_iso = datetime.utcnow().isoformat() + "Z"
        dedup_key = f"{event.get('source_name', '')}|{event.get('title', '').strip().lower()}|{event.get('source_url', '')}"
        dedup_cluster_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()[:24]
        latency_ms = None
        try:
            latency_ms = int((datetime.utcnow() - datetime.fromisoformat(occurred_at.replace("Z", "+00:00")).replace(tzinfo=None)).total_seconds() * 1000)
            if latency_ms < 0:
                latency_ms = 0
        except Exception:
            latency_ms = None
        if latency_ms is not None:
            COLLECTOR_SOURCE_PUBLISH_TO_INGEST_SECONDS.labels(connector=connector_name).observe(float(latency_ms) / 1000.0)

        market_scope = str(event.get("market_scope") or "crypto").strip().lower()
        enriched_entities = event.get("entities", []) or extract_entities(
            event.get("title", ""),
            str(event.get("payload", {}).get("summary", "")),
        )
        enriched_entities = self._normalize_entities(enriched_entities, market_scope=market_scope)

        event_payload = dict(event.get("payload", {}))
        event_payload["collector_ingested_at"] = datetime.utcnow().isoformat() + "Z"
        event_payload["ingest_dedup_key"] = dedup_key
        event_payload["source_fetch_status"] = fetch_status
        event_payload["source_confidence"] = float(event.get("confidence_score", 0.5) or 0.5)
        event_payload["source_tier_weight"] = self._tier_weight(int(event.get("source_tier", 3) or 3))
        event_payload["source_health"] = {
            "success": int(state.get("success", 0.0)),
            "failure": int(state.get("failure", 0.0)),
            "empty_result_count": int(state.get("empty_result_count", 0.0)),
            "rate_limit_count": int(state.get("rate_limit_count", 0.0)),
            "cooldown_skips": int(state.get("cooldown_skips", 0.0)),
            "last_fetch_latency_sec": float(state.get("last_fetch_latency_sec", 0.0)),
        }

        req_body = {
            "events": [
                {
                    "event_type": event["event_type"],
                    "title": event["title"],
                    "occurred_at": occurred_at,
                    "published_at": event.get("published_at") or occurred_at,
                    "ingested_at": now_iso,
                    "available_at": event.get("available_at") or now_iso,
                    "effective_at": event.get("effective_at") or now_iso,
                    "source_url": event.get("source_url") or "",
                    "source_name": event.get("source_name") or "",
                    "source_timezone": event.get("source_timezone", "UTC"),
                    "source_tier": int(event.get("source_tier", 3)),
                    "confidence_score": float(event.get("confidence_score", 0.5)),
                    "event_importance": float(event.get("event_importance", event.get("confidence_score", 0.5))),
                    "novelty_score": float(event.get("novelty_score", 0.5)),
                    "entity_confidence": float(event.get("entity_confidence", 0.5)),
                    "latency_ms": int(latency_ms or 0),
                    "source_latency_ms": int(event.get("source_latency_ms", latency_ms or 0) or 0),
                    "dedup_cluster_id": event.get("dedup_cluster_id") or dedup_cluster_id,
                    "market_scope": market_scope if market_scope in {"crypto", "equity", "macro"} else "crypto",
                    "payload": event_payload,
                    "entities": enriched_entities,
                }
            ]
        }
        delay = 0.5
        for attempt in range(1, self.ingest_max_retries + 1):
            try:
                resp = requests.post(self.ingest_endpoint, json=req_body, timeout=self.ingest_timeout_sec)
                resp.raise_for_status()
                COLLECTOR_EVENT_PUBLISH_TOTAL.labels(connector=connector_name, status="success").inc()
                return
            except Exception as exc:
                if attempt >= self.ingest_max_retries:
                    COLLECTOR_EVENT_PUBLISH_TOTAL.labels(connector=connector_name, status="failed").inc()
                    raise RuntimeError(f"ingest failed after retries: {exc}") from exc
                time.sleep(delay)
                delay = min(5.0, delay * 2.0)

    def run_once(self) -> int:
        total = 0
        for connector in self.connectors:
            try:
                raw_rows, fetch_status = self._fetch_connector_rows(connector)
                logger.info("connector=%s fetched=%d status=%s", connector.name, len(raw_rows), fetch_status)
                if fetch_status in {"failure", "cooldown"}:
                    continue
                for raw in raw_rows:
                    event = connector.normalize(raw)
                    try:
                        self.publish_event(event, connector_name=connector.name, fetch_status=fetch_status)
                        total += 1
                    except Exception as exc:
                        logger.error("connector=%s publish failed: %s", connector.name, exc)
            except Exception as exc:
                logger.error("connector=%s failed: %s", connector.name, exc)
        return total

    def run(self, interval: int = 300):
        logger.info("collector-v2 started interval=%ss", interval)
        while True:
            try:
                sent = self.run_once()
                logger.info("cycle published=%d", sent)
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("collector-v2 stopped")
                break
            except Exception as exc:
                logger.error("collector-v2 cycle error: %s", exc)
                time.sleep(30)


if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    interval = int(os.getenv("COLLECT_INTERVAL", "300"))
    DataCollectorV2(redis_url=redis_url).run(interval=interval)
