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
from typing import Any, Dict, List

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from connectors import (
    EarningsAlphaVantageConnector,
    GDELTConnector,
    MacroFREDConnector,
    OnChainCoinGeckoConnector,
    RedditConnector,
    RSSConnector,
    SECSubmissionsConnector,
    TelegramConnector,
    XTwitterConnector,
    YouTubeConnector,
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

SOCIAL_PAYLOAD_DEFAULTS: Dict[str, Any] = {
    "social_platform": "none",
    "author": "",
    "author_followers": 0,
    "engagement_score": 0.0,
    "comment_sentiment": 0.0,
    "post_sentiment": 0.0,
    "n_comments": 0,
    "n_replies": 0,
    "is_verified": False,
    "influence_tier": "unknown",
    "symbol_mentions": [],
}


class DataCollectorV2:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.api_base = os.getenv("API_BASE", "http://backend:8000").rstrip("/")
        self.ingest_endpoint = f"{self.api_base}/api/v2/ingest/events"
        self.ingest_timeout_sec = float(os.getenv("INGEST_TIMEOUT_SEC", "8"))
        self.ingest_connect_timeout_sec = float(os.getenv("INGEST_CONNECT_TIMEOUT_SEC", "3"))
        self.ingest_read_timeout_sec = float(os.getenv("INGEST_READ_TIMEOUT_SEC", str(self.ingest_timeout_sec)))
        self.ingest_max_retries = int(os.getenv("INGEST_MAX_RETRIES", "3"))
        self.ingest_http_max_retries = int(os.getenv("INGEST_HTTP_MAX_RETRIES", "2"))
        self.ingest_http_backoff_sec = float(os.getenv("INGEST_HTTP_BACKOFF_SEC", "0.5"))
        self.fetch_max_retries = int(os.getenv("CONNECTOR_MAX_RETRIES", "3"))
        self.fetch_backoff_base_sec = float(os.getenv("CONNECTOR_BACKOFF_BASE_SEC", "0.5"))
        self.fetch_backoff_max_sec = float(os.getenv("CONNECTOR_BACKOFF_MAX_SEC", "10.0"))
        self.fetch_failure_threshold = int(os.getenv("CONNECTOR_FAILURE_THRESHOLD", "3"))
        self.fetch_cooldown_sec = float(os.getenv("CONNECTOR_COOLDOWN_SEC", "120.0"))
        self.cycle_error_backoff_sec = float(os.getenv("COLLECTOR_CYCLE_ERROR_BACKOFF_SEC", "30.0"))
        self.cycle_error_backoff_max_sec = float(os.getenv("COLLECTOR_CYCLE_ERROR_BACKOFF_MAX_SEC", "180.0"))
        self.metrics_port = int(os.getenv("COLLECTOR_METRICS_PORT", "9101"))
        self.liquid_symbols = {s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK").split(",") if s.strip()}
        self.connector_state: Dict[str, Dict[str, float]] = {}
        self._http = self._build_http_session()

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
        self.connectors.extend(self._build_social_connectors())
        logger.info("collector connectors enabled=%s", [c.name for c in self.connectors])
        try:
            start_http_server(self.metrics_port)
            logger.info("collector metrics server started on :%s", self.metrics_port)
        except Exception as exc:
            logger.warning("collector metrics server not started: %s", exc)

    def _build_http_session(self) -> requests.Session:
        sess = requests.Session()
        retries = Retry(
            total=max(0, int(self.ingest_http_max_retries)),
            connect=max(0, int(self.ingest_http_max_retries)),
            read=max(0, int(self.ingest_http_max_retries)),
            backoff_factor=max(0.0, float(self.ingest_http_backoff_sec)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        return sess

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

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _build_social_connectors(self) -> List[Any]:
        enabled = self._env_flag("ENABLE_SOCIAL_CONNECTORS", default=False)
        raw_sources = [s.strip().lower() for s in os.getenv("SOCIAL_CONNECTORS", "").split(",") if s.strip()]
        if not enabled and not raw_sources:
            return []
        selected = set(raw_sources) if raw_sources else {"twitter", "reddit", "youtube", "telegram"}
        social_connectors: List[Any] = []

        if "twitter" in selected or "x" in selected:
            twitter_query = os.getenv(
                "TWITTER_QUERY",
                "(bitcoin OR ethereum OR solana OR $BTC OR $ETH OR $SOL) lang:en -is:retweet",
            )
            if not self._env_flag("TWITTER_INCLUDE_REPLIES", default=False):
                twitter_query += " -is:reply"
            social_connectors.append(
                XTwitterConnector(
                    bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
                    query=twitter_query,
                    max_results=int(os.getenv("TWITTER_MAX_RESULTS", "25")),
                    known_symbols=sorted(self.liquid_symbols),
                )
            )

        if "reddit" in selected:
            social_connectors.append(
                RedditConnector(
                    subreddits=[
                        s.strip()
                        for s in os.getenv("REDDIT_SUBREDDITS", "cryptocurrency,bitcoin,ethereum,solana").split(",")
                        if s.strip()
                    ],
                    sort=os.getenv("REDDIT_SORT", "new"),
                    limit=int(os.getenv("REDDIT_LIMIT", "25")),
                    user_agent=os.getenv("REDDIT_USER_AGENT", "monitoring-system/2.0"),
                    mode=os.getenv("REDDIT_MODE", "public"),
                    client_id=os.getenv("REDDIT_CLIENT_ID", ""),
                    client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
                    fetch_comments=self._env_flag("REDDIT_FETCH_COMMENTS", default=True),
                    comments_per_post=int(os.getenv("REDDIT_COMMENTS_PER_POST", "20")),
                    max_posts_for_comments=int(os.getenv("REDDIT_MAX_POSTS_FOR_COMMENTS", "80")),
                )
            )

        if "youtube" in selected:
            social_connectors.append(
                YouTubeConnector(
                    mode=os.getenv("YOUTUBE_MODE", "rss"),
                    channel_ids=[
                        c.strip()
                        for c in os.getenv("YOUTUBE_CHANNEL_IDS", "").split(",")
                        if c.strip()
                    ],
                    query=os.getenv("YOUTUBE_QUERY", "crypto bitcoin ethereum solana"),
                    api_key=os.getenv("YOUTUBE_API_KEY", ""),
                    max_results=int(os.getenv("YOUTUBE_MAX_RESULTS", "20")),
                )
            )

        if "telegram" in selected:
            social_connectors.append(
                TelegramConnector(
                    feeds=[
                        f.strip()
                        for f in os.getenv("TELEGRAM_FEEDS", "").split(",")
                        if f.strip()
                    ]
                )
            )

        return social_connectors

    @staticmethod
    def _normalize_social_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload or {})
        for key, default in SOCIAL_PAYLOAD_DEFAULTS.items():
            if key not in out or out.get(key) is None:
                out[key] = list(default) if isinstance(default, list) else default
        try:
            out["author_followers"] = max(0, int(float(out.get("author_followers", 0) or 0)))
        except (TypeError, ValueError):
            out["author_followers"] = 0
        for key in ("engagement_score", "comment_sentiment", "post_sentiment"):
            try:
                out[key] = float(out.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                out[key] = 0.0
        for key in ("n_comments", "n_replies"):
            try:
                out[key] = max(0, int(float(out.get(key, 0) or 0)))
            except (TypeError, ValueError):
                out[key] = 0
        out["is_verified"] = bool(out.get("is_verified", False))
        out["author"] = str(out.get("author") or "")
        out["influence_tier"] = str(out.get("influence_tier") or "unknown")
        out["social_platform"] = str(out.get("social_platform") or "none")
        mentions = out.get("symbol_mentions")
        if not isinstance(mentions, list):
            mentions = [mentions] if mentions else []
        normalized_mentions = []
        seen = set()
        for sym in mentions:
            token = str(sym or "").strip().upper().replace("$", "")
            if not token or token in seen:
                continue
            seen.add(token)
            normalized_mentions.append(token)
        out["symbol_mentions"] = normalized_mentions
        return out

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
        if market_scope == "crypto" and not seen_symbols and self._env_flag("ENABLE_BROAD_CRYPTO_FALLBACK_LINKS", default=False):
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

        event_payload = self._normalize_social_payload(dict(event.get("payload", {})))
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
        market_scope = str(event.get("market_scope") or "crypto").strip().lower()
        enriched_entities = event.get("entities", []) or extract_entities(
            event.get("title", ""),
            str(event_payload.get("summary", "")),
        )
        enriched_entities = self._normalize_entities(enriched_entities, market_scope=market_scope)
        if market_scope == "crypto":
            existing = {str((e or {}).get("symbol") or "").strip().upper() for e in enriched_entities if isinstance(e, dict)}
            mentions = event_payload.get("symbol_mentions") if isinstance(event_payload.get("symbol_mentions"), list) else []
            for sym_raw in mentions:
                sym = str(sym_raw or "").strip().upper().replace("$", "")
                if not sym or sym in existing or sym not in self.liquid_symbols:
                    continue
                enriched_entities.append(
                    {
                        "entity_type": "asset",
                        "name": sym,
                        "symbol": sym,
                        "country": None,
                        "sector": "crypto",
                        "metadata": {"source": "social_symbol_mention"},
                    }
                )
                existing.add(sym)

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
                resp = self._http.post(
                    self.ingest_endpoint,
                    json=req_body,
                    timeout=(
                        max(0.1, float(self.ingest_connect_timeout_sec)),
                        max(0.1, float(self.ingest_read_timeout_sec)),
                    ),
                )
                resp.raise_for_status()
                COLLECTOR_EVENT_PUBLISH_TOTAL.labels(connector=connector_name, status="success").inc()
                return
            except Exception as exc:
                if attempt >= self.ingest_max_retries:
                    COLLECTOR_EVENT_PUBLISH_TOTAL.labels(connector=connector_name, status="failed").inc()
                    raise RuntimeError(f"ingest failed after retries: {exc}") from exc
                time.sleep(delay)
                delay = min(5.0, delay * 2.0)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

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
        error_backoff = max(1.0, float(self.cycle_error_backoff_sec))
        while True:
            try:
                sent = self.run_once()
                logger.info("cycle published=%d", sent)
                error_backoff = max(1.0, float(self.cycle_error_backoff_sec))
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("collector-v2 stopped")
                break
            except Exception as exc:
                logger.error("collector-v2 cycle error: %s", exc)
                time.sleep(error_backoff)
                error_backoff = min(
                    max(error_backoff * 2.0, error_backoff + 5.0),
                    max(10.0, float(self.cycle_error_backoff_max_sec)),
                )
        self.close()


if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    interval = int(os.getenv("COLLECT_INTERVAL", "300"))
    DataCollectorV2(redis_url=redis_url).run(interval=interval)
