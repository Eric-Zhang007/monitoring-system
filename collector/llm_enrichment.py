from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _clamp(v: object, low: float, high: float, default: float) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        out = float(default)
    return max(low, min(high, out))


def _clean_symbol(raw: object) -> str:
    text = str(raw or "").strip().upper().replace("$", "")
    if not text:
        return ""
    if not re.fullmatch(r"[A-Z0-9]{2,12}", text):
        return ""
    return text


def _detect_language(text: str) -> str:
    body = str(text or "")
    if not body.strip():
        return "other"
    if _CJK_RE.search(body):
        return "zh"
    if re.search(r"[A-Za-z]", body):
        return "en"
    return "other"


def _heuristic_sentiment(text: str) -> float:
    body = str(text or "").lower()
    if not body:
        return 0.0
    pos = (
        "bull",
        "bullish",
        "rally",
        "surge",
        "growth",
        "beat",
        "approval",
        "上涨",
        "利好",
        "看涨",
    )
    neg = (
        "bear",
        "bearish",
        "crash",
        "drop",
        "hack",
        "lawsuit",
        "ban",
        "下跌",
        "暴跌",
        "利空",
    )
    pos_n = sum(1 for k in pos if k in body)
    neg_n = sum(1 for k in neg if k in body)
    if pos_n == 0 and neg_n == 0:
        return 0.0
    return max(-1.0, min(1.0, float(pos_n - neg_n) / float(max(1, pos_n + neg_n))))


class LLMEnricher:
    def __init__(self) -> None:
        self.enabled = _env_flag("LLM_ENRICHMENT_ENABLED", default=False)
        self.api_key = str(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        self.api_base_url = str(os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")).strip().rstrip("/")
        self.chat_path = str(os.getenv("LLM_CHAT_COMPLETIONS_PATH", "/chat/completions")).strip() or "/chat/completions"
        self.model = str(os.getenv("LLM_MODEL", "gpt-4o-mini")).strip()
        self.timeout_sec = max(1.0, float(os.getenv("LLM_TIMEOUT_SEC", "20")))
        self.max_retries = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
        self.backoff_sec = max(0.1, float(os.getenv("LLM_BACKOFF_SEC", "1.0")))
        self.max_input_chars = max(512, int(os.getenv("LLM_MAX_INPUT_CHARS", "4000")))
        self.max_summary_chars = max(80, int(os.getenv("LLM_MAX_SUMMARY_CHARS", "320")))
        self.provider = str(os.getenv("LLM_PROVIDER", "openai_compatible")).strip()
        self.temperature = _clamp(os.getenv("LLM_TEMPERATURE", "0.0"), 0.0, 1.5, 0.0)
        self.max_tokens = max(64, int(os.getenv("LLM_MAX_TOKENS", "350")))
        self._http = requests.Session()

    def enrich_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(event.get("payload") or {})
        title = str(event.get("title") or "").strip()
        summary = str(payload.get("summary") or payload.get("description") or payload.get("text") or "").strip()
        source_name = str(event.get("source_name") or "")
        source_url = str(event.get("source_url") or "")
        market_scope = str(event.get("market_scope") or "crypto")
        text = "\n".join([x for x in [title, summary] if x]).strip()
        if len(text) > self.max_input_chars:
            text = text[: self.max_input_chars]

        if not self.enabled:
            return self._fallback(text=text, title=title, status="disabled", error="")
        if not self.api_key:
            return self._fallback(text=text, title=title, status="no_api_key", error="llm_api_key_missing")

        request_body = self._build_request(
            title=title,
            summary=summary,
            source_name=source_name,
            source_url=source_url,
            market_scope=market_scope,
            occurred_at=str(event.get("occurred_at") or ""),
        )
        try:
            raw = self._call_api_with_retry(request_body)
            parsed = self._parse_response(raw)
            if not parsed:
                return self._fallback(text=text, title=title, status="parse_error", error="invalid_llm_payload")
            return self._normalize_output(parsed=parsed, text=text, title=title)
        except Exception as exc:
            logger.warning("llm enrichment unavailable: %s", exc)
            return self._fallback(text=text, title=title, status="unavailable", error=str(exc))

    def _build_request(
        self,
        *,
        title: str,
        summary: str,
        source_name: str,
        source_url: str,
        market_scope: str,
        occurred_at: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You extract structured market-event signals. "
            "Return strict JSON object with keys: language, sentiment, summary, confidence, entities. "
            "language must be one of zh/en/other. sentiment in [-1,1]. "
            f"summary max {self.max_summary_chars} chars. confidence in [0,1]. "
            "entities is an array of objects with keys: name, entity_type, symbol, confidence. "
            "entity_type must be asset/company/investor. "
            "If unknown, return empty entities."
        )
        user_payload = {
            "title": title,
            "summary": summary,
            "source_name": source_name,
            "source_url": source_url,
            "market_scope": market_scope,
            "occurred_at": occurred_at,
        }
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

    def _call_api_with_retry(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base_url}{self.chat_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        delay = self.backoff_sec
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._http.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=self.timeout_sec,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"llm_retryable_http_{resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError("llm_non_object_response")
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(delay * (1.0 + random.uniform(0.0, 0.25)))
                delay = min(30.0, delay * 2.0)
        raise RuntimeError(f"llm_request_failed: {last_error}")

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            return {}
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            merged = []
            for part in content:
                if isinstance(part, dict):
                    merged.append(str(part.get("text") or ""))
                else:
                    merged.append(str(part))
            content_text = "".join(merged).strip()
        else:
            content_text = str(content or "").strip()
        if not content_text:
            return {}
        try:
            direct = json.loads(content_text)
            if isinstance(direct, dict):
                return direct
        except Exception:
            pass
        m = _JSON_BLOCK_RE.search(content_text)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _normalize_output(self, parsed: Dict[str, Any], text: str, title: str) -> Dict[str, Any]:
        language = str(parsed.get("language") or "").strip().lower()
        if language not in {"zh", "en", "other"}:
            language = _detect_language(text)

        sentiment = _clamp(parsed.get("sentiment"), -1.0, 1.0, _heuristic_sentiment(text))
        summary = str(parsed.get("summary") or "").strip()
        if not summary:
            summary = (title or text)[: self.max_summary_chars]
        summary = summary[: self.max_summary_chars]
        confidence = _clamp(parsed.get("confidence"), 0.0, 1.0, 0.5)
        entities = self._normalize_entities(parsed.get("entities"))

        return {
            "status": "ok",
            "provider": self.provider,
            "model": self.model,
            "language": language,
            "sentiment": sentiment,
            "summary": summary,
            "confidence": confidence,
            "entities": entities,
        }

    def _normalize_entities(self, raw_entities: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_entities, list):
            return []
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in raw_entities[:20]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            raw_type = str(item.get("entity_type") or "").strip().lower()
            if raw_type in {"asset", "token", "coin", "crypto"}:
                entity_type = "asset"
            elif raw_type in {"investor", "fund", "vc", "institution"}:
                entity_type = "investor"
            else:
                entity_type = "company"
            symbol = _clean_symbol(item.get("symbol"))
            if entity_type == "asset" and not symbol:
                symbol = _clean_symbol(name)
            confidence = _clamp(item.get("confidence"), 0.0, 1.0, 0.5)
            key = (entity_type, symbol or name.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "name": name[:120],
                    "entity_type": entity_type,
                    "symbol": symbol,
                    "confidence": confidence,
                }
            )
        return out

    def _fallback(self, *, text: str, title: str, status: str, error: str) -> Dict[str, Any]:
        summary = (title or text or "event")[: self.max_summary_chars]
        out = {
            "status": status,
            "provider": self.provider,
            "model": self.model,
            "language": _detect_language(text),
            "sentiment": _heuristic_sentiment(text),
            "summary": summary,
            "confidence": 0.2,
            "entities": [],
        }
        if error:
            out["error"] = str(error)[:240]
        return out

