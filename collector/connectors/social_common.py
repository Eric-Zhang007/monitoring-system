from __future__ import annotations

import datetime as dt
import re
from typing import Dict, Iterable, List, Sequence, Tuple


_POSITIVE_WORDS = {
    "bull",
    "bullish",
    "breakout",
    "adoption",
    "growth",
    "surge",
    "rally",
    "beat",
    "upgrade",
    "launch",
    "partnership",
    "approval",
    "record",
    "strong",
    "gain",
}
_NEGATIVE_WORDS = {
    "bear",
    "bearish",
    "crash",
    "drop",
    "hack",
    "lawsuit",
    "fraud",
    "ban",
    "downgrade",
    "delay",
    "selloff",
    "outage",
    "weak",
    "decline",
    "loss",
}
_NEGATION_WORDS = {"not", "no", "never", "none", "without", "hardly", "barely", "n't"}
_INTENSIFIERS = {"very", "extremely", "massively", "strongly", "super", "highly", "deeply"}
_DOWNPLAYERS = {"slightly", "somewhat", "kinda", "kindof", "kind-of", "mildly", "barely"}
_POSITIVE_EMOJIS = {"🚀", "📈", "🟢", "💚", "🔥", "✅"}
_NEGATIVE_EMOJIS = {"📉", "🔴", "💥", "❌", "🧨", "⚠️"}
_SARCASM_MARKERS = {"/s", "yeah right", "as if", "sure buddy"}


def utc_now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def parse_datetime_to_iso_z(raw: object) -> str:
    if isinstance(raw, (int, float)):
        return dt.datetime.fromtimestamp(float(raw), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(raw or "").strip()
    if not text:
        return utc_now_iso()
    if text.isdigit():
        return dt.datetime.fromtimestamp(float(text), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        as_float = float(text)
        if as_float > 0 and "." in text:
            return dt.datetime.fromtimestamp(as_float, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    candidate = text.replace(" ", "T")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return utc_now_iso()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def sanitize_symbol_list(symbols: Iterable[object]) -> List[str]:
    seen = set()
    out: List[str] = []
    for sym in symbols:
        token = str(sym or "").strip().upper().replace("$", "")
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def detect_language_hint(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return "other"
    if re.search(r"[\u4e00-\u9fff]", body):
        return "zh"
    if re.search(r"[A-Za-z]", body):
        return "en"
    return "other"


def extract_symbol_mentions(text: str, known_symbols: Sequence[str] | None = None) -> List[str]:
    body = str(text or "")
    candidates = re.findall(r"\$?[A-Z]{2,10}", body)
    cleaned = sanitize_symbol_list(candidates)
    if not known_symbols:
        return cleaned[:12]
    allowed = {str(x).strip().upper() for x in known_symbols if str(x).strip()}
    if not allowed:
        return cleaned[:12]
    filtered = [s for s in cleaned if s in allowed]
    return filtered[:12]


def sentiment_score(text: str) -> float:
    body = str(text or "")
    if not body:
        return 0.0
    lower = body.lower()
    tokens = [t for t in re.findall(r"[A-Za-z']+|[0-9]+", lower) if t]
    if not tokens:
        tokens = lower.split()
    if not tokens and not any(e in body for e in (_POSITIVE_EMOJIS | _NEGATIVE_EMOJIS)):
        return 0.0
    raw_score = 0.0
    weight_sum = 0.0
    for i, tok in enumerate(tokens):
        base = 0.0
        if tok in _POSITIVE_WORDS:
            base = 1.0
        elif tok in _NEGATIVE_WORDS:
            base = -1.0
        if base == 0.0:
            continue
        prev = tokens[max(0, i - 2):i]
        negated = any((p in _NEGATION_WORDS) or p.endswith("n't") for p in prev)
        if negated:
            base *= -1.0
        amp = 1.0
        if any(p in _INTENSIFIERS for p in prev):
            amp *= 1.35
        if any(p in _DOWNPLAYERS for p in prev):
            amp *= 0.7
        raw_score += base * amp
        weight_sum += abs(amp)
    emoji_pos = sum(1 for e in _POSITIVE_EMOJIS if e in body)
    emoji_neg = sum(1 for e in _NEGATIVE_EMOJIS if e in body)
    if emoji_pos or emoji_neg:
        raw_score += float(emoji_pos - emoji_neg) * 0.6
        weight_sum += float(emoji_pos + emoji_neg) * 0.6
    if weight_sum <= 1e-9:
        return 0.0
    score = raw_score / weight_sum
    exclam = body.count("!")
    if exclam > 0:
        score *= min(1.25, 1.0 + 0.03 * exclam)
    if any(marker in lower for marker in _SARCASM_MARKERS):
        score *= 0.5
    return clamp(score, -1.0, 1.0)


def influence_tier_from_followers(followers: int) -> str:
    f = max(0, int(followers))
    if f >= 1_000_000:
        return "mega"
    if f >= 100_000:
        return "macro"
    if f >= 10_000:
        return "mid"
    if f >= 1_000:
        return "micro"
    return "nano"


def estimate_engagement_score(
    likes: object = 0,
    comments: object = 0,
    shares: object = 0,
    views: object = 0,
    quotes: object = 0,
) -> float:
    like_n = safe_int(likes)
    comment_n = safe_int(comments)
    share_n = safe_int(shares)
    view_n = safe_int(views)
    quote_n = safe_int(quotes)
    score = (
        like_n * 1.0
        + comment_n * 2.0
        + share_n * 1.8
        + quote_n * 1.6
        + (view_n * 0.01)
    )
    return round(max(0.0, score), 3)


def social_payload(
    *,
    social_platform: str,
    author: str,
    author_followers: object,
    engagement_score: object,
    comment_sentiment: object,
    post_sentiment: object,
    n_comments: object,
    n_replies: object,
    is_verified: object,
    influence_tier: str | None,
    symbol_mentions: Iterable[object],
    summary: str,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    followers = max(0, safe_int(author_followers, default=0))
    tier = str(influence_tier or "").strip().lower() or influence_tier_from_followers(followers)
    payload: Dict[str, object] = {
        "summary": str(summary or "")[:1200],
        "social_platform": str(social_platform or "unknown").strip().lower() or "unknown",
        "author": str(author or "unknown")[:120],
        "author_followers": followers,
        "engagement_score": max(0.0, safe_float(engagement_score, default=0.0)),
        "comment_sentiment": clamp(safe_float(comment_sentiment, default=0.0), -1.0, 1.0),
        "post_sentiment": clamp(safe_float(post_sentiment, default=0.0), -1.0, 1.0),
        "n_comments": max(0, safe_int(n_comments, default=0)),
        "n_replies": max(0, safe_int(n_replies, default=0)),
        "is_verified": bool(is_verified),
        "influence_tier": tier,
        "symbol_mentions": sanitize_symbol_list(symbol_mentions),
    }
    if extra:
        payload.update(extra)
    return payload


def symbols_to_entities(symbols: Sequence[str]) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    for sym in sanitize_symbol_list(symbols):
        entities.append(
            {
                "entity_type": "asset",
                "name": sym,
                "symbol": sym,
                "country": None,
                "sector": "crypto",
                "metadata": {"source": "social_symbol_mention"},
            }
        )
    return entities


def social_event_type(text: str) -> Tuple[str, str]:
    body = str(text or "").lower()
    if any(k in body for k in ("sec", "lawsuit", "regulator", "etf approval", "ban")):
        return "regulatory", "macro"
    return "market", "crypto"
