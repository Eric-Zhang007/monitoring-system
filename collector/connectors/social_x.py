from __future__ import annotations

from typing import Dict, List

import requests

from connectors.base import BaseConnector, RateLimitError
from connectors.social_common import (
    detect_language_hint,
    estimate_engagement_score,
    extract_symbol_mentions,
    parse_datetime_to_iso_z,
    sentiment_score,
    social_event_type,
    social_payload,
    symbols_to_entities,
)


class XTwitterConnector(BaseConnector):
    name = "social_x"

    def __init__(
        self,
        bearer_token: str,
        query: str,
        max_results: int = 25,
        known_symbols: List[str] | None = None,
    ):
        self.bearer_token = str(bearer_token or "").strip()
        self.query = str(query or "").strip()
        self.max_results = max(10, min(100, int(max_results)))
        self.known_symbols = known_symbols or []

    def fetch(self) -> List[Dict]:
        if not self.bearer_token or not self.query:
            return []
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": self.query,
            "max_results": self.max_results,
            "tweet.fields": "created_at,lang,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username,verified,public_metrics",
        }
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code == 429:
            raise RateLimitError("x_rate_limited")
        resp.raise_for_status()
        data = resp.json() or {}
        users = {str(u.get("id")): u for u in (data.get("includes") or {}).get("users") or []}
        out: List[Dict] = []
        for tweet in data.get("data") or []:
            user = users.get(str(tweet.get("author_id"))) or {}
            out.append({"tweet": tweet, "user": user})
        return out

    def normalize(self, raw: Dict) -> Dict:
        tweet = raw.get("tweet") or {}
        user = raw.get("user") or {}
        text = str(tweet.get("text") or "").strip()
        tweet_id = str(tweet.get("id") or "").strip()
        username = str(user.get("username") or "unknown")

        metrics = tweet.get("public_metrics") or {}
        likes = int(metrics.get("like_count") or 0)
        replies = int(metrics.get("reply_count") or 0)
        retweets = int(metrics.get("retweet_count") or 0)
        quotes = int(metrics.get("quote_count") or 0)
        followers = int(((user.get("public_metrics") or {}).get("followers_count") or 0))

        symbols = extract_symbol_mentions(text, known_symbols=self.known_symbols)
        post_s = sentiment_score(text)
        event_type, market_scope = social_event_type(text)
        payload = social_payload(
            social_platform="x",
            author=f"@{username}",
            author_followers=followers,
            engagement_score=estimate_engagement_score(
                likes=likes,
                comments=replies,
                shares=retweets,
                views=0,
                quotes=quotes,
            ),
            comment_sentiment=post_s if replies > 0 else 0.0,
            post_sentiment=post_s,
            n_comments=replies,
            n_replies=replies,
            is_verified=bool(user.get("verified")),
            influence_tier=None,
            symbol_mentions=symbols,
            summary=text,
            extra={
                "tweet_id": tweet_id,
                "language": str(tweet.get("lang") or "").strip().lower() or detect_language_hint(text),
                "retweet_count": retweets,
                "like_count": likes,
                "reply_count": replies,
                "quote_count": quotes,
            },
        )

        title = text[:220] if text else f"X post by @{username}"
        return {
            "event_type": event_type,
            "market_scope": market_scope,
            "title": title,
            "occurred_at": parse_datetime_to_iso_z(tweet.get("created_at")),
            "source_url": f"https://x.com/{username}/status/{tweet_id}" if tweet_id else "",
            "source_name": "x_api_v2",
            "source_timezone": "UTC",
            "source_tier": 2,
            "confidence_score": 0.63,
            "event_importance": 0.6,
            "novelty_score": 0.56,
            "entity_confidence": 0.52,
            "payload": payload,
            "entities": symbols_to_entities(symbols),
        }
