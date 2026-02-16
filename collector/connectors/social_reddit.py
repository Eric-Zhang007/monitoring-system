from __future__ import annotations

import base64
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


class RedditConnector(BaseConnector):
    name = "social_reddit"

    def __init__(
        self,
        subreddits: List[str],
        limit: int = 25,
        sort: str = "new",
        user_agent: str = "monitoring-system/2.0",
        mode: str = "public",
        client_id: str = "",
        client_secret: str = "",
        fetch_comments: bool = True,
        comments_per_post: int = 20,
        max_posts_for_comments: int = 80,
    ):
        self.subreddits = [s.strip() for s in subreddits if s and s.strip()]
        self.limit = max(1, min(100, int(limit)))
        self.sort = (sort or "new").strip()
        self.user_agent = user_agent
        self.mode = (mode or "public").strip().lower()
        self.client_id = str(client_id or "").strip()
        self.client_secret = str(client_secret or "").strip()
        self.fetch_comments = bool(fetch_comments)
        self.comments_per_post = max(1, min(100, int(comments_per_post)))
        self.max_posts_for_comments = max(1, int(max_posts_for_comments))

    def _oauth_token(self) -> str:
        if not self.client_id or not self.client_secret:
            return ""
        token_url = "https://www.reddit.com/api/v1/access_token"
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        headers = {
            "Authorization": "Basic " + base64.b64encode(raw).decode("utf-8"),
            "User-Agent": self.user_agent,
        }
        data = {"grant_type": "client_credentials"}
        resp = requests.post(token_url, headers=headers, data=data, timeout=15)
        if resp.status_code == 429:
            raise RateLimitError("reddit_auth_rate_limited")
        resp.raise_for_status()
        return str((resp.json() or {}).get("access_token") or "")

    def fetch(self) -> List[Dict]:
        if not self.subreddits:
            return []
        out: List[Dict] = []

        token = ""
        if self.mode == "api":
            token = self._oauth_token()
        comment_fetch_count = 0
        for subreddit in self.subreddits:
            if token:
                base = f"https://oauth.reddit.com/r/{subreddit}/{self.sort}"
                headers = {
                    "Authorization": f"bearer {token}",
                    "User-Agent": self.user_agent,
                }
            else:
                base = f"https://www.reddit.com/r/{subreddit}/{self.sort}.json"
                headers = {"User-Agent": self.user_agent}
            params = {"limit": self.limit}
            resp = requests.get(base, params=params, headers=headers, timeout=20)
            if resp.status_code == 429:
                raise RateLimitError("reddit_rate_limited")
            if resp.status_code != 200:
                continue
            data = (resp.json() or {}).get("data") or {}
            children = data.get("children") or []
            for child in children:
                post = (child or {}).get("data") or {}
                item = {"subreddit": subreddit, "post": post}
                if self.fetch_comments and comment_fetch_count < self.max_posts_for_comments:
                    stats = self._fetch_comment_stats(
                        permalink=str(post.get("permalink") or ""),
                        token=token,
                        headers=headers,
                    )
                    item.update(stats)
                    comment_fetch_count += 1
                out.append(item)
        return out

    def _fetch_comment_stats(self, permalink: str, token: str, headers: Dict[str, str]) -> Dict[str, object]:
        link = str(permalink or "").strip()
        if not link:
            return {"comment_sentiment": 0.0, "n_replies": 0, "comment_symbol_mentions": []}
        if token:
            url = f"https://oauth.reddit.com{link}.json"
            req_headers = dict(headers)
        else:
            url = f"https://www.reddit.com{link}.json"
            req_headers = {"User-Agent": self.user_agent}
        params = {"limit": self.comments_per_post, "depth": 1, "sort": "top"}
        try:
            resp = requests.get(url, params=params, headers=req_headers, timeout=20)
            if resp.status_code == 429:
                raise RateLimitError("reddit_comment_rate_limited")
            if resp.status_code != 200:
                return {"comment_sentiment": 0.0, "n_replies": 0, "comment_symbol_mentions": []}
            raw = resp.json() or []
        except Exception:
            return {"comment_sentiment": 0.0, "n_replies": 0, "comment_symbol_mentions": []}
        if not isinstance(raw, list) or len(raw) < 2:
            return {"comment_sentiment": 0.0, "n_replies": 0, "comment_symbol_mentions": []}
        comments_node = (((raw[1] or {}).get("data") or {}).get("children") or [])
        sentiments: List[float] = []
        mentions: List[str] = []
        replies_cnt = 0
        for node in comments_node[: self.comments_per_post]:
            data = (node or {}).get("data") or {}
            body = str(data.get("body") or "").strip()
            if not body:
                continue
            sentiments.append(float(sentiment_score(body)))
            mentions.extend(extract_symbol_mentions(body))
            try:
                replies_cnt += int(data.get("replies") and ((data.get("replies") or {}).get("data") or {}).get("dist", 0) or 0)
            except Exception:
                replies_cnt += 0
        avg_sent = float(sum(sentiments) / max(1, len(sentiments))) if sentiments else 0.0
        return {
            "comment_sentiment": avg_sent,
            "n_replies": int(replies_cnt),
            "comment_symbol_mentions": mentions,
        }

    def normalize(self, raw: Dict) -> Dict:
        post = raw.get("post") or {}
        subreddit = str(raw.get("subreddit") or post.get("subreddit") or "")
        title = str(post.get("title") or "").strip()
        body = str(post.get("selftext") or "").strip()
        combined = f"{title}\n{body}".strip()

        score = int(post.get("score") or 0)
        comments = int(post.get("num_comments") or 0)
        upvote_ratio = float(post.get("upvote_ratio") or 0.0)

        symbols = extract_symbol_mentions(combined)
        comment_symbols = extract_symbol_mentions(" ".join(str(x) for x in (raw.get("comment_symbol_mentions") or [])))
        if comment_symbols:
            symbols = list(dict.fromkeys(list(symbols) + list(comment_symbols)))
        post_s = sentiment_score(combined)
        comment_s = float(raw.get("comment_sentiment") or 0.0) if comments > 0 else 0.0
        event_type, market_scope = social_event_type(combined)

        permalink = str(post.get("permalink") or "").strip()
        source_url = f"https://www.reddit.com{permalink}" if permalink else ""
        payload = social_payload(
            social_platform="reddit",
            author=str(post.get("author") or "unknown"),
            author_followers=0,
            engagement_score=estimate_engagement_score(
                likes=max(0, score),
                comments=comments,
                shares=0,
                views=0,
            ),
            comment_sentiment=comment_s,
            post_sentiment=post_s,
            n_comments=comments,
            n_replies=int(raw.get("n_replies") or comments),
            is_verified=False,
            influence_tier="unknown",
            symbol_mentions=symbols,
            summary=combined,
            extra={
                "subreddit": subreddit,
                "language": detect_language_hint(combined),
                "upvote_ratio": upvote_ratio,
                "post_score": score,
                "post_id": str(post.get("id") or ""),
            },
        )

        return {
            "event_type": event_type,
            "market_scope": market_scope,
            "title": title[:220] if title else f"Reddit post r/{subreddit}",
            "occurred_at": parse_datetime_to_iso_z(post.get("created_utc")),
            "source_url": source_url,
            "source_name": f"reddit:r/{subreddit}" if subreddit else "reddit",
            "source_timezone": "UTC",
            "source_tier": 3,
            "confidence_score": 0.58,
            "event_importance": 0.52,
            "novelty_score": 0.55,
            "entity_confidence": 0.48,
            "payload": payload,
            "entities": symbols_to_entities(symbols),
        }
