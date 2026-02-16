from __future__ import annotations

from typing import Dict, List
import xml.etree.ElementTree as ET

import requests
try:
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None

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


class YouTubeConnector(BaseConnector):
    name = "social_youtube"

    def __init__(
        self,
        mode: str = "rss",
        channel_ids: List[str] | None = None,
        query: str = "crypto",
        api_key: str = "",
        max_results: int = 20,
    ):
        self.mode = (mode or "rss").strip().lower()
        self.channel_ids = [c.strip() for c in (channel_ids or []) if c and c.strip()]
        self.query = str(query or "crypto").strip()
        self.api_key = str(api_key or "").strip()
        self.max_results = max(1, min(50, int(max_results)))

    def _fetch_rss(self) -> List[Dict]:
        rows: List[Dict] = []
        channels = self.channel_ids or []
        if not channels:
            return []
        for channel_id in channels:
            feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            if feedparser is not None:
                feed = feedparser.parse(feed_url)
                for entry in (feed.entries or [])[: self.max_results]:
                    rows.append({"mode": "rss", "entry": entry, "channel_id": channel_id})
                continue
            try:
                resp = requests.get(feed_url, timeout=20)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
            except Exception:
                continue
            ns = {
                "a": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
            }
            for entry in root.findall(".//a:entry", ns)[: self.max_results]:
                link_elem = entry.find("a:link", ns)
                author_elem = entry.find("a:author/a:name", ns)
                rows.append(
                    {
                        "mode": "rss",
                        "entry": {
                            "title": (entry.findtext("a:title", default="", namespaces=ns) or "").strip(),
                            "summary": (entry.findtext("a:summary", default="", namespaces=ns) or "").strip(),
                            "description": (entry.findtext("a:summary", default="", namespaces=ns) or "").strip(),
                            "published": (entry.findtext("a:published", default="", namespaces=ns) or "").strip(),
                            "updated": (entry.findtext("a:updated", default="", namespaces=ns) or "").strip(),
                            "link": str(link_elem.get("href") if link_elem is not None else "").strip(),
                            "author": (author_elem.text if author_elem is not None else "").strip(),
                            "yt_videoid": (entry.findtext("yt:videoId", default="", namespaces=ns) or "").strip(),
                        },
                        "channel_id": channel_id,
                    }
                )
        return rows

    def _fetch_api(self) -> List[Dict]:
        if not self.api_key:
            return []
        rows: List[Dict] = []
        search_url = "https://www.googleapis.com/youtube/v3/search"

        scopes = self.channel_ids if self.channel_ids else [""]
        for channel_id in scopes:
            params = {
                "part": "snippet",
                "type": "video",
                "order": "date",
                "maxResults": self.max_results,
                "q": self.query,
                "key": self.api_key,
            }
            if channel_id:
                params["channelId"] = channel_id
            resp = requests.get(search_url, params=params, timeout=20)
            if resp.status_code == 429:
                raise RateLimitError("youtube_rate_limited")
            resp.raise_for_status()
            data = resp.json() or {}
            items = data.get("items") or []
            video_ids = [str((i.get("id") or {}).get("videoId") or "") for i in items]
            video_ids = [x for x in video_ids if x]
            stats = self._video_stats(video_ids)
            for item in items:
                vid = str((item.get("id") or {}).get("videoId") or "")
                rows.append(
                    {
                        "mode": "api",
                        "item": item,
                        "stats": stats.get(vid) or {},
                        "channel_id": channel_id,
                    }
                )
        return rows

    def _video_stats(self, video_ids: List[str]) -> Dict[str, Dict]:
        if not video_ids or not self.api_key:
            return {}
        stats_url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "statistics",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        resp = requests.get(stats_url, params=params, timeout=20)
        if resp.status_code == 429:
            raise RateLimitError("youtube_stats_rate_limited")
        resp.raise_for_status()
        data = resp.json() or {}
        out: Dict[str, Dict] = {}
        for item in data.get("items") or []:
            out[str(item.get("id") or "")] = item.get("statistics") or {}
        return out

    def fetch(self) -> List[Dict]:
        if self.mode == "api":
            return self._fetch_api()
        return self._fetch_rss()

    def normalize(self, raw: Dict) -> Dict:
        mode = str(raw.get("mode") or "rss")
        if mode == "api":
            item = raw.get("item") or {}
            snippet = item.get("snippet") or {}
            title = str(snippet.get("title") or "").strip()
            desc = str(snippet.get("description") or "").strip()
            text = f"{title}\n{desc}"
            stats = raw.get("stats") or {}
            likes = int(stats.get("likeCount") or 0)
            comments = int(stats.get("commentCount") or 0)
            views = int(stats.get("viewCount") or 0)
            video_id = str((item.get("id") or {}).get("videoId") or "")
            channel_title = str(snippet.get("channelTitle") or "unknown")
            occurred_at = parse_datetime_to_iso_z(snippet.get("publishedAt"))
            source_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
            source_name = f"youtube:{channel_title}"
        else:
            entry = raw.get("entry") or {}
            title = str(entry.get("title") or "").strip()
            desc = str(entry.get("summary") or entry.get("description") or "").strip()
            text = f"{title}\n{desc}"
            likes = 0
            comments = 0
            views = 0
            video_id = str(entry.get("yt_videoid") or "")
            if not video_id:
                links = entry.get("links") or []
                href = str((links[0] or {}).get("href") or "") if links else ""
                if "v=" in href:
                    video_id = href.split("v=")[-1].split("&")[0]
            channel_title = str(entry.get("author") or "unknown")
            occurred_at = parse_datetime_to_iso_z(entry.get("published") or entry.get("updated"))
            source_url = str(entry.get("link") or "")
            source_name = f"youtube:{channel_title}"

        symbols = extract_symbol_mentions(text)
        post_s = sentiment_score(text)
        event_type, market_scope = social_event_type(text)
        payload = social_payload(
            social_platform="youtube",
            author=channel_title,
            author_followers=0,
            engagement_score=estimate_engagement_score(
                likes=likes,
                comments=comments,
                shares=0,
                views=views,
            ),
            comment_sentiment=post_s if comments > 0 else 0.0,
            post_sentiment=post_s,
            n_comments=comments,
            n_replies=comments,
            is_verified=False,
            influence_tier="unknown",
            symbol_mentions=symbols,
            summary=text,
            extra={
                "video_id": video_id,
                "language": detect_language_hint(text),
                "view_count": views,
                "like_count": likes,
                "comment_count": comments,
                "youtube_mode": mode,
            },
        )

        return {
            "event_type": event_type,
            "market_scope": market_scope,
            "title": title[:220] if title else f"YouTube post {video_id}",
            "occurred_at": occurred_at,
            "source_url": source_url,
            "source_name": source_name,
            "source_timezone": "UTC",
            "source_tier": 3,
            "confidence_score": 0.57,
            "event_importance": 0.5,
            "novelty_score": 0.52,
            "entity_confidence": 0.45,
            "payload": payload,
            "entities": symbols_to_entities(symbols),
        }
