from __future__ import annotations

from typing import Dict, List


def build_explanation(
    model_version: str,
    feature_version: str,
    event_context: List[Dict],
    feature_contribs: List[Dict],
) -> Dict:
    return {
        "top_event_contributors": [
            {
                "event_id": e.get("id"),
                "event_type": e.get("event_type"),
                "title": e.get("title"),
                "weight": round(0.1 + float(e.get("confidence_score", 0.5)) * 0.8, 3),
            }
            for e in event_context[:5]
        ],
        "top_feature_contributors": feature_contribs[:5],
        "evidence_links": [e.get("source_url") for e in event_context if e.get("source_url")][:5],
        "model_version": model_version,
        "feature_version": feature_version,
    }
