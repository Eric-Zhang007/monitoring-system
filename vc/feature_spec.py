from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


VC_FEATURE_KEYS = [
    "event_type_score",
    "source_tier_norm",
    "confidence_score",
    "event_importance",
    "novelty_score",
]


def event_type_score(event_type: str) -> float:
    et = str(event_type or "").strip().lower()
    if et == "funding":
        return 1.0
    if et == "mna":
        return 0.7
    if et == "product":
        return 0.4
    if et == "regulatory":
        return -0.3
    return 0.0


def vector_from_event_row(row: Dict[str, Any]) -> np.ndarray:
    et = str(row.get("event_type") or "")
    tier = float(row.get("source_tier") or 5.0)
    conf = float(row.get("confidence_score") or 0.0)
    imp = float(row.get("event_importance") or 0.0)
    nov = float(row.get("novelty_score") or 0.0)
    return np.array([
        event_type_score(et),
        (6.0 - tier) / 5.0,
        conf,
        imp,
        nov,
    ], dtype=np.float32)


def label_from_event_row(row: Dict[str, Any]) -> int:
    et = str(row.get("event_type") or "").strip().lower()
    return 1 if et in {"funding", "mna"} else 0


def vector_from_context(events: List[Dict[str, Any]]) -> np.ndarray:
    if not events:
        return np.zeros((len(VC_FEATURE_KEYS),), dtype=np.float32)
    return vector_from_event_row(dict(events[0]))
