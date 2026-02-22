from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


VC_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "vc_feature_schema.yaml"


def _load_vc_feature_keys() -> List[str]:
    raw = yaml.safe_load(VC_SCHEMA_PATH.read_text(encoding="utf-8"))
    feats = raw.get("features") if isinstance(raw, dict) else []
    out: List[str] = []
    for item in feats if isinstance(feats, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            out.append(name)
    if not out:
        raise RuntimeError("vc_feature_schema_empty")
    return out


VC_FEATURE_KEYS = _load_vc_feature_keys()
VC_SCHEMA_HASH = hashlib.sha256(VC_SCHEMA_PATH.read_bytes()).hexdigest()


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
    if len(events) == 1:
        return vector_from_event_row(dict(events[0]))
    mats = []
    weights = []
    n = len(events)
    for i, ev in enumerate(events):
        mats.append(vector_from_event_row(dict(ev)))
        # recency weight (earlier list item assumed newer)
        weights.append(float(np.exp(-(i / max(1.0, n / 3.0)))))
    mat = np.stack(mats, axis=0)
    w = np.array(weights, dtype=np.float32).reshape(-1, 1)
    return (mat * w).sum(axis=0) / np.clip(w.sum(axis=0), 1e-8, None)
