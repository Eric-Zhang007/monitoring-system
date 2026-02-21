from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class AllocatorSignal:
    target: str
    track: str
    horizon: str
    action: str
    score: float
    confidence: float
    strategy_bucket: str


def _parse_bucket_limits(raw: str, default_risk_budget: float) -> Dict[str, float]:
    defaults = {
        "trend": default_risk_budget * 0.55,
        "event": default_risk_budget * 0.70,
        "mean_reversion": default_risk_budget * 0.45,
    }
    out = dict(defaults)
    text = str(raw or "").strip()
    if not text:
        return out
    for token in text.split(","):
        part = token.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = str(k).strip().lower()
        try:
            val = float(v.strip())
        except Exception:
            continue
        if key and val >= 0.0:
            out[key] = float(val)
    return out


def _signal_side(action: str) -> float:
    a = str(action or "").strip().lower()
    if a == "buy":
        return 1.0
    if a == "sell":
        return -1.0
    return 0.0


def _returns_from_prices(rows: List[Dict[str, Any]]) -> np.ndarray:
    px = [float(r.get("price") or 0.0) for r in rows if float(r.get("price") or 0.0) > 0.0]
    if len(px) < 8:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(px, dtype=np.float64)
    ret = np.diff(arr) / np.clip(arr[:-1], 1e-12, None)
    return ret.astype(np.float64)


def _corr_penalties(
    targets: Iterable[str],
    load_price_history: Callable[[str, int], List[Dict[str, Any]]],
    *,
    lookback_days: int,
    threshold: float,
    penalty: float,
) -> Dict[str, float]:
    names = [str(t).upper() for t in targets if str(t).strip()]
    if len(names) <= 1:
        return {t: 1.0 for t in names}
    series: Dict[str, np.ndarray] = {}
    for t in names:
        try:
            series[t] = _returns_from_prices(load_price_history(t, lookback_days))
        except Exception:
            series[t] = np.zeros((0,), dtype=np.float64)
    out = {t: 1.0 for t in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa = series.get(a)
            sb = series.get(b)
            if sa is None or sb is None:
                continue
            n = int(min(sa.size, sb.size))
            if n < 16:
                continue
            xa = sa[-n:]
            xb = sb[-n:]
            if float(np.std(xa)) <= 1e-12 or float(np.std(xb)) <= 1e-12:
                continue
            corr = float(np.corrcoef(xa, xb)[0, 1])
            if not np.isfinite(corr):
                continue
            if abs(corr) >= float(threshold):
                out[a] = float(out[a] * max(0.1, 1.0 - penalty))
                out[b] = float(out[b] * max(0.1, 1.0 - penalty))
    return out


def allocate_targets(
    *,
    signals: List[AllocatorSignal],
    risk_budget: float,
    load_price_history: Callable[[str, int], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    rb = max(0.0, float(risk_budget))
    if rb <= 0.0 or not signals:
        return {"weights": {}, "contributions": {}, "bucket_exposure": {}, "raw_strength": {}}
    single_symbol_cap = float(os.getenv("ALLOCATOR_SINGLE_SYMBOL_MAX", "0.20") or 0.20)
    corr_days = int(os.getenv("ALLOCATOR_CORR_LOOKBACK_DAYS", "30") or 30)
    corr_th = float(os.getenv("ALLOCATOR_CORR_THRESHOLD", "0.85") or 0.85)
    corr_pen = float(os.getenv("ALLOCATOR_CORR_PENALTY", "0.5") or 0.5)
    bucket_limits = _parse_bucket_limits(os.getenv("ALLOCATOR_BUCKET_LIMITS", ""), default_risk_budget=rb)
    dedupe = _corr_penalties(
        [s.target for s in signals],
        load_price_history,
        lookback_days=max(7, corr_days),
        threshold=max(0.0, min(1.0, corr_th)),
        penalty=max(0.0, min(0.9, corr_pen)),
    )

    raw_strength_by_symbol: Dict[str, float] = {}
    contrib: Dict[str, Dict[str, float]] = {}
    bucket_by_contrib: Dict[Tuple[str, str], str] = {}
    for sig in signals:
        side = _signal_side(sig.action)
        if abs(side) <= 1e-12:
            continue
        target = str(sig.target).upper()
        hh = str(sig.horizon).strip().lower() or "1h"
        bucket = str(sig.strategy_bucket or "event").strip().lower() or "event"
        key = (bucket, hh)
        strength = side * max(0.0, abs(float(sig.score))) * max(0.0, min(1.0, float(sig.confidence)))
        strength *= float(dedupe.get(target, 1.0))
        raw_strength_by_symbol[target] = raw_strength_by_symbol.get(target, 0.0) + strength
        slot = contrib.setdefault(target, {})
        slot[hh] = slot.get(hh, 0.0) + strength
        bucket_by_contrib[(target, hh)] = f"{bucket}:{hh}"

    gross = float(sum(abs(v) for v in raw_strength_by_symbol.values()))
    if gross <= 1e-12:
        return {"weights": {}, "contributions": contrib, "bucket_exposure": {}, "raw_strength": raw_strength_by_symbol}

    weights = {k: float((v / gross) * rb) for k, v in raw_strength_by_symbol.items()}
    for k, v in list(weights.items()):
        if abs(v) > single_symbol_cap:
            weights[k] = single_symbol_cap if v > 0 else -single_symbol_cap

    # Apply bucket caps on (strategy_bucket, horizon) and strategy-bucket aggregate.
    bucket_exposure: Dict[str, float] = {}
    for target, hmap in contrib.items():
        for h, val in hmap.items():
            bucket_key = bucket_by_contrib.get((target, h), f"event:{h}")
            # Exposure contribution uses allocated target weight split by raw contribution ratio.
            den = sum(abs(x) for x in hmap.values()) or 1.0
            share = abs(val) / den
            exp = abs(weights.get(target, 0.0)) * share
            bucket_exposure[bucket_key] = bucket_exposure.get(bucket_key, 0.0) + exp
            b0 = bucket_key.split(":", 1)[0]
            bucket_exposure[b0] = bucket_exposure.get(b0, 0.0) + exp

    for bucket_key, exposure in list(bucket_exposure.items()):
        limit = float(bucket_limits.get(bucket_key, bucket_limits.get(bucket_key.split(":", 1)[0], rb)))
        if exposure <= limit or exposure <= 1e-12:
            continue
        scale = max(0.0, min(1.0, limit / exposure))
        for target, hmap in contrib.items():
            hit = False
            for h in hmap.keys():
                bkey = bucket_by_contrib.get((target, h), f"event:{h}")
                if bkey == bucket_key or bkey.split(":", 1)[0] == bucket_key:
                    hit = True
                    break
            if hit:
                weights[target] = float(weights.get(target, 0.0) * scale)

    gross_after = float(sum(abs(v) for v in weights.values()))
    if gross_after > rb and gross_after > 1e-12:
        sc = rb / gross_after
        for k in list(weights.keys()):
            weights[k] = float(weights[k] * sc)

    return {
        "weights": weights,
        "contributions": contrib,
        "bucket_exposure": bucket_exposure,
        "raw_strength": raw_strength_by_symbol,
        "dedupe_factor": dedupe,
        "single_symbol_cap": single_symbol_cap,
        "risk_budget": rb,
    }
