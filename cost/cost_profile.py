from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


HORIZON_ORDER = ("1h", "4h", "1d", "7d")


@dataclass(frozen=True)
class CostProfile:
    name: str
    fee_bps: float
    slippage_bps: float
    impact_base_bps: float
    horizon_multipliers: Dict[str, float]
    liquidity_lambda: float
    volatility_lambda: float
    turnover_lambda: float
    infra_bps_per_hour: float


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _parse_horizon_map(raw: str, default: Mapping[str, float]) -> Dict[str, float]:
    out = {str(k): float(v) for k, v in default.items()}
    text = str(raw or "").strip()
    if not text:
        return out
    for token in text.split(","):
        part = token.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        kk = str(k).strip().lower()
        if kk not in out:
            continue
        try:
            out[kk] = float(v.strip())
        except Exception:
            continue
    return out


def load_cost_profile(name: str = "standard") -> CostProfile:
    key = str(name or "standard").strip().lower() or "standard"
    prefix = f"COST_PROFILE_{key.upper()}"
    multipliers = _parse_horizon_map(
        os.getenv(f"{prefix}_HORIZON_MULTIPLIERS", ""),
        {"1h": 1.0, "4h": 1.6, "1d": 2.4, "7d": 3.2},
    )
    return CostProfile(
        name=key,
        fee_bps=_env_float(f"{prefix}_FEE_BPS", 5.0),
        slippage_bps=_env_float(f"{prefix}_SLIPPAGE_BPS", 3.0),
        impact_base_bps=_env_float(f"{prefix}_IMPACT_BASE_BPS", 2.0),
        horizon_multipliers=multipliers,
        liquidity_lambda=_env_float(f"{prefix}_LIQUIDITY_LAMBDA", 0.6),
        volatility_lambda=_env_float(f"{prefix}_VOLATILITY_LAMBDA", 0.8),
        turnover_lambda=_env_float(f"{prefix}_TURNOVER_LAMBDA", 0.3),
        infra_bps_per_hour=_env_float(f"{prefix}_INFRA_BPS_PER_HOUR", 0.0),
    )


def compute_cost_bps(
    *,
    horizon: str,
    profile: CostProfile | None = None,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    turnover_estimate: float | None = None,
    infra_usd: float | None = None,
) -> float:
    p = profile or load_cost_profile("standard")
    h = str(horizon or "1h").strip().lower()
    base = float(p.fee_bps + p.slippage_bps + p.impact_base_bps)
    mult = float(p.horizon_multipliers.get(h, p.horizon_multipliers.get("1h", 1.0)))

    liq = liquidity_features or {}
    mkt = market_state or {}

    depth = float(liq.get("orderbook_depth_total", liq.get("depth_total", 0.0)) or 0.0)
    liquidity_score = float(liq.get("liquidity_score", 0.0) or 0.0)
    if liquidity_score <= 0.0 and depth > 0.0:
        liquidity_score = max(0.0, min(1.0, depth / (depth + 1_000_000.0)))
    liquidity_adj = float(1.0 + p.liquidity_lambda * (1.0 - max(0.0, min(1.0, liquidity_score))))

    realized_vol = float(mkt.get("realized_vol", mkt.get("volatility", 0.0)) or 0.0)
    vol_adj = float(1.0 + p.volatility_lambda * max(0.0, realized_vol))

    turn = float(max(0.0, turnover_estimate or 0.0))
    turn_adj = float(1.0 + p.turnover_lambda * turn)

    infra_bps = 0.0
    if infra_usd is not None:
        notional = float(mkt.get("notional_usd", 0.0) or 0.0)
        if notional > 1e-9:
            infra_bps = float(max(0.0, infra_usd) / notional * 1e4)
    else:
        horizon_hours = {"1h": 1.0, "4h": 4.0, "1d": 24.0, "7d": 168.0}
        infra_bps = float(p.infra_bps_per_hour * horizon_hours.get(h, 1.0))

    return float(max(0.0, base * mult * liquidity_adj * vol_adj * turn_adj + infra_bps))


def compute_cost_map(
    *,
    horizons: Sequence[str] = HORIZON_ORDER,
    profile: CostProfile | None = None,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    turnover_estimate: float | None = None,
    infra_usd: float | None = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in horizons:
        hh = str(h).strip().lower()
        out[hh] = compute_cost_bps(
            horizon=hh,
            profile=profile,
            market_state=market_state,
            liquidity_features=liquidity_features,
            turnover_estimate=turnover_estimate,
            infra_usd=infra_usd,
        )
    return out
