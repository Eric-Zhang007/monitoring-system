from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    funding_lambda: float
    infra_cny_per_hour: float
    usd_cny: float
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
        funding_lambda=_env_float(f"{prefix}_FUNDING_LAMBDA", 1.0),
        infra_cny_per_hour=_env_float(f"{prefix}_INFRA_CNY_PER_HOUR", 1.88),
        usd_cny=_env_float(f"{prefix}_USD_CNY", 7.2),
        infra_bps_per_hour=_env_float(f"{prefix}_INFRA_BPS_PER_HOUR", 0.0),
    )


def _horizon_hours(h: str) -> float:
    key = str(h or "1h").strip().lower()
    if key.endswith("h"):
        return max(1.0, float(key[:-1] or 1.0))
    if key.endswith("d"):
        return max(1.0, float(key[:-1] or 1.0) * 24.0)
    if key.endswith("m"):
        return max(1.0 / 60.0, float(key[:-1] or 5.0) / 60.0)
    return 1.0


def estimate_turnover(
    *,
    horizon: str,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    account_state: Mapping[str, Any] | None = None,
    base_turnover: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    mkt = dict(market_state or {})
    liq = dict(liquidity_features or {})
    acct = dict(account_state or {})
    cfg = dict(config or {})

    if base_turnover is None:
        raw_base = acct.get("turnover_estimate", cfg.get("base_turnover", os.getenv("COST_TURNOVER_BASE", "")))
        if raw_base is None or str(raw_base).strip() == "":
            base_turnover = 0.35
        else:
            try:
                base_turnover = float(raw_base)
            except Exception:
                base_turnover = 0.35
    base = float(max(0.0, base_turnover))

    hours = _horizon_hours(str(horizon or "1h"))
    horizon_scale = 1.0 / max(1.0, float(hours) ** 0.5)

    realized_vol = float(abs(mkt.get("realized_vol", mkt.get("volatility", 0.0)) or 0.0))
    vol_ref = max(1e-6, _env_float("COST_TURNOVER_VOL_REF", 0.02))
    vol_beta = max(0.0, _env_float("COST_TURNOVER_VOL_BETA", 0.8))
    vol_mult = 1.0 + vol_beta * min(3.0, realized_vol / vol_ref)

    depth = float(liq.get("orderbook_depth_total", mkt.get("depth_total", 0.0)) or 0.0)
    liq_score = float(liq.get("liquidity_score", 0.0) or 0.0)
    if liq_score <= 0.0 and depth > 0.0:
        liq_score = max(0.0, min(1.0, depth / (depth + 1_000_000.0)))
    liq_beta = max(0.0, _env_float("COST_TURNOVER_LIQUIDITY_BETA", 0.5))
    liq_mult = 1.0 + liq_beta * (1.0 - max(0.0, min(1.0, liq_score)))

    spread_bps = float(abs(liq.get("spread_bps", mkt.get("spread_bps", mkt.get("orderbook_spread_bps", 0.0))) or 0.0))
    spread_ref = max(1e-6, _env_float("COST_TURNOVER_SPREAD_REF_BPS", 8.0))
    spread_beta = max(0.0, _env_float("COST_TURNOVER_SPREAD_BETA", 0.25))
    spread_mult = 1.0 + spread_beta * min(3.0, spread_bps / spread_ref)

    funding_rate = float(abs(mkt.get("funding_rate", liq.get("funding_rate", 0.0)) or 0.0))
    funding_ref = max(1e-8, _env_float("COST_TURNOVER_FUNDING_REF", 0.0005))
    funding_beta = max(0.0, _env_float("COST_TURNOVER_FUNDING_BETA", 0.1))
    funding_mult = 1.0 + funding_beta * min(3.0, funding_rate / funding_ref)

    est = float(base * horizon_scale * vol_mult * liq_mult * spread_mult * funding_mult)
    min_turn = max(0.0, _env_float("COST_TURNOVER_MIN", 0.01))
    max_turn = max(min_turn + 1e-6, _env_float("COST_TURNOVER_MAX", 3.0))
    return float(min(max_turn, max(min_turn, est)))


def compute_cost_breakdown_bps(
    *,
    horizon: str,
    profile: CostProfile | None = None,
    symbol: str | None = None,
    ts: datetime | None = None,
    account_state: Mapping[str, Any] | None = None,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    turnover_estimate: float | None = None,
    infra_usd: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    _ = symbol
    _ = ts
    p = profile or load_cost_profile("standard")
    h = str(horizon or "1h").strip().lower()
    mult = float(p.horizon_multipliers.get(h, p.horizon_multipliers.get("1h", 1.0)))

    mkt = dict(market_state or {})
    liq = dict(liquidity_features or {})
    acct = dict(account_state or {})
    cfg = dict(config or {})

    depth = float(liq.get("orderbook_depth_total", liq.get("depth_total", mkt.get("depth_total", 0.0))) or 0.0)
    spread = float(liq.get("spread_bps", mkt.get("spread_bps", mkt.get("orderbook_spread_bps", 0.0))) or 0.0)
    liquidity_score = float(liq.get("liquidity_score", 0.0) or 0.0)
    if liquidity_score <= 0.0 and depth > 0.0:
        liquidity_score = max(0.0, min(1.0, depth / (depth + 1_000_000.0)))
    realized_vol = float(mkt.get("realized_vol", mkt.get("volatility", 0.0)) or 0.0)
    funding_rate = float(mkt.get("funding_rate", liq.get("funding_rate", 0.0)) or 0.0)
    impact_scale = float(cfg.get("impact_scale", mkt.get("impact_scale", 1.0)) or 1.0)

    if turnover_estimate is None:
        turn = estimate_turnover(
            horizon=h,
            market_state=mkt,
            liquidity_features=liq,
            account_state=acct,
            config=cfg,
        )
    else:
        turn = float(max(0.0, turnover_estimate))
    fee_bps = float(max(0.0, p.fee_bps * mult))
    slippage_bps = float(max(0.0, (p.slippage_bps + 0.1 * spread) * mult * (1.0 + p.liquidity_lambda * (1.0 - max(0.0, min(1.0, liquidity_score))))))
    impact_bps = float(max(0.0, p.impact_base_bps * mult * impact_scale * (1.0 + p.turnover_lambda * turn) * (1.0 + 0.5 * max(0.0, realized_vol))))
    funding_bps = float(max(0.0, abs(funding_rate) * p.funding_lambda * _horizon_hours(h) * 10000.0))

    notional = float(acct.get("notional_usd", mkt.get("notional_usd", cfg.get("notional_usd", 0.0))) or 0.0)
    if infra_usd is None:
        infra_usd = float((p.infra_cny_per_hour / max(1e-8, p.usd_cny)) * _horizon_hours(h))
    infra_bps = 0.0
    if notional > 1e-9:
        infra_bps = float(max(0.0, infra_usd) / notional * 1e4)
    else:
        infra_bps = float(max(0.0, p.infra_bps_per_hour * _horizon_hours(h)))

    total_bps = float(max(0.0, fee_bps + slippage_bps + impact_bps + funding_bps + infra_bps))
    return {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "impact_bps": impact_bps,
        "funding_bps": funding_bps,
        "infra_bps": infra_bps,
        "total_bps": total_bps,
    }


def compute_cost_bps(
    *,
    horizon: str,
    profile: CostProfile | None = None,
    symbol: str | None = None,
    ts: datetime | None = None,
    account_state: Mapping[str, Any] | None = None,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    turnover_estimate: float | None = None,
    infra_usd: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    out = compute_cost_breakdown_bps(
        horizon=horizon,
        profile=profile,
        symbol=symbol,
        ts=ts,
        account_state=account_state,
        market_state=market_state,
        liquidity_features=liquidity_features,
        turnover_estimate=turnover_estimate,
        infra_usd=infra_usd,
        config=config,
    )
    return float(out["total_bps"])


def compute_cost_map(
    *,
    horizons: Sequence[str] = HORIZON_ORDER,
    profile: CostProfile | None = None,
    symbol: str | None = None,
    ts: datetime | None = None,
    account_state: Mapping[str, Any] | None = None,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    turnover_estimate: float | None = None,
    infra_usd: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in horizons:
        hh = str(h).strip().lower()
        out[hh] = compute_cost_bps(
            horizon=hh,
            profile=profile,
            symbol=symbol,
            ts=ts,
            account_state=account_state,
            market_state=market_state,
            liquidity_features=liquidity_features,
            turnover_estimate=turnover_estimate,
            infra_usd=infra_usd,
            config=config,
        )
    return out


def cost_profile_snapshot(name: str = "standard") -> Dict[str, Any]:
    p = load_cost_profile(name)
    payload = asdict(p)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["hash"] = hashlib.sha256(body).hexdigest()
    return payload
