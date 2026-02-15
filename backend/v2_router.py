from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from schemas_v2 import (
    AsyncTaskSubmitResponse,
    AutoGateEvaluateRequest,
    AutoGateEvaluateResponse,
    BacktestRunRequest,
    BacktestRunResponse,
    DataQualityAuditUpdate,
    DataQualitySampleRequest,
    DataQualityConsistencyResponse,
    DataQualityStatsResponse,
    DriftEvaluateRequest,
    DriftEvaluateResponse,
    ExecuteOrdersRequest,
    ExecuteOrdersResponse,
    ExecutionOrderStatusResponse,
    KillSwitchResetRequest,
    KillSwitchStateResponse,
    KillSwitchTriggerRequest,
    ModelGateRequest,
    ModelGateResponse,
    RolloutAdvanceRequest,
    RolloutAdvanceResponse,
    RolloutStateResponse,
    IngestEventsRequest,
    IngestEventsResponse,
    SchedulerAuditLogRequest,
    LineageConsistencyRequest,
    LineageConsistencyResponse,
    LiquidPredictRequest,
    PnLAttributionResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    PositionOpeningStatusResponse,
    PortfolioScoreRequest,
    PortfolioScoreResponse,
    PnLAttributionRequest,
    RebalancePosition,
    RiskCheckRequest,
    RiskCheckResponse,
    RiskLimitsResponse,
    RollbackCheckRequest,
    RollbackCheckResponse,
    SignalGenerateRequest,
    SignalGenerateResponse,
    SubmitExecutionOrdersRequest,
    SubmitExecutionOrdersResponse,
    TaskStatusResponse,
    TradeAuditResponse,
    VCPredictRequest,
)
from execution_engine import ExecutionEngine
from metrics import (
    DATA_FRESHNESS_SECONDS,
    EXECUTION_LATENCY_SECONDS,
    EXECUTION_ORDERS_TOTAL,
    EXECUTION_REJECT_RATE,
    MODEL_DRIFT_EVENTS_TOTAL,
    RISK_HARD_BLOCKS_TOTAL,
    SIGNAL_LATENCY_SECONDS,
)
from v2_repository import V2Repository
from task_queue import enqueue_task, get_task

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")

router = APIRouter(prefix="/api/v2", tags=["v2"])
repo = V2Repository(DATABASE_URL)
exec_engine = ExecutionEngine()


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + (2.718281828 ** (-x)))


def _risk_limits() -> Dict[str, float]:
    return {
        "max_single_weight": float(os.getenv("RISK_MAX_SINGLE_WEIGHT", "0.2")),
        "max_gross_exposure": float(os.getenv("RISK_MAX_GROSS_EXPOSURE", "1.0")),
        "max_turnover_per_rebalance": float(os.getenv("RISK_MAX_TURNOVER", "0.35")),
        "max_realized_drawdown": float(os.getenv("RISK_MAX_DRAWDOWN", "0.2")),
        "max_sector_exposure": float(os.getenv("RISK_MAX_SECTOR_EXPOSURE", "0.45")),
        "max_style_exposure": float(os.getenv("RISK_MAX_STYLE_EXPOSURE", "0.55")),
    }


def _risk_runtime_limits() -> Dict[str, float]:
    return {
        "max_daily_loss": float(os.getenv("RISK_MAX_DAILY_LOSS", "0.03")),
        "max_consecutive_losses": float(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "5")),
    }


def _risk_hard_block_minutes() -> int:
    return int(max(1, min(60 * 24, int(os.getenv("RISK_HARD_BLOCK_MINUTES", "1")))))


def _infer_daily_loss_ratio(track: str) -> float:
    try:
        return float(repo.get_execution_daily_loss_ratio(track=track, lookback_hours=24))
    except Exception:
        totals = repo.build_pnl_attribution(track=track, lookback_hours=24).get("totals", {})
        net_pnl = float(totals.get("net_pnl", 0.0) or 0.0)
        gross_notional = abs(float(totals.get("gross_notional_signed", 0.0) or 0.0))
        if net_pnl >= 0 or gross_notional <= 1e-9:
            return 0.0
        return float(max(0.0, min(5.0, (-net_pnl) / gross_notional)))


def _execution_volatility_violations(orders: List[Dict[str, Any]]) -> List[str]:
    def _symbol_threshold(target: str) -> float:
        base = float(os.getenv("RISK_MAX_ABS_RETURN", "0.06"))
        raw_map = os.getenv("RISK_MAX_ABS_RETURN_SYMBOLS", "").strip()
        symbol_threshold = base
        if raw_map:
            for chunk in raw_map.split(","):
                part = chunk.strip()
                if not part or "=" not in part:
                    continue
                k, v = part.split("=", 1)
                if k.strip().upper() == target.upper():
                    try:
                        symbol_threshold = float(v.strip())
                    except Exception:
                        symbol_threshold = base
                    break
        multiplier = 1.0
        tod = os.getenv("RISK_MAX_ABS_RETURN_TOD_MULTIPLIER", "").strip()
        if tod:
            hour = datetime.utcnow().hour
            for rule in tod.split(","):
                piece = rule.strip()
                if not piece or ":" not in piece:
                    continue
                span, mult_raw = piece.split(":", 1)
                if "-" not in span:
                    continue
                start_raw, end_raw = span.split("-", 1)
                try:
                    start = int(start_raw.strip())
                    end = int(end_raw.strip())
                    mult = float(mult_raw.strip())
                except Exception:
                    continue
                if start <= hour <= end:
                    multiplier = mult
                    break
        return max(0.0, symbol_threshold * multiplier)

    out: List[str] = []
    for target in sorted({str(o.get("target") or "").upper() for o in orders if str(o.get("target") or "").strip()}):
        max_abs_ret = _symbol_threshold(target)
        if max_abs_ret <= 0:
            continue
        rows = repo.load_price_history(target, lookback_days=2)
        prices = np.array([float(r.get("price") or 0.0) for r in rows if float(r.get("price") or 0.0) > 0], dtype=np.float64)
        if prices.size < 32:
            continue
        rets = np.diff(prices) / np.clip(prices[:-1], 1e-12, None)
        recent = rets[-96:] if rets.size > 96 else rets
        if recent.size == 0:
            continue
        if float(np.max(np.abs(recent))) > max_abs_ret:
            out.append(f"abnormal_volatility_circuit_breaker:{target}")
    return out


def _normalize_execution_payload(execution: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(execution or {})
    lifecycle_in = execution.get("lifecycle") if isinstance(execution, dict) else []
    lifecycle_out: List[Dict[str, Any]] = []
    now_iso = datetime.utcnow().isoformat() + "Z"
    if isinstance(lifecycle_in, list):
        for ev in lifecycle_in:
            if not isinstance(ev, dict):
                continue
            metrics: Dict[str, float] = {}
            for k in ("filled_qty", "remaining_qty", "avg_fill_price", "retry", "http_code", "timeout_sec"):
                v = ev.get(k)
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            lifecycle_out.append(
                {
                    "event": str(ev.get("event") or "unknown"),
                    "status": str(ev.get("status") or "unknown"),
                    "time": str(ev.get("time") or now_iso),
                    "metrics": metrics,
                }
            )
    out["lifecycle"] = lifecycle_out
    return out


def _bucket_limits() -> Dict[str, float]:
    return {
        "trend": float(os.getenv("RISK_BUCKET_TREND", "0.5")),
        "event": float(os.getenv("RISK_BUCKET_EVENT", "0.4")),
        "mean_reversion": float(os.getenv("RISK_BUCKET_MEAN_REVERSION", "0.35")),
    }


def _position_map(positions: List[RebalancePosition]) -> Dict[Tuple[str, str], float]:
    return {(p.target.upper(), p.track): float(p.weight) for p in positions}


def _fallback_sector(target: str, track: str) -> str:
    t = target.upper()
    if track == "vc":
        return "private"
    if t in {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}:
        return "crypto"
    return "equity"


def _fallback_style(target: str, track: str) -> str:
    t = target.upper()
    if track == "vc":
        return "venture"
    if t in {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}:
        return "crypto_beta"
    return "core"


def _resolve_position_profiles(positions: List[RebalancePosition]) -> List[RebalancePosition]:
    if not positions:
        return positions
    missing_targets = [p.target.upper() for p in positions if not p.sector or not p.style_bucket]
    try:
        target_profiles = repo.get_target_profiles(missing_targets) if missing_targets else {}
    except Exception:
        target_profiles = {}
    resolved: List[RebalancePosition] = []
    for p in positions:
        key = p.target.upper()
        profile = target_profiles.get(key, {})
        sector = p.sector or profile.get("sector") or _fallback_sector(key, p.track)
        style = p.style_bucket or profile.get("style_bucket") or _fallback_style(key, p.track)
        resolved.append(RebalancePosition(target=key, track=p.track, weight=float(p.weight), sector=sector, style_bucket=style))
    return resolved


def _kill_switch_block_reason(track: str, strategy_id: str = "global") -> Optional[str]:
    if repo.is_kill_switch_triggered(track, strategy_id):
        return f"kill_switch_triggered:{track}:{strategy_id}"
    if strategy_id != "global" and repo.is_kill_switch_triggered(track, "global"):
        return f"kill_switch_triggered:{track}:global"
    return None


def _remaining_seconds(expires_at: Optional[datetime]) -> int:
    if not isinstance(expires_at, datetime):
        return 0
    exp = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return int(max(0.0, (exp - now).total_seconds()))


def _evaluate_risk(
    proposed: List[RebalancePosition],
    current: List[RebalancePosition],
    realized_drawdown: float,
    max_sector_exposure_override: Optional[float] = None,
    max_style_exposure_override: Optional[float] = None,
) -> Tuple[List[RebalancePosition], List[str], float, float]:
    limits = _risk_limits()
    if max_sector_exposure_override is not None:
        limits["max_sector_exposure"] = max_sector_exposure_override
    if max_style_exposure_override is not None:
        limits["max_style_exposure"] = max_style_exposure_override
    violations: List[str] = []
    proposed = _resolve_position_profiles(proposed)
    current = _resolve_position_profiles(current)

    capped: List[RebalancePosition] = []
    for p in proposed:
        w = float(p.weight)
        max_single = limits["max_single_weight"]
        if abs(w) > max_single:
            violations.append(f"single_weight_exceeded:{p.target}")
            w = max_single if w > 0 else -max_single
        capped.append(
            RebalancePosition(
                target=p.target,
                track=p.track,
                weight=w,
                sector=p.sector,
                style_bucket=p.style_bucket,
            )
        )

    gross = sum(abs(p.weight) for p in capped)
    if gross > limits["max_gross_exposure"] and gross > 0:
        violations.append("gross_exposure_exceeded")
        scale = limits["max_gross_exposure"] / gross
        capped = [
            RebalancePosition(
                target=p.target,
                track=p.track,
                weight=p.weight * scale,
                sector=p.sector,
                style_bucket=p.style_bucket,
            )
            for p in capped
        ]
        gross = limits["max_gross_exposure"]

    sector_exposure: Dict[str, float] = {}
    for p in capped:
        k = str(p.sector or _fallback_sector(p.target, p.track)).lower()
        sector_exposure[k] = sector_exposure.get(k, 0.0) + abs(float(p.weight))
    for sector, exp in sector_exposure.items():
        if exp > limits["max_sector_exposure"] and exp > 0:
            violations.append(f"sector_exposure_exceeded:{sector}")
            scale = limits["max_sector_exposure"] / exp
            capped = [
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=p.weight * scale if str(p.sector or _fallback_sector(p.target, p.track)).lower() == sector else p.weight,
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
                for p in capped
            ]

    style_exposure: Dict[str, float] = {}
    for p in capped:
        k = str(p.style_bucket or _fallback_style(p.target, p.track)).lower()
        style_exposure[k] = style_exposure.get(k, 0.0) + abs(float(p.weight))
    for style, exp in style_exposure.items():
        if exp > limits["max_style_exposure"] and exp > 0:
            violations.append(f"style_exposure_exceeded:{style}")
            scale = limits["max_style_exposure"] / exp
            capped = [
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=p.weight * scale if str(p.style_bucket or _fallback_style(p.target, p.track)).lower() == style else p.weight,
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
                for p in capped
            ]

    gross = sum(abs(p.weight) for p in capped)

    current_map = _position_map(current)
    next_map = _position_map(capped)
    universe = set(current_map.keys()) | set(next_map.keys())
    turnover = sum(abs(next_map.get(k, 0.0) - current_map.get(k, 0.0)) for k in universe)
    if turnover > limits["max_turnover_per_rebalance"]:
        violations.append("turnover_exceeded")

    if realized_drawdown > limits["max_realized_drawdown"]:
        violations.append("drawdown_exceeded")

    return capped, violations, gross, turnover


def _build_vc_prediction(company_name: str, horizon_months: int) -> Dict[str, object]:
    context = repo.recent_event_context(company_name, limit=8)

    funding_hits = sum(1 for e in context if e["event_type"] == "funding")
    product_hits = sum(1 for e in context if e["event_type"] == "product")
    regulatory_hits = sum(1 for e in context if e["event_type"] == "regulatory")
    score_raw = funding_hits * 0.9 + product_hits * 0.4 - regulatory_hits * 0.6

    horizon_adj = {6: 0.1, 12: 0.0, 24: -0.05}[horizon_months]
    p_next_round = max(0.01, min(0.99, _sigmoid(score_raw * 0.7 + horizon_adj)))
    p_exit = max(0.01, min(0.95, _sigmoid(score_raw * 0.3 - 0.4)))
    expected_moic = {
        "p10": round(0.6 + p_next_round * 0.8, 2),
        "p50": round(1.0 + p_next_round * 1.6, 2),
        "p90": round(1.2 + p_next_round * 3.5 + p_exit * 2.0, 2),
    }

    explanation = {
        "top_event_contributors": [
            {
                "event_id": e["id"],
                "event_type": e["event_type"],
                "title": e["title"],
                "weight": round(0.2 + e.get("confidence_score", 0.5) * 0.6, 3),
            }
            for e in context[:5]
        ],
        "top_feature_contributors": [
            {"feature": "recent_funding_event_count", "value": funding_hits, "contribution": round(funding_hits * 0.9, 3)},
            {"feature": "recent_product_event_count", "value": product_hits, "contribution": round(product_hits * 0.4, 3)},
            {"feature": "recent_regulatory_event_count", "value": regulatory_hits, "contribution": round(-regulatory_hits * 0.6, 3)},
        ],
        "evidence_links": [e["source_url"] for e in context if e.get("source_url")][:5],
        "model_version": "vc-survival-baseline-v2",
        "feature_version": "feature-store-v2.0",
    }

    outputs = {
        "p_next_round": {
            "6m": round(max(0.01, min(0.99, p_next_round + 0.05)), 4),
            "12m": round(p_next_round, 4),
            "24m": round(max(0.01, min(0.99, p_next_round - 0.08)), 4),
        },
        "p_exit_24m": round(p_exit, 4),
        "expected_moic_distribution": expected_moic,
        "as_of": datetime.utcnow().isoformat(),
    }

    return {
        "target": company_name,
        "score": float(round(p_next_round, 4)),
        "confidence": float(round(min(0.95, 0.55 + len(context) * 0.04), 4)),
        "outputs": outputs,
        "explanation": explanation,
    }


def _build_liquid_prediction(symbol: str, horizon: str) -> Dict[str, object]:
    price_row = repo.latest_price_snapshot(symbol)
    if not price_row:
        raise HTTPException(status_code=404, detail=f"no price snapshot for {symbol}")
    ts = price_row.get("timestamp")
    if isinstance(ts, datetime):
        ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        freshness = max(0.0, (datetime.utcnow().replace(tzinfo=timezone.utc) - ts_utc.astimezone(timezone.utc)).total_seconds())
        DATA_FRESHNESS_SECONDS.labels(target=symbol.upper()).set(freshness)

    context = repo.recent_event_context(symbol, limit=8)
    market_events = sum(1 for e in context if e["event_type"] == "market")
    regulatory_events = sum(1 for e in context if e["event_type"] == "regulatory")

    horizon_scale = {"1h": 0.3, "1d": 1.0, "7d": 2.1}[horizon]
    base_return = (market_events * 0.002 - regulatory_events * 0.003) * horizon_scale
    vol_forecast = max(0.01, 0.02 + (market_events + regulatory_events) * 0.0025)
    confidence = max(0.3, min(0.95, 0.55 + len(context) * 0.03))

    outputs = {
        "expected_return": round(base_return, 6),
        "vol_forecast": round(vol_forecast, 6),
        "signal_confidence": round(confidence, 4),
        "current_price": float(price_row["price"]),
        "horizon": horizon,
        "as_of": datetime.utcnow().isoformat(),
    }

    explanation = {
        "top_event_contributors": [
            {
                "event_id": e["id"],
                "event_type": e["event_type"],
                "title": e["title"],
                "weight": round(0.1 + e.get("confidence_score", 0.5) * 0.7, 3),
            }
            for e in context[:5]
        ],
        "top_feature_contributors": [
            {"feature": "recent_market_event_count", "value": market_events, "contribution": round(market_events * 0.002, 6)},
            {"feature": "recent_regulatory_event_count", "value": regulatory_events, "contribution": round(-regulatory_events * 0.003, 6)},
            {"feature": "price_level", "value": float(price_row["price"]), "contribution": 0.0},
        ],
        "evidence_links": [e["source_url"] for e in context if e.get("source_url")][:5],
        "model_version": "liquid-tsfm-adapter-baseline-v2",
        "feature_version": "feature-store-v2.0",
    }

    return {
        "target": symbol.upper(),
        "score": float(round(base_return, 6)),
        "confidence": float(round(confidence, 4)),
        "outputs": outputs,
        "explanation": explanation,
    }


def _walk_forward_metrics(price_rows: List[Dict], train_days: int, test_days: int, fee_bps: float, slippage_bps: float) -> Dict[str, float]:
    # Crypto runs 7x24; keep full timeline without weekend filtering.
    rows = list(price_rows)
    if len(rows) < (train_days + test_days + 2):
        return {
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
        }

    prices = np.array([float(r["price"]) for r in rows], dtype=np.float64)
    volumes = np.array([float(r.get("volume") or 0.0) for r in rows], dtype=np.float64)
    rets = np.diff(prices) / np.clip(prices[:-1], 1e-12, None)

    scores: List[float] = []
    future_rets: List[float] = []
    equity_curve: List[float] = [1.0]
    wins = 0
    trades = 0
    turnover = 0.0
    prev_signal = 0.0

    start = train_days
    while start < len(rets) - 1:
        end = min(start + test_days, len(rets) - 1)
        train_slice = rets[start - train_days : start]
        mu = float(np.mean(train_slice)) if train_slice.size else 0.0
        sigma = float(np.std(train_slice)) if train_slice.size else 0.0
        threshold = max(1e-6, sigma * 0.2)

        for i in range(start, end):
            mom = float(rets[i - 1] - mu)
            signal = 1.0 if mom > threshold else -1.0 if mom < -threshold else 0.0
            # capacity/impact proxy: cap exposure by recent realized volatility
            cap = float(max(0.2, min(1.0, 0.8 - sigma * 20.0)))
            signal = signal * cap
            nxt = float(rets[i])
            vol = float(max(1.0, volumes[i]))
            impact_bps = min(25.0, 1200.0 / np.sqrt(vol))
            cost = ((fee_bps + slippage_bps + impact_bps) / 10000.0) * abs(signal - prev_signal)
            pnl = signal * nxt - cost

            eq = equity_curve[-1] * (1.0 + pnl)
            equity_curve.append(eq)

            scores.append(signal)
            future_rets.append(nxt)
            turnover += abs(signal - prev_signal)
            trades += 1
            if signal * nxt > 0:
                wins += 1
            prev_signal = signal

        start += test_days

    if not scores:
        return {
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
        }

    s = np.array(scores, dtype=np.float64)
    y = np.array(future_rets, dtype=np.float64)
    ic = float(np.corrcoef(s, y)[0, 1]) if np.std(s) > 0 and np.std(y) > 0 else 0.0
    hit_rate = float(wins / max(1, trades))
    pnl_after_cost = float(equity_curve[-1] - 1.0)

    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / max(peak, 1e-12)
        max_dd = max(max_dd, dd)

    return {
        "samples": int(len(scores)),
        "ic": round(ic, 6),
        "hit_rate": round(hit_rate, 6),
        "turnover": round(float(turnover), 6),
        "pnl_after_cost": round(pnl_after_cost, 6),
        "max_drawdown": round(float(max_dd), 6),
    }


def _feature_signal_score(payload: Dict[str, Any]) -> float:
    if not isinstance(payload, dict):
        return 0.0
    ret_1 = float(payload.get("ret_1", 0.0) or 0.0)
    ret_3 = float(payload.get("ret_3", 0.0) or 0.0)
    ret_12 = float(payload.get("ret_12", 0.0) or 0.0)
    ret_48 = float(payload.get("ret_48", 0.0) or 0.0)
    vol_12 = float(payload.get("vol_12", 0.0) or 0.0)
    vol_48 = float(payload.get("vol_48", 0.0) or 0.0)
    vol_96 = float(payload.get("vol_96", 0.0) or 0.0)
    ob = float(payload.get("orderbook_imbalance", 0.0) or 0.0)
    funding = float(payload.get("funding_rate", 0.0) or 0.0)
    onchain = float(payload.get("onchain_norm", 0.0) or 0.0)
    event_decay = float(payload.get("event_decay", 0.0) or 0.0)
    vol_penalty = abs(vol_12) + abs(vol_48) + abs(vol_96)
    score = (
        0.20 * ret_1
        + 0.25 * ret_3
        + 0.35 * ret_12
        + 0.15 * ret_48
        + 0.10 * ob
        + 0.08 * funding
        + 0.07 * onchain
        + 0.05 * event_decay
        - 0.15 * vol_penalty
    )
    return float(score)


def _run_model_replay_backtest(
    feature_rows: List[Dict[str, Any]],
    price_rows: List[Dict[str, Any]],
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    if not feature_rows:
        return {
            "status": "failed",
            "reason": "insufficient_features",
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "lineage_coverage": 0.0,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
        }
    if len(price_rows) < 3:
        return {
            "status": "failed",
            "reason": "insufficient_prices",
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "lineage_coverage": 0.0,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
        }
    prices = np.array([float(r.get("price") or 0.0) for r in price_rows], dtype=np.float64)
    vols = np.array([float(r.get("volume") or 0.0) for r in price_rows], dtype=np.float64)
    rets = np.diff(prices) / np.clip(prices[:-1], 1e-12, None)
    n = min(len(feature_rows), len(rets))
    if n < 8:
        return {
            "status": "failed",
            "reason": "insufficient_features",
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "lineage_coverage": 0.0,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
        }

    scores: List[float] = []
    future_rets: List[float] = []
    prev_signal = 0.0
    eq = 1.0
    eq_curve = [eq]
    fee_cost = 0.0
    slippage_cost = 0.0
    impact_cost = 0.0
    wins = 0
    turnover = 0.0
    unique_lineages = set()
    for i in range(n):
        f = feature_rows[i].get("feature_payload") or {}
        unique_lineages.add(str(feature_rows[i].get("lineage_id") or ""))
        raw = _feature_signal_score(f)
        signal = 1.0 if raw > 0.0005 else -1.0 if raw < -0.0005 else 0.0
        nxt = float(rets[i])
        vol = float(max(1.0, vols[i + 1] if i + 1 < len(vols) else vols[-1]))
        impact_bps = min(30.0, 1200.0 / np.sqrt(vol))
        c_fee = fee_bps / 10000.0 * abs(signal - prev_signal)
        c_slip = slippage_bps / 10000.0 * abs(signal - prev_signal)
        c_imp = impact_bps / 10000.0 * abs(signal - prev_signal)
        pnl = signal * nxt - (c_fee + c_slip + c_imp)
        eq *= (1.0 + pnl)
        eq_curve.append(eq)
        fee_cost += c_fee
        slippage_cost += c_slip
        impact_cost += c_imp
        scores.append(signal)
        future_rets.append(nxt)
        if signal * nxt > 0:
            wins += 1
        turnover += abs(signal - prev_signal)
        prev_signal = signal

    s = np.array(scores, dtype=np.float64)
    y = np.array(future_rets, dtype=np.float64)
    ic = float(np.corrcoef(s, y)[0, 1]) if np.std(s) > 0 and np.std(y) > 0 else 0.0
    hit_rate = float(wins / max(1, len(scores)))
    peak = eq_curve[0]
    max_dd = 0.0
    for v in eq_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / max(peak, 1e-12))
    return {
        "status": "completed",
        "reason": "ok",
        "samples": int(len(scores)),
        "ic": round(ic, 6),
        "hit_rate": round(hit_rate, 6),
        "turnover": round(float(turnover), 6),
        "pnl_after_cost": round(float(eq_curve[-1] - 1.0), 6),
        "max_drawdown": round(float(max_dd), 6),
        "lineage_coverage": round(float(len(unique_lineages) / max(1, len(scores))), 6),
        "cost_breakdown": {
            "fee": round(float(fee_cost), 8),
            "slippage": round(float(slippage_cost), 8),
            "impact": round(float(impact_cost), 8),
        },
    }


def _default_model_by_track(track: str) -> Tuple[str, str]:
    if track == "liquid":
        return "liquid_ttm_ensemble", "v2.1"
    return "vc_survival_model", "v2.1"


def _evaluate_gate(
    track: str,
    min_ic: float,
    min_pnl_after_cost: float,
    max_drawdown: float,
    windows: int,
) -> Tuple[bool, str, Dict[str, float], int]:
    runs = repo.list_recent_backtest_runs(track=track, limit=windows)
    usable = [r for r in runs if isinstance(r.get("metrics"), dict) and r["metrics"].get("status") == "completed"]
    if len(usable) < windows:
        return False, "insufficient_windows", {"ic": 0.0, "pnl_after_cost": 0.0, "max_drawdown": 1.0}, len(usable)

    ic_vals = [float(r["metrics"].get("ic", 0.0)) for r in usable[:windows]]
    pnl_vals = [float(r["metrics"].get("pnl_after_cost", 0.0)) for r in usable[:windows]]
    dd_vals = [float(r["metrics"].get("max_drawdown", 1.0)) for r in usable[:windows]]

    summary = {
        "ic": round(float(np.mean(ic_vals)), 6),
        "pnl_after_cost": round(float(np.mean(pnl_vals)), 6),
        "max_drawdown": round(float(np.mean(dd_vals)), 6),
    }
    passed = summary["ic"] >= min_ic and summary["pnl_after_cost"] >= min_pnl_after_cost and summary["max_drawdown"] <= max_drawdown
    reason = "passed" if passed else "threshold_failed"
    return passed, reason, summary, windows


def _strategy_bucket(track: str, score: float, confidence: float, target: str) -> str:
    if track == "vc":
        return "event"
    px = repo.latest_price_snapshot(target)
    if not px:
        return "event"
    price = float(px.get("price") or 0.0)
    volume = float(px.get("volume") or 0.0)
    fast_signal = np.tanh(np.log1p(max(volume, 0.0)) * 0.05)
    slow_signal = np.tanh(np.log1p(max(price, 0.0)) * 0.03 + score * 2.0)
    mixed = 0.6 * slow_signal + 0.4 * fast_signal
    if abs(mixed) > 0.35 and confidence > 0.55:
        return "trend"
    if abs(mixed) < 0.1 and confidence > 0.6:
        return "mean_reversion"
    return "event"


def _ks_statistic(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    x = np.sort(np.array(a, dtype=np.float64))
    y = np.sort(np.array(b, dtype=np.float64))
    all_vals = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, all_vals, side="right") / max(1, len(x))
    cdf_y = np.searchsorted(y, all_vals, side="right") / max(1, len(y))
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _psi(reference: List[float], current: List[float], bins: int = 10) -> float:
    if not reference or not current:
        return 0.0
    ref = np.array(reference, dtype=np.float64)
    cur = np.array(current, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _flatten_numeric_payloads(payloads: List[Dict[str, float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for p in payloads:
        if not isinstance(p, dict):
            continue
        for k, v in p.items():
            if isinstance(v, (int, float)):
                out.setdefault(str(k), []).append(float(v))
    return out


def _feature_drift_score(reference_payloads: List[Dict[str, float]], current_payloads: List[Dict[str, float]]) -> Tuple[float, float]:
    ref_map = _flatten_numeric_payloads(reference_payloads)
    cur_map = _flatten_numeric_payloads(current_payloads)
    keys = set(ref_map.keys()) | set(cur_map.keys())
    if not keys:
        return 0.0, 0.0
    psi_vals = []
    mean_shift = []
    for k in keys:
        rv = ref_map.get(k, [])
        cv = cur_map.get(k, [])
        if rv and cv:
            psi_vals.append(_psi(rv, cv))
            mean_shift.append(abs(float(np.mean(cv)) - float(np.mean(rv))))
    if not psi_vals:
        return 0.0, 0.0
    return float(np.mean(psi_vals)), float(np.mean(mean_shift))


@router.post("/ingest/events", response_model=IngestEventsResponse)
async def ingest_events(payload: IngestEventsRequest) -> IngestEventsResponse:
    accepted, inserted, deduplicated, event_ids = repo.ingest_events(payload.events)
    return IngestEventsResponse(
        accepted=accepted,
        inserted=inserted,
        deduplicated=deduplicated,
        event_ids=event_ids,
    )


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: int):
    entity = repo.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="entity not found")
    return entity


@router.post("/predict/vc")
async def predict_vc(payload: VCPredictRequest):
    result = _build_vc_prediction(payload.company_name, payload.horizon_months)
    prediction_id = repo.insert_prediction(
        track="vc",
        target=payload.company_name,
        score=result["score"],
        confidence=result["confidence"],
        outputs=result["outputs"],
        explanation=result["explanation"],
        horizon=f"{payload.horizon_months}m",
        feature_set_id="feature-store-v2.0",
    )

    return {
        "prediction_id": prediction_id,
        "track": "vc",
        "target": payload.company_name,
        "outputs": result["outputs"],
        "explanation": result["explanation"],
    }


@router.post("/predict/liquid")
async def predict_liquid(payload: LiquidPredictRequest):
    result = _build_liquid_prediction(payload.symbol, payload.horizon)
    prediction_id = repo.insert_prediction(
        track="liquid",
        target=payload.symbol.upper(),
        score=result["score"],
        confidence=result["confidence"],
        outputs=result["outputs"],
        explanation=result["explanation"],
        horizon=payload.horizon,
        feature_set_id="feature-store-v2.0",
    )

    return {
        "prediction_id": prediction_id,
        "track": "liquid",
        "target": payload.symbol.upper(),
        "outputs": result["outputs"],
        "explanation": result["explanation"],
    }


@router.post("/portfolio/score", response_model=PortfolioScoreResponse)
async def portfolio_score(payload: PortfolioScoreRequest) -> PortfolioScoreResponse:
    if not payload.positions:
        raise HTTPException(status_code=400, detail="positions cannot be empty")

    weighted_return = sum(p.score * (1.0 / max(p.risk, 0.01)) for p in payload.positions)
    risk_proxy = sum(p.risk for p in payload.positions) / len(payload.positions)

    alpha_score = weighted_return / len(payload.positions)
    expected_vol = min(2.5, risk_proxy / payload.risk_budget)

    recommendations: List[Dict] = []
    for p in payload.positions:
        action = "hold"
        if p.score > 0 and p.risk < payload.risk_budget:
            action = "increase"
        elif p.score < 0:
            action = "reduce"
        recommendations.append({"target": p.target, "action": action, "track": p.track})

    return PortfolioScoreResponse(
        alpha_score=round(alpha_score, 6),
        expected_return=round(alpha_score * 0.8, 6),
        expected_volatility=round(expected_vol, 6),
        recommendations=recommendations,
    )


@router.get("/predictions/{prediction_id}/explanation")
async def get_prediction_explanation(prediction_id: int):
    row = repo.get_prediction_explanation(prediction_id)
    if not row:
        raise HTTPException(status_code=404, detail="prediction not found")
    return row


@router.post("/backtest/run", response_model=BacktestRunResponse)
async def run_backtest(payload: BacktestRunRequest) -> BacktestRunResponse:
    targets = payload.targets
    if not targets:
        if payload.track == "liquid":
            targets = [s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()]
        else:
            targets = ["OpenAI", "Anthropic", "Scale AI"]

    run_name = f"{payload.track}-wf-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    config = payload.model_dump()
    config["targets"] = targets
    run_id = repo.create_backtest_run(run_name=run_name, track=payload.track, config=config)
    model_name = payload.model_name or _default_model_by_track(payload.track)[0]
    model_version = payload.model_version or _default_model_by_track(payload.track)[1]
    if not repo.model_artifact_exists(model_name=model_name, track=payload.track, model_version=model_version):
        agg = {
            "status": "failed",
            "reason": "model_artifact_missing",
            "targets": targets,
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "model_name": model_name,
            "model_version": model_version,
            "lineage_coverage": 0.0,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
            "gate_passed": False,
        }
        repo.finish_backtest_run(run_id, agg)
        return BacktestRunResponse(
            run_id=run_id,
            run_name=run_name,
            track=payload.track,
            status="failed",
            metrics=agg,
            config=config,
        )

    all_metrics: List[Dict[str, float]] = []
    for target in targets:
        feature_rows = repo.load_feature_history(
            target=target,
            track=payload.track,
            lookback_days=payload.lookback_days,
            data_version=payload.data_version,
        )
        price_rows = repo.load_price_history(target, lookback_days=payload.lookback_days)
        m = _run_model_replay_backtest(
            feature_rows=feature_rows,
            price_rows=price_rows,
            fee_bps=payload.fee_bps,
            slippage_bps=payload.slippage_bps,
        )
        all_metrics.append(m)

    samples = sum(int(m.get("samples", 0)) for m in all_metrics)
    if samples == 0:
        agg = {
            "status": "failed",
            "reason": "insufficient_features" if any(m.get("reason") == "insufficient_features" for m in all_metrics) else "insufficient_prices",
            "targets": targets,
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "model_name": model_name,
            "model_version": model_version,
            "lineage_coverage": 0.0,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
            "gate_passed": False,
        }
    else:
        agg = {
            "status": "completed",
            "targets": targets,
            "samples": int(samples),
            "ic": round(float(np.mean([m["ic"] for m in all_metrics])), 6),
            "hit_rate": round(float(np.mean([m["hit_rate"] for m in all_metrics])), 6),
            "turnover": round(float(np.mean([m["turnover"] for m in all_metrics])), 6),
            "pnl_after_cost": round(float(np.mean([m["pnl_after_cost"] for m in all_metrics])), 6),
            "max_drawdown": round(float(np.mean([m["max_drawdown"] for m in all_metrics])), 6),
            "model_name": model_name,
            "model_version": model_version,
            "lineage_coverage": round(float(np.mean([float(m.get("lineage_coverage", 0.0)) for m in all_metrics])), 6),
            "cost_breakdown": {
                "fee": round(float(np.mean([float((m.get("cost_breakdown") or {}).get("fee", 0.0)) for m in all_metrics])), 8),
                "slippage": round(float(np.mean([float((m.get("cost_breakdown") or {}).get("slippage", 0.0)) for m in all_metrics])), 8),
                "impact": round(float(np.mean([float((m.get("cost_breakdown") or {}).get("impact", 0.0)) for m in all_metrics])), 8),
            },
        }
        agg["gate_passed"] = bool(agg["ic"] > 0 and agg["pnl_after_cost"] > 0 and agg["max_drawdown"] < 0.2)

    repo.finish_backtest_run(run_id, agg)

    gate_model_name = model_name
    gate_model_version = model_version
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=0.0,
        min_pnl_after_cost=0.0,
        max_drawdown=0.2,
        windows=3,
    )
    repo.promote_model(
        track=payload.track,
        model_name=gate_model_name,
        model_version=gate_model_version,
        passed=passed,
        metrics={**summary, "windows_checked": checked},
        gate_reason=reason,
    )
    current = repo.get_active_model_state(payload.track)
    prev_name = current["active_model_name"] if current else None
    prev_ver = current["active_model_version"] if current else None
    if passed:
        repo.upsert_active_model_state(
            track=payload.track,
            active_model_name=gate_model_name,
            active_model_version=gate_model_version,
            previous_model_name=prev_name,
            previous_model_version=prev_ver,
            status="active",
            metadata={"source": "auto_gate", "summary": summary},
        )

    return BacktestRunResponse(
        run_id=run_id,
        run_name=run_name,
        track=payload.track,
        status=agg["status"],
        metrics=agg,
        config=config,
    )


@router.post("/tasks/backtest", response_model=AsyncTaskSubmitResponse)
async def submit_backtest_task(payload: BacktestRunRequest) -> AsyncTaskSubmitResponse:
    task_id = enqueue_task("backtest_run", payload.model_dump())
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type="backtest_run",
        status="queued",
        created_at=datetime.utcnow(),
    )


@router.get("/backtest/{run_id}")
async def get_backtest(run_id: int):
    row = repo.get_backtest_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="backtest run not found")
    return row


@router.post("/signals/generate", response_model=SignalGenerateResponse)
async def generate_signal(payload: SignalGenerateRequest) -> SignalGenerateResponse:
    _t0 = time.perf_counter()
    kill_block = _kill_switch_block_reason(payload.track, payload.strategy_id)
    if kill_block:
        raise HTTPException(status_code=423, detail=kill_block)
    decision_id = uuid.uuid4().hex

    if payload.track == "liquid":
        pred = _build_liquid_prediction(payload.target, payload.horizon)
        model_version = "liquid-tsfm-adapter-baseline-v2"
    else:
        pred = _build_vc_prediction(payload.target, 12)
        model_version = "vc-survival-baseline-v2"

    action = "hold"
    score = float(pred["score"])
    confidence = float(pred["confidence"])
    bucket = _strategy_bucket(payload.track, score, confidence, payload.target)

    if payload.policy == "ensemble-v1":
        # slow model: historical/event score, fast model: recent price-volume pulse
        px = repo.latest_price_snapshot(payload.target)
        fast = 0.0
        if px:
            p = float(px.get("price") or 0.0)
            v = float(px.get("volume") or 0.0)
            fast = float(np.tanh(np.log1p(max(v, 0.0)) * 0.03 + np.log1p(max(p, 0.0)) * 0.01))
        score = round(0.7 * score + 0.3 * fast, 6)
        confidence = round(min(0.95, confidence + 0.03), 4)

    if confidence >= payload.min_confidence:
        if score > 0.01:
            action = "buy"
        elif score < -0.01:
            action = "sell"

    reason = f"policy={payload.policy};model={model_version};score={round(score, 6)}"
    signal_id = repo.insert_signal_candidate(
        track=payload.track,
        target=payload.target.upper(),
        horizon=payload.horizon,
        score=score,
        confidence=confidence,
        action=action,
        policy=payload.policy,
        decision_id=decision_id,
        metadata={
            "reason": reason,
            "min_confidence": payload.min_confidence,
            "strategy_id": payload.strategy_id,
            "cost_profile": payload.cost_profile,
            "risk_profile": payload.risk_profile,
        },
    )

    resp = SignalGenerateResponse(
        signal_id=signal_id,
        track=payload.track,
        target=payload.target.upper(),
        horizon=payload.horizon,
        action=action,
        score=score,
        confidence=confidence,
        reason=reason,
        policy=payload.policy,
        strategy_bucket=bucket,
        created_at=datetime.utcnow(),
    )
    SIGNAL_LATENCY_SECONDS.labels(track=payload.track).observe(max(0.0, time.perf_counter() - _t0))
    return resp


@router.post("/models/gate/evaluate", response_model=ModelGateResponse)
async def evaluate_model_gate(payload: ModelGateRequest) -> ModelGateResponse:
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=payload.min_ic,
        min_pnl_after_cost=payload.min_pnl_after_cost,
        max_drawdown=payload.max_drawdown,
        windows=payload.windows,
    )
    repo.promote_model(
        track=payload.track,
        model_name=payload.model_name,
        model_version=payload.model_version,
        passed=passed,
        metrics={**summary, "windows_checked": checked},
        gate_reason=reason,
    )
    if passed:
        current = repo.get_active_model_state(payload.track)
        prev_name = current["active_model_name"] if current else None
        prev_ver = current["active_model_version"] if current else None
        repo.upsert_active_model_state(
            track=payload.track,
            active_model_name=payload.model_name,
            active_model_version=payload.model_version,
            previous_model_name=prev_name,
            previous_model_version=prev_ver,
            status="active",
            metadata={"source": "manual_gate", "summary": summary},
        )

    return ModelGateResponse(
        passed=passed,
        track=payload.track,
        model_name=payload.model_name,
        model_version=payload.model_version,
        reason=reason,
        windows_checked=checked,
        metrics_summary=summary,
    )


@router.post("/models/rollback/check", response_model=RollbackCheckResponse)
async def check_model_rollback(payload: RollbackCheckRequest) -> RollbackCheckResponse:
    runs = repo.list_recent_backtest_runs(track=payload.track, limit=payload.max_recent_losses)
    usable = [r for r in runs if isinstance(r.get("metrics"), dict) and r["metrics"].get("status") == "completed"]
    if not usable:
        return RollbackCheckResponse(
            rollback_triggered=False,
            reason="no_recent_metrics",
            from_model=f"{payload.model_name}:{payload.model_version}",
            to_model=f"{payload.model_name}:{payload.model_version}",
            windows_failed=0,
            trigger_rule="none",
            metrics={},
        )

    hit_rate = float(np.mean([float(r["metrics"].get("hit_rate", 0.0)) for r in usable]))
    dd = float(np.mean([float(r["metrics"].get("max_drawdown", 1.0)) for r in usable]))
    pnl = float(np.mean([float(r["metrics"].get("pnl_after_cost", 0.0)) for r in usable]))

    fail_windows = 0
    for r in usable:
        m = r["metrics"]
        is_fail = (
            float(m.get("hit_rate", 0.0)) < payload.min_recent_hit_rate
            or float(m.get("max_drawdown", 1.0)) > payload.max_recent_drawdown
            or float(m.get("pnl_after_cost", 0.0)) < 0.0
        )
        if is_fail:
            fail_windows += 1
        else:
            break
    required_fails = int(max(1, min(payload.max_recent_losses, int(os.getenv("ROLLBACK_CONSECUTIVE_WINDOWS", "2")))))
    trigger = fail_windows >= required_fails
    metrics = {"hit_rate": round(hit_rate, 6), "max_drawdown": round(dd, 6), "pnl_after_cost": round(pnl, 6)}
    from_model = f"{payload.model_name}:{payload.model_version}"
    to_model = from_model
    reason = "stable"

    current = repo.get_active_model_state(payload.track)
    if trigger and current:
        prev_name = current.get("previous_model_name")
        prev_ver = current.get("previous_model_version")
        if prev_name and prev_ver:
            repo.upsert_active_model_state(
                track=payload.track,
                active_model_name=prev_name,
                active_model_version=prev_ver,
                previous_model_name=current.get("active_model_name"),
                previous_model_version=current.get("active_model_version"),
                status="rolled_back",
                metadata={"source": "rollback_check", "metrics": metrics},
            )
            to_model = f"{prev_name}:{prev_ver}"
            reason = "rollback_to_previous_model"
        else:
            reason = "rollback_triggered_but_no_previous_model"

        repo.save_risk_event(
            decision_id=f"rollback-{uuid.uuid4().hex[:12]}",
            severity="critical",
            code="model_kpi_degradation",
            message=reason,
            payload=metrics,
        )

    return RollbackCheckResponse(
        rollback_triggered=trigger,
        reason=reason,
        from_model=from_model,
        to_model=to_model,
        windows_failed=fail_windows,
        trigger_rule=f"consecutive_windows>={required_fails}",
        metrics=metrics,
    )


@router.get("/risk/limits", response_model=RiskLimitsResponse)
async def get_risk_limits() -> RiskLimitsResponse:
    limits = _risk_limits()
    return RiskLimitsResponse(updated_at=datetime.utcnow(), **limits)


@router.post("/risk/check", response_model=RiskCheckResponse)
async def risk_check(payload: RiskCheckRequest) -> RiskCheckResponse:
    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed=payload.proposed_positions,
        current=payload.current_positions,
        realized_drawdown=payload.realized_drawdown,
        max_sector_exposure_override=payload.max_sector_exposure_override,
        max_style_exposure_override=payload.max_style_exposure_override,
    )
    approved = len(violations) == 0
    hard_block = "drawdown_exceeded" in violations
    track = payload.proposed_positions[0].track if payload.proposed_positions else "liquid"
    strategy_id = payload.strategy_id or "global"
    runtime_limits = _risk_runtime_limits()
    if payload.daily_loss > runtime_limits["max_daily_loss"]:
        violations.append("daily_loss_exceeded")
        hard_block = True
    if payload.consecutive_losses > int(runtime_limits["max_consecutive_losses"]):
        violations.append("consecutive_loss_exceeded")
        hard_block = True
    if hard_block:
        hard_reasons = [v for v in violations if v in {"drawdown_exceeded", "daily_loss_exceeded", "consecutive_loss_exceeded"}]
        hard_reason = hard_reasons[0] if hard_reasons else "risk_hard_block"
        scope_strategy = strategy_id if "consecutive_loss_exceeded" in violations else "global"
        repo.upsert_kill_switch_state(
            track=track,
            strategy_id=scope_strategy,
            state="triggered",
            reason=hard_reason,
            duration_minutes=_risk_hard_block_minutes(),
            metadata={"violations": violations},
        )
    kill_block = _kill_switch_block_reason(track, strategy_id)
    kill_switch_state = "triggered" if kill_block else "armed"
    if kill_block:
        approved = False
        if kill_block not in violations:
            violations = [*violations, kill_block]
    elif violations:
        approved = False
    risk_budget_used = round(float(gross), 6)
    if hard_block:
        RISK_HARD_BLOCKS_TOTAL.labels(track=track if payload.proposed_positions else "unknown").inc()

    if violations:
        decision_id = f"risk-check-{uuid.uuid4().hex[:12]}"
        for code in violations:
            repo.save_risk_event(
                decision_id=decision_id,
                severity="warning",
                code=code,
                message="risk check violation",
                payload={"gross_exposure": gross, "turnover": turnover},
            )

    return RiskCheckResponse(
        approved=approved,
        violations=violations,
        adjusted_positions=adjusted,
        gross_exposure=round(gross, 6),
        expected_turnover=round(turnover, 6),
        hard_block=hard_block,
        kill_switch_state=kill_switch_state,
        risk_budget_used=risk_budget_used,
    )


@router.get("/risk/kill-switch", response_model=KillSwitchStateResponse)
async def get_kill_switch_state(
    track: str = Query(..., description="track"),
    strategy_id: str = Query("global", description="strategy id"),
) -> KillSwitchStateResponse:
    state = repo.get_kill_switch_state(track=track, strategy_id=strategy_id)
    if not state:
        return KillSwitchStateResponse(
            track=track,
            strategy_id=strategy_id,
            state="armed",
            reason="not_set",
            triggered=False,
            updated_at=datetime.utcnow(),
            expires_at=None,
            remaining_seconds=0,
            metadata={},
        )
    block = _kill_switch_block_reason(track, strategy_id)
    expires_at = state.get("expires_at")
    return KillSwitchStateResponse(
        track=track,
        strategy_id=strategy_id,
        state=str(state.get("state") or "armed"),
        reason=str(state.get("reason") or ""),
        triggered=bool(block),
        updated_at=state.get("updated_at") or datetime.utcnow(),
        expires_at=expires_at,
        remaining_seconds=_remaining_seconds(expires_at),
        metadata=state.get("metadata") or {},
    )


@router.get("/risk/opening-status", response_model=PositionOpeningStatusResponse)
async def get_opening_status(
    track: str = Query(..., description="track"),
    strategy_id: str = Query("global", description="strategy id"),
) -> PositionOpeningStatusResponse:
    state = repo.get_kill_switch_state(track=track, strategy_id=strategy_id) or {}
    block_reason = _kill_switch_block_reason(track, strategy_id)
    expires_at = state.get("expires_at")
    return PositionOpeningStatusResponse(
        track=track,  # type: ignore[arg-type]
        strategy_id=strategy_id,
        can_open_new_positions=not bool(block_reason),
        state="triggered" if block_reason else "armed",
        block_reason=str(block_reason or "none"),
        updated_at=state.get("updated_at") or datetime.utcnow(),
        expires_at=expires_at,
        remaining_seconds=_remaining_seconds(expires_at),
    )


@router.post("/risk/kill-switch/trigger", response_model=KillSwitchStateResponse)
async def trigger_kill_switch(payload: KillSwitchTriggerRequest) -> KillSwitchStateResponse:
    row = repo.upsert_kill_switch_state(
        track=payload.track,
        strategy_id=payload.strategy_id,
        state="triggered",
        reason=payload.reason,
        duration_minutes=payload.duration_minutes,
        metadata={"source": "manual", "duration_minutes": payload.duration_minutes},
    )
    return KillSwitchStateResponse(
        track=payload.track,
        strategy_id=payload.strategy_id,
        state="triggered",
        reason=payload.reason,
        triggered=True,
        updated_at=row.get("updated_at") or datetime.utcnow(),
        expires_at=row.get("expires_at"),
        remaining_seconds=_remaining_seconds(row.get("expires_at")),
        metadata=row.get("metadata") or {},
    )


@router.post("/risk/kill-switch/reset", response_model=KillSwitchStateResponse)
async def reset_kill_switch(payload: KillSwitchResetRequest) -> KillSwitchStateResponse:
    row = repo.upsert_kill_switch_state(
        track=payload.track,
        strategy_id=payload.strategy_id,
        state="armed",
        reason=payload.reason,
        duration_minutes=None,
        metadata={"source": "manual_reset"},
    )
    return KillSwitchStateResponse(
        track=payload.track,
        strategy_id=payload.strategy_id,
        state="armed",
        reason=payload.reason,
        triggered=False,
        updated_at=row.get("updated_at") or datetime.utcnow(),
        expires_at=row.get("expires_at"),
        remaining_seconds=0,
        metadata=row.get("metadata") or {},
    )


@router.post("/portfolio/rebalance", response_model=PortfolioRebalanceResponse)
async def portfolio_rebalance(payload: PortfolioRebalanceRequest) -> PortfolioRebalanceResponse:
    if not payload.signals:
        raise HTTPException(status_code=400, detail="signals cannot be empty")
    rebalance_track = payload.signals[0].track
    kill_block = _kill_switch_block_reason(rebalance_track, "global")
    if kill_block:
        raise HTTPException(status_code=423, detail=kill_block)

    raw: List[Tuple[str, str, float, str]] = []
    for sig in payload.signals:
        side = 0.0
        if sig.action == "buy":
            side = 1.0
        elif sig.action == "sell":
            side = -1.0
        strength = side * abs(sig.score) * max(0.1, sig.confidence)
        bucket = _strategy_bucket(sig.track, float(sig.score), float(sig.confidence), sig.target)
        raw.append((sig.target.upper(), sig.track, strength, bucket))

    gross_strength = sum(abs(x[2]) for x in raw)
    if gross_strength <= 0:
        raise HTTPException(status_code=400, detail="no actionable signals")

    bucket_caps = _bucket_limits()
    target_positions: List[RebalancePosition] = []
    pre_bucket_exposure: Dict[str, float] = {}
    for target, track, strength, bucket in raw:
        weight = (strength / gross_strength) * payload.risk_budget
        pre_bucket_exposure[bucket] = pre_bucket_exposure.get(bucket, 0.0) + abs(weight)
        target_positions.append(RebalancePosition(target=target, track=track, weight=weight, style_bucket=bucket))

    bucket_violations: List[str] = []
    for bucket, exposure in pre_bucket_exposure.items():
        cap = payload.risk_budget * bucket_caps.get(bucket, 1.0)
        if cap > 0 and exposure > cap:
            scale = cap / exposure
            bucket_violations.append(f"bucket_exposure_exceeded:{bucket}")
            target_positions = [
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=(p.weight * scale if p.style_bucket == bucket else p.weight),
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
                for p in target_positions
            ]

    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed=target_positions,
        current=payload.current_positions,
        realized_drawdown=0.0,
    )
    violations = bucket_violations + violations

    current_map = _position_map(payload.current_positions)
    new_map = _position_map(adjusted)
    universe = set(current_map.keys()) | set(new_map.keys())

    orders: List[Dict] = []
    decision_id = uuid.uuid4().hex
    for target, track in universe:
        delta = new_map.get((target, track), 0.0) - current_map.get((target, track), 0.0)
        if abs(delta) < 1e-6:
            continue
        side = "buy" if delta > 0 else "sell"
        px = repo.latest_price_snapshot(target)
        est_price = float(px["price"]) if px else None
        orders.append(
            {
                "target": target,
                "track": track,
                "side": side,
                "quantity": round(abs(delta) * payload.capital, 8),
                "est_price": est_price,
                "est_cost_bps": 8.0,
                "status": "simulated",
                "metadata": {"weight_delta": delta},
            }
        )

    repo.save_positions_snapshot(decision_id, [p.model_dump() for p in adjusted], reason="rebalance_v2")
    repo.save_orders_sim(decision_id, orders)
    for code in violations:
        repo.save_risk_event(
            decision_id=decision_id,
            severity="warning",
            code=code,
            message="rebalance risk violation",
            payload={"gross_exposure": gross, "turnover": turnover},
        )

    return PortfolioRebalanceResponse(
        decision_id=decision_id,
        target_positions=adjusted,
        expected_turnover=round(turnover, 6),
        orders=orders,
        risk_ok=len(violations) == 0,
        risk_violations=violations,
    )


@router.post("/execution/orders", response_model=SubmitExecutionOrdersResponse)
async def submit_execution_orders(payload: SubmitExecutionOrdersRequest) -> SubmitExecutionOrdersResponse:
    if not payload.orders:
        raise HTTPException(status_code=400, detail="orders cannot be empty")
    decision_id = uuid.uuid4().hex
    order_payloads = [
        {
            "target": o.target.upper(),
            "track": o.track,
            "side": o.side,
            "quantity": o.quantity,
            "est_price": o.est_price,
            "strategy_id": o.strategy_id,
            "metadata": o.metadata,
        }
        for o in payload.orders
    ]
    for order in payload.orders:
        block_reason = _kill_switch_block_reason(order.track, order.strategy_id)
        if block_reason:
            raise HTTPException(status_code=423, detail=block_reason)
    order_ids = repo.create_execution_orders(
        decision_id=decision_id,
        adapter=payload.adapter,
        venue=payload.venue,
        time_in_force=payload.time_in_force,
        max_slippage_bps=payload.max_slippage_bps,
        orders=order_payloads,
    )
    return SubmitExecutionOrdersResponse(
        decision_id=decision_id,
        adapter=payload.adapter,
        venue=payload.venue,
        accepted_orders=len(order_ids),
        order_ids=order_ids,
    )


@router.get("/execution/orders/{order_id}", response_model=ExecutionOrderStatusResponse)
async def get_execution_order(order_id: int) -> ExecutionOrderStatusResponse:
    row = repo.get_order_by_id(order_id)
    if not row:
        raise HTTPException(status_code=404, detail="order not found")
    return ExecutionOrderStatusResponse(
        order_id=int(row["id"]),
        decision_id=str(row["decision_id"]),
        target=str(row["target"]),
        side=str(row["side"]),
        quantity=float(row["quantity"]),
        status=str(row["status"]),
        track=str(row["track"]),
        venue=str(row.get("venue") or "coinbase"),
        adapter=str(row.get("adapter") or "paper"),
        created_at=row["created_at"],
        metadata=row.get("metadata") or {},
    )


@router.post("/execution/run", response_model=ExecuteOrdersResponse)
async def run_execution(payload: ExecuteOrdersRequest) -> ExecuteOrdersResponse:
    _t0 = time.perf_counter()
    orders = repo.fetch_orders_for_decision(payload.decision_id, limit=payload.max_orders)
    if not orders:
        raise HTTPException(status_code=404, detail="no orders for decision")
    for order in orders:
        order_track = str(order.get("track") or "liquid")
        strategy_id = str(order.get("strategy_id") or "default-liquid-v1")
        block_reason = _kill_switch_block_reason(order_track, strategy_id)
        if block_reason:
            raise HTTPException(status_code=423, detail=block_reason)
    scaled_orders: List[Dict[str, Any]] = []
    for order in orders:
        order_track = str(order.get("track") or "liquid")
        rollout = repo.get_model_rollout_state(order_track)
        pct = int(rollout.get("stage_pct") or 100) if rollout else 100
        scale = max(0.0, min(1.0, pct / 100.0))
        q = float(order.get("quantity") or 0.0) * scale
        od = dict(order)
        od["quantity"] = round(q, 8)
        od["metadata"] = {**(order.get("metadata") or {}), "rollout_stage_pct": pct}
        scaled_orders.append(od)
    precheck_violations = _execution_volatility_violations(scaled_orders)
    if precheck_violations:
        track = str(scaled_orders[0].get("track") or "liquid")
        repo.upsert_kill_switch_state(
            track=track,
            strategy_id="global",
            state="triggered",
            reason="abnormal_volatility_circuit_breaker",
            duration_minutes=_risk_hard_block_minutes(),
            metadata={"violations": precheck_violations},
        )
        for code in precheck_violations:
            repo.save_risk_event(
                decision_id=f"risk-precheck-{uuid.uuid4().hex[:12]}",
                severity="critical",
                code=code,
                message="execution volatility precheck blocked",
                payload={"decision_id": payload.decision_id},
            )
        raise HTTPException(status_code=423, detail=f"risk_blocked:{','.join(precheck_violations)}")
    runtime_limits = _risk_runtime_limits()
    strategy_pairs = sorted(
        {
            (str(o.get("track") or "liquid"), str(o.get("strategy_id") or "default-liquid-v1"))
            for o in scaled_orders
            if float(o.get("quantity") or 0.0) > 0
        }
    )
    for pair_track, pair_strategy in strategy_pairs:
        try:
            streak = int(repo.get_execution_consecutive_losses(track=pair_track, lookback_hours=24, limit=200, strategy_id=pair_strategy))
        except Exception:
            streak = 0
        if streak > int(runtime_limits["max_consecutive_losses"]):
            code = f"consecutive_loss_exceeded:{pair_strategy}"
            repo.upsert_kill_switch_state(
                track=pair_track,
                strategy_id=pair_strategy,
                state="triggered",
                reason="consecutive_loss_exceeded",
                duration_minutes=_risk_hard_block_minutes(),
                metadata={"violations": [code], "streak": streak},
            )
            repo.save_risk_event(
                decision_id=f"risk-precheck-{uuid.uuid4().hex[:12]}",
                severity="critical",
                code=code,
                message="execution consecutive loss precheck blocked",
                payload={"decision_id": payload.decision_id, "streak": streak},
            )
            raise HTTPException(status_code=423, detail=f"risk_blocked:{code}")
    # enforce risk check before execution; execution is blocked if risk fails.
    inferred_positions = []
    for o in scaled_orders:
        qty = float(o.get("quantity") or 0.0)
        if qty <= 0:
            continue
        side = str(o.get("side") or "buy").lower()
        w = qty if side == "buy" else -qty
        inferred_positions.append(RebalancePosition(target=str(o.get("target") or "").upper(), track=str(o.get("track") or "liquid"), weight=w))
    track = str(scaled_orders[0].get("track") or "liquid")
    realized_daily_loss = _infer_daily_loss_ratio(track=track)
    risk_resp = await risk_check(
        RiskCheckRequest(
            proposed_positions=inferred_positions,
            current_positions=[],
            realized_drawdown=0.0,
            daily_loss=realized_daily_loss,
            consecutive_losses=0,
            strategy_id=str(scaled_orders[0].get("strategy_id") or "global"),
        )
    )
    if not risk_resp.approved:
        raise HTTPException(status_code=423, detail=f"risk_blocked:{','.join(risk_resp.violations)}")
    results = exec_engine.run(
        payload.adapter,
        scaled_orders,
        context={
            "time_in_force": payload.time_in_force,
            "max_slippage_bps": payload.max_slippage_bps,
            "venue": payload.venue,
            "limit_timeout_sec": payload.limit_timeout_sec,
            "max_retries": payload.max_retries,
            "fee_bps": payload.fee_bps,
        },
    )
    filled = 0
    rejected = 0
    merged = []
    for order, res in zip(scaled_orders, results):
        normalized_res = _normalize_execution_payload(res)
        status = str(normalized_res.get("status") or "rejected")
        EXECUTION_ORDERS_TOTAL.labels(adapter=payload.adapter, status=status).inc()
        if status in {"filled", "partially_filled"}:
            filled += 1
        elif status == "rejected":
            rejected += 1
        repo.update_order_execution(order["id"], status=status, metadata={"execution": normalized_res})
        merged.append({**order, "execution": normalized_res})

    resp = ExecuteOrdersResponse(
        decision_id=payload.decision_id,
        adapter=payload.adapter,
        total=len(orders),
        filled=filled,
        rejected=rejected,
        orders=merged,
    )
    EXECUTION_LATENCY_SECONDS.labels(adapter=payload.adapter).observe(max(0.0, time.perf_counter() - _t0))
    return resp


@router.get("/execution/audit/{decision_id}", response_model=TradeAuditResponse)
async def get_execution_audit(decision_id: str) -> TradeAuditResponse:
    chain = repo.get_trade_audit_chain(decision_id)
    if not chain["orders"] and not chain["signals"] and not chain["positions"]:
        raise HTTPException(status_code=404, detail="decision not found")
    track = "liquid"
    for order in chain["orders"]:
        track = str(order.get("track") or track)
        break
    pnl = repo.build_pnl_attribution(track=track, lookback_hours=24)
    return TradeAuditResponse(
        decision_id=decision_id,
        signals=chain["signals"],
        orders=chain["orders"],
        positions=chain["positions"],
        pnl=pnl,
        generated_at=datetime.utcnow(),
    )


@router.post("/models/drift/evaluate", response_model=DriftEvaluateResponse)
async def evaluate_model_drift(payload: DriftEvaluateRequest) -> DriftEvaluateResponse:
    current_scores = repo.get_recent_prediction_scores(payload.track, payload.lookback_hours)
    reference_scores = repo.get_prediction_scores_window(
        payload.track,
        offset_hours=payload.lookback_hours,
        window_hours=payload.reference_hours,
    )
    current_features = repo.get_recent_feature_payloads(payload.track, payload.lookback_hours, limit=500)
    reference_features = repo.get_feature_payloads_window(
        payload.track,
        offset_hours=payload.lookback_hours,
        window_hours=payload.reference_hours,
        limit=500,
    )
    current_hit = repo.get_recent_backtest_metric(payload.track, "hit_rate", payload.lookback_hours)
    ref_hit = repo.get_backtest_metric_window(payload.track, "hit_rate", payload.lookback_hours, payload.reference_hours)
    slippage = repo.get_execution_slippage_samples(payload.track, payload.lookback_hours)
    reject_rate_current = repo.get_execution_reject_rate(payload.track, payload.lookback_hours)
    reject_rate_ref = repo.get_execution_reject_rate_window(payload.track, payload.lookback_hours, payload.reference_hours)

    psi_val = _psi(reference_scores, current_scores)
    ks_val = _ks_statistic(reference_scores, current_scores)
    feature_psi, feature_mean_shift = _feature_drift_score(reference_features, current_features)
    label_stability_delta = abs(float(np.mean(current_hit or [0.0])) - float(np.mean(ref_hit or [0.0])))
    reject_rate_delta = abs(reject_rate_current - reject_rate_ref)
    EXECUTION_REJECT_RATE.labels(track=payload.track).set(reject_rate_current)
    slippage_abs_mean = float(np.mean(np.abs(slippage))) if slippage else 0.0
    drift = (
        psi_val >= payload.psi_threshold
        or ks_val >= payload.ks_threshold
        or feature_psi >= payload.psi_threshold
        or label_stability_delta >= 0.15
        or reject_rate_delta >= 0.05
    )
    action = "rollback_check" if drift else ("warn" if psi_val > payload.psi_threshold * 0.7 else "keep")
    MODEL_DRIFT_EVENTS_TOTAL.labels(track=payload.track, action=action).inc()
    reason = "distribution_shift_detected" if drift else "stable"

    return DriftEvaluateResponse(
        track=payload.track,
        drift_detected=drift,
        action=action,
        metrics={
            "psi": round(psi_val, 6),
            "ks": round(ks_val, 6),
            "feature_psi": round(feature_psi, 6),
            "feature_mean_shift": round(feature_mean_shift, 6),
            "label_stability_delta": round(label_stability_delta, 6),
            "execution_reject_rate_delta": round(reject_rate_delta, 6),
            "execution_slippage_abs_mean": round(slippage_abs_mean, 8),
            "current_samples": float(len(current_scores)),
            "reference_samples": float(len(reference_scores)),
        },
        reason=reason,
        evaluated_at=datetime.utcnow(),
    )


@router.post("/models/gate/auto-evaluate", response_model=AutoGateEvaluateResponse)
async def auto_evaluate_model_gate(payload: AutoGateEvaluateRequest) -> AutoGateEvaluateResponse:
    default_name, default_version = _default_model_by_track(payload.track)
    model_name = payload.model_name or default_name
    model_version = payload.model_version or default_version
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=payload.min_ic,
        min_pnl_after_cost=payload.min_pnl_after_cost,
        max_drawdown=payload.max_drawdown,
        windows=payload.windows,
    )
    repo.promote_model(
        track=payload.track,
        model_name=model_name,
        model_version=model_version,
        passed=passed,
        metrics={**summary, "windows_checked": checked, "source": "auto_gate"},
        gate_reason=reason,
    )
    promoted = False
    if passed and payload.auto_promote:
        current = repo.get_active_model_state(payload.track)
        repo.upsert_active_model_state(
            track=payload.track,
            active_model_name=model_name,
            active_model_version=model_version,
            previous_model_name=current.get("active_model_name") if current else None,
            previous_model_version=current.get("active_model_version") if current else None,
            status="active",
            metadata={"source": "auto_gate", "summary": summary},
        )
        promoted = True
    return AutoGateEvaluateResponse(
        passed=passed,
        promoted=promoted,
        track=payload.track,
        model_name=model_name,
        model_version=model_version,
        reason=reason,
        windows_checked=checked,
        metrics_summary=summary,
    )


@router.post("/models/rollout/advance", response_model=RolloutAdvanceResponse)
async def advance_model_rollout(payload: RolloutAdvanceRequest) -> RolloutAdvanceResponse:
    runs = repo.list_recent_backtest_runs(track=payload.track, limit=payload.windows)
    usable = [r for r in runs if isinstance(r.get("metrics"), dict) and r["metrics"].get("status") == "completed"]
    if len(usable) < payload.windows:
        return RolloutAdvanceResponse(
            track=payload.track,
            model_name=payload.model_name,
            model_version=payload.model_version,
            current_stage_pct=payload.current_stage_pct,
            next_stage_pct=payload.next_stage_pct,
            promoted=False,
            reason="insufficient_windows",
            hard_limits={
                "min_hit_rate": payload.min_hit_rate,
                "min_pnl_after_cost": payload.min_pnl_after_cost,
                "max_drawdown": payload.max_drawdown,
            },
            metrics={"hit_rate": 0.0, "pnl_after_cost": 0.0, "max_drawdown": 1.0},
        )
    hit_rate = float(np.mean([float(r["metrics"].get("hit_rate", 0.0)) for r in usable[: payload.windows]]))
    pnl = float(np.mean([float(r["metrics"].get("pnl_after_cost", 0.0)) for r in usable[: payload.windows]]))
    dd = float(np.mean([float(r["metrics"].get("max_drawdown", 1.0)) for r in usable[: payload.windows]]))
    promoted = hit_rate >= payload.min_hit_rate and pnl >= payload.min_pnl_after_cost and dd <= payload.max_drawdown
    ladder = {10: 30, 30: 100, 100: 100}
    expected_next = ladder.get(payload.current_stage_pct, payload.current_stage_pct)
    requested_next = payload.next_stage_pct
    next_stage = expected_next if promoted else payload.current_stage_pct
    if promoted and requested_next != expected_next:
        promoted = False
        next_stage = payload.current_stage_pct
    status = "active" if next_stage >= 100 else ("canary" if next_stage > 10 else "shadow")
    repo.upsert_model_rollout_state(
        track=payload.track,
        model_name=payload.model_name,
        model_version=payload.model_version,
        stage_pct=next_stage,
        status=status,
        hard_limits={
            "min_hit_rate": payload.min_hit_rate,
            "min_pnl_after_cost": payload.min_pnl_after_cost,
            "max_drawdown": payload.max_drawdown,
        },
        metrics={"hit_rate": round(hit_rate, 6), "pnl_after_cost": round(pnl, 6), "max_drawdown": round(dd, 6)},
    )
    return RolloutAdvanceResponse(
        track=payload.track,
        model_name=payload.model_name,
        model_version=payload.model_version,
        current_stage_pct=payload.current_stage_pct,
        next_stage_pct=next_stage,
        promoted=promoted,
        reason="promoted" if promoted else ("invalid_rollout_step" if requested_next != expected_next else "hard_limit_failed"),
        hard_limits={
            "min_hit_rate": payload.min_hit_rate,
            "min_pnl_after_cost": payload.min_pnl_after_cost,
            "max_drawdown": payload.max_drawdown,
        },
        metrics={"hit_rate": round(hit_rate, 6), "pnl_after_cost": round(pnl, 6), "max_drawdown": round(dd, 6)},
    )


@router.get("/models/rollout/state", response_model=RolloutStateResponse)
async def get_rollout_state(track: str = Query(..., description="track")) -> RolloutStateResponse:
    state = repo.get_model_rollout_state(track)
    if not state:
        model_name, model_version = _default_model_by_track(track)
        return RolloutStateResponse(
            track=track,  # type: ignore[arg-type]
            model_name=model_name,
            model_version=model_version,
            stage_pct=10,
            status="shadow",
            hard_limits={},
            metrics={},
            updated_at=datetime.utcnow(),
        )
    return RolloutStateResponse(
        track=track,  # type: ignore[arg-type]
        model_name=str(state.get("model_name") or _default_model_by_track(track)[0]),
        model_version=str(state.get("model_version") or _default_model_by_track(track)[1]),
        stage_pct=int(state.get("stage_pct") or 10),
        status=str(state.get("status") or "shadow"),
        hard_limits=state.get("hard_limits") or {},
        metrics=state.get("metrics") or {},
        updated_at=state.get("updated_at") or datetime.utcnow(),
    )


@router.post("/models/audit/log")
async def create_scheduler_audit_log(payload: SchedulerAuditLogRequest):
    repo.save_scheduler_audit_log(
        track=payload.track,
        action=payload.action,
        window=payload.window,
        thresholds=payload.thresholds,
        decision=payload.decision,
    )
    return {"status": "ok"}


@router.get("/metrics/pnl-attribution", response_model=PnLAttributionResponse)
async def get_pnl_attribution(track: str = "liquid", lookback_hours: int = 24 * 7) -> PnLAttributionResponse:
    if track not in {"liquid", "vc"}:
        raise HTTPException(status_code=400, detail="track must be liquid or vc")
    attribution = repo.build_pnl_attribution(track=track, lookback_hours=lookback_hours)
    return PnLAttributionResponse(
        track=track,  # type: ignore[arg-type]
        lookback_hours=lookback_hours,
        totals=attribution["totals"],
        by_target=attribution["by_target"],
        generated_at=datetime.utcnow(),
    )


@router.post("/tasks/pnl-attribution", response_model=AsyncTaskSubmitResponse)
async def submit_pnl_attribution_task(payload: PnLAttributionRequest) -> AsyncTaskSubmitResponse:
    task_id = enqueue_task("pnl_attribution", payload.model_dump())
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type="pnl_attribution",
        status="queued",
        created_at=datetime.utcnow(),
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    item = get_task(task_id)
    if not item:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskStatusResponse(
        task_id=str(item.get("task_id") or task_id),
        task_type=str(item.get("task_type") or "unknown"),
        status=str(item.get("status") or "queued"),  # type: ignore[arg-type]
        created_at=item.get("created_at") or datetime.utcnow(),
        updated_at=item.get("updated_at"),
        started_at=item.get("started_at"),
        finished_at=item.get("finished_at"),
        result=item.get("result"),
        error=item.get("error"),
    )


@router.post("/alerts/notify")
async def ingest_alert_notification(payload: Dict[str, Any]):
    status = str(payload.get("status") or "firing")
    alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
    for a in alerts:
        if not isinstance(a, dict):
            continue
        labels = a.get("labels") if isinstance(a.get("labels"), dict) else {}
        annotations = a.get("annotations") if isinstance(a.get("annotations"), dict) else {}
        code = str(labels.get("alertname") or "alert")
        severity = "critical" if str(labels.get("severity", "")).upper() == "P1" else "warning"
        repo.save_risk_event(
            decision_id=f"alert-{uuid.uuid4().hex[:16]}",
            severity=severity,
            code=f"alertmanager:{code}",
            message=str(annotations.get("summary") or code),
            payload={"status": status, "labels": labels, "annotations": annotations},
        )
    return {"status": "ok", "alerts_received": len(alerts)}


@router.post("/data-quality/sample")
async def sample_data_quality(payload: DataQualitySampleRequest):
    return {
        "items": repo.sample_data_quality(limit=payload.limit, min_quality_score=payload.min_quality_score),
        "limit": payload.limit,
    }


@router.get("/data-quality/stats", response_model=DataQualityStatsResponse)
async def data_quality_stats(lookback_days: int = 7) -> DataQualityStatsResponse:
    lookback_days = max(1, min(90, int(lookback_days)))
    stats = repo.get_data_quality_stats(lookback_days=lookback_days)
    return DataQualityStatsResponse(
        lookback_days=lookback_days,
        totals=stats.get("totals", {}),
        by_source=stats.get("by_source", []),
        generated_at=datetime.utcnow(),
    )


@router.get("/data-quality/consistency", response_model=DataQualityConsistencyResponse)
async def data_quality_consistency(lookback_days: int = 30) -> DataQualityConsistencyResponse:
    lookback_days = max(1, min(365, int(lookback_days)))
    stats = repo.get_data_quality_consistency(lookback_days=lookback_days)
    return DataQualityConsistencyResponse(
        lookback_days=lookback_days,
        total_review_logs=int(stats.get("total_review_logs", 0)),
        multi_review_events=int(stats.get("multi_review_events", 0)),
        pairwise_agreement=float(stats.get("pairwise_agreement", 0.0)),
        reviewer_pairs=stats.get("reviewer_pairs", []),
        generated_at=datetime.utcnow(),
    )


@router.post("/data-quality/lineage/check", response_model=LineageConsistencyResponse)
async def check_data_lineage_consistency(payload: LineageConsistencyRequest) -> LineageConsistencyResponse:
    res = repo.check_feature_lineage_consistency(
        track=payload.track,
        lineage_id=payload.lineage_id,
        target=payload.target,
        data_version=payload.data_version,
        strict=payload.strict,
        max_mismatch_keys=payload.max_mismatch_keys,
        tolerance=payload.tolerance,
    )
    return LineageConsistencyResponse(
        passed=bool(res.get("passed", False)),
        track=payload.track,
        target=payload.target,
        lineage_id=payload.lineage_id,
        data_version=payload.data_version,
        compared_snapshots=int(res.get("compared_snapshots", 0)),
        max_abs_diff=float(res.get("max_abs_diff", 0.0)),
        mean_abs_diff=float(res.get("mean_abs_diff", 0.0)),
        mismatch_keys=[str(x) for x in res.get("mismatch_keys", [])[: payload.max_mismatch_keys]],
        reason=str(res.get("reason", "unknown")),
    )


@router.post("/data-quality/audit")
async def update_data_quality_audit(payload: DataQualityAuditUpdate):
    repo.update_data_quality_audit(
        audit_id=payload.audit_id,
        reviewer=payload.reviewer,
        verdict=payload.verdict,
        note=payload.note,
    )
    return {"status": "ok", "audit_id": payload.audit_id}
