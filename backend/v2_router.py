from __future__ import annotations

import os
import json
import time
import uuid
import logging
import sys
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Query
try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    lgb = None  # type: ignore[assignment]
    HAS_LGB = False

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
    ParityCheckRequest,
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
from execution_policy import build_execution_trace
from portfolio_allocator import AllocatorSignal, allocate_targets
from liquid_model_registry import get_candidate_model as get_liquid_candidate_model
from liquid_model_registry import load_registry as load_liquid_registry
from liquid_model_registry import promote_candidate as promote_liquid_candidate
from liquid_model_registry import rollback_active as rollback_liquid_active
from liquid_model_registry import upsert_active as upsert_liquid_active
from liquid_model_registry import upsert_candidate as upsert_liquid_candidate
from execution_api import (
    infer_execution_risk_positions as _executionapi_infer_execution_risk_positions,
    normalize_execution_payload as _executionapi_normalize_execution_payload,
    normalize_reject_reason as _executionapi_normalize_reject_reason,
    order_to_risk_weight as _executionapi_order_to_risk_weight,
)
from feature_signal import feature_signal_score
from liquid_model_service import LiquidModelService
from metrics import (
    BACKTEST_FAILED_RUNS_TOTAL,
    BITGET_UNCLASSIFIED_ERRORS_TOTAL,
    DATA_FRESHNESS_SECONDS,
    EXECUTION_LATENCY_SECONDS,
    EXECUTION_ORDERS_TOTAL,
    EXECUTION_REJECTS_TOTAL,
    EXECUTION_REJECT_RATE,
    METRIC_GATE_STATUS,
    MODEL_COVERAGE_RATIO,
    MODEL_DRIFT_EVENTS_TOTAL,
    RISK_WEIGHT_CONVERSION_ERRORS_TOTAL,
    RISK_HARD_BLOCKS_TOTAL,
    SIGNAL_LATENCY_SECONDS,
    SIGNAL_PREDICT_DIVERGENCE,
    STRICT_ASOF_FAIL_COUNT,
    INGEST_EVENTS_TOTAL,
)
from drift_ops import feature_drift_score as _driftops_feature_drift_score
from drift_ops import ks_statistic as _driftops_ks_statistic
from drift_ops import psi as _driftops_psi
from risk_control import drawdown_thresholds as _riskcontrol_drawdown_thresholds
from strategy_backtest import evaluate_gate as _strategybacktest_evaluate_gate
from strategy_backtest import score_source_filter as _strategybacktest_score_source_filter
from strategy_signal import strategy_bucket as _strategysignal_strategy_bucket
from v2_repository import V2Repository
from task_queue import enqueue_task, get_task

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "feature-store-v2.1")
FEATURE_PAYLOAD_SCHEMA_VERSION = os.getenv("FEATURE_PAYLOAD_SCHEMA_VERSION", "main")
FEATURE_PAYLOAD_SCHEMA_COMPAT = {FEATURE_PAYLOAD_SCHEMA_VERSION, "main", "v2.3", "v2.2"}
DATA_VERSION = os.getenv("DATA_VERSION", "v1")

router = APIRouter(prefix="/api/v2", tags=["v2"])


class _UnavailableRepo:
    def __init__(self, reason: str):
        self._reason = reason

    def __getattr__(self, name: str):
        raise RuntimeError(f"V2Repository unavailable: {self._reason}")


def _init_repo() -> Any:
    if "pytest" in sys.modules and os.getenv("PYTEST_INIT_DB", "0").strip() not in {"1", "true", "yes"}:
        return _UnavailableRepo("skipped repo init during pytest import")
    try:
        return V2Repository(DATABASE_URL)
    except Exception as exc:
        logger.warning("V2 repository init skipped at import time: %s", exc)
        return _UnavailableRepo(str(exc))


repo = _init_repo()
exec_engine = ExecutionEngine()
DEFAULT_LIQUID_SYMBOLS = "BTC,ETH,SOL"
_TABULAR_MODEL_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
LIQUID_FEATURE_KEYS: List[str] = [
    "ret_1",
    "ret_3",
    "ret_12",
    "ret_48",
    "ret_96",
    "ret_288",
    "vol_3",
    "vol_12",
    "vol_48",
    "vol_96",
    "vol_288",
    "ret_accel_1_3",
    "ret_accel_3_12",
    "ret_accel_12_48",
    "vol_term_3_12",
    "vol_term_12_48",
    "vol_term_48_288",
    "log_volume",
    "vol_z",
    "volume_impact",
    "orderbook_imbalance",
    "funding_rate",
    "onchain_norm",
    "event_decay",
    "orderbook_missing_flag",
    "funding_missing_flag",
    "onchain_missing_flag",
    "source_tier_weight",
    "source_confidence",
    "social_post_sentiment",
    "social_comment_sentiment",
    "social_engagement_norm",
    "social_influence_norm",
    "social_event_ratio",
    "social_buzz",
    "event_velocity_1h",
    "event_velocity_6h",
    "event_disagreement",
    "source_diversity",
    "cross_source_consensus",
    "comment_skew",
    "event_lag_bucket_0_1h",
    "event_lag_bucket_1_6h",
    "event_lag_bucket_6_24h",
    "event_density",
    "sentiment_abs",
    "social_comment_rate",
    "event_importance_mean",
    "novelty_confidence_blend",
]
_ROOT = Path(__file__).resolve().parents[1]
_INFER_DIR = _ROOT / "inference"
if str(_INFER_DIR) not in sys.path:
    sys.path.append(str(_INFER_DIR))
try:
    from liquid_feature_contract import ONLINE_LIQUID_FEATURE_KEYS as _CONTRACT_KEYS  # type: ignore
    from liquid_feature_contract import LIQUID_FEATURE_SCHEMA_VERSION as _CONTRACT_SCHEMA  # type: ignore
    from model_router import ModelRouter  # type: ignore

    if isinstance(_CONTRACT_KEYS, list) and _CONTRACT_KEYS:
        LIQUID_FEATURE_KEYS = list(_CONTRACT_KEYS)
    FEATURE_PAYLOAD_SCHEMA_VERSION = str(_CONTRACT_SCHEMA or FEATURE_PAYLOAD_SCHEMA_VERSION)
    FEATURE_PAYLOAD_SCHEMA_COMPAT = {
        FEATURE_PAYLOAD_SCHEMA_VERSION,
        "main",
        "v2.3",
        "v2.2",
    }
except Exception:
    ModelRouter = None  # type: ignore[assignment]
    pass

_MODEL_ROUTER: Optional[Any] = None
_LIQUID_MODEL_SERVICE: Optional[LiquidModelService] = None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso_z() -> str:
    return _utcnow().isoformat().replace("+00:00", "Z")


def _default_model_dir() -> Path:
    env_path = str(os.getenv("MODEL_DIR", "")).strip()
    if env_path:
        return Path(env_path)
    local_models = Path(__file__).resolve().parent / "models"
    if local_models.exists():
        return local_models
    return Path("/opt/monitoring-system/models")


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + (2.718281828 ** (-x)))


def _default_liquid_targets() -> List[str]:
    return [s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", DEFAULT_LIQUID_SYMBOLS).split(",") if s.strip()]


def _get_model_router() -> Any:
    global _MODEL_ROUTER
    if _MODEL_ROUTER is not None:
        return _MODEL_ROUTER
    if ModelRouter is None:
        raise RuntimeError("model_router_unavailable")
    _MODEL_ROUTER = ModelRouter()
    return _MODEL_ROUTER


def _active_liquid_model() -> Tuple[str, str]:
    return _get_liquid_model_service()._active_liquid_model()  # type: ignore[attr-defined]


def _get_liquid_model_service() -> LiquidModelService:
    global _LIQUID_MODEL_SERVICE
    if _LIQUID_MODEL_SERVICE is not None and _LIQUID_MODEL_SERVICE.repo is repo:
        return _LIQUID_MODEL_SERVICE
    default_name, default_version = _default_model_by_track("liquid")
    _LIQUID_MODEL_SERVICE = LiquidModelService(
        repo=repo,
        feature_keys=LIQUID_FEATURE_KEYS,
        feature_version=FEATURE_VERSION,
        data_version=DATA_VERSION,
        default_model_name=default_name,
        default_model_version=default_version,
    )
    return _LIQUID_MODEL_SERVICE


def _latest_liquid_feature_payload(symbol: str, *, max_lookback_days: int = 30) -> Dict[str, float]:
    payload = _get_liquid_model_service().latest_feature_payload(symbol, max_lookback_days=max_lookback_days)
    if payload:
        return payload
    raise HTTPException(status_code=424, detail=f"feature_payload_unavailable:{str(symbol or '').upper()}")


def _build_liquid_model_prediction(symbol: str, horizon: str) -> Dict[str, object]:
    result = _get_liquid_model_service().predict_with_context(symbol=symbol, horizon=horizon)
    target = str(result.get("target") or "").upper()
    # Keep data freshness metric sourced from price snapshot timestamp if present.
    ts_raw = (repo.latest_price_snapshot(target) or {}).get("timestamp")
    if isinstance(ts_raw, datetime):
        ts_utc = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
        freshness = max(0.0, (_utcnow() - ts_utc.astimezone(timezone.utc)).total_seconds())
        DATA_FRESHNESS_SECONDS.labels(target=target).set(freshness)
    return result


def _risk_limits() -> Dict[str, float]:
    return {
        "max_single_weight": float(os.getenv("RISK_MAX_SINGLE_WEIGHT", "0.2")),
        "max_gross_exposure": float(os.getenv("RISK_MAX_GROSS_EXPOSURE", "1.0")),
        "max_turnover_per_rebalance": float(os.getenv("RISK_MAX_TURNOVER", "0.35")),
        "max_realized_drawdown": float(os.getenv("RISK_MAX_DRAWDOWN", "0.2")),
        "max_sector_exposure": float(os.getenv("RISK_MAX_SECTOR_EXPOSURE", "0.45")),
        "max_style_exposure": float(os.getenv("RISK_MAX_STYLE_EXPOSURE", "0.55")),
    }


def _drawdown_thresholds(dd_limit: float) -> Tuple[float, float]:
    return _riskcontrol_drawdown_thresholds(
        dd_limit=float(dd_limit),
        warn_default=float(os.getenv("RISK_DRAWDOWN_WARN_THRESHOLD", "0.08")),
        near_default=float(os.getenv("RISK_DRAWDOWN_NEAR_LIMIT", "0.10")),
    )


def _risk_runtime_limits() -> Dict[str, float]:
    return {
        "max_daily_loss": float(os.getenv("RISK_MAX_DAILY_LOSS", "0.03")),
        "max_consecutive_losses": float(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "5")),
        "single_trade_stop_loss_pct": float(os.getenv("RISK_SINGLE_STOP_LOSS_PCT", "0.018")),
        "single_trade_take_profit_pct": float(os.getenv("RISK_SINGLE_TAKE_PROFIT_PCT", "0.036")),
        "intraday_drawdown_halt_pct": float(os.getenv("RISK_INTRADAY_DRAWDOWN_HALT_PCT", "0.05")),
    }


def _risk_hard_block_minutes() -> int:
    return int(max(1, min(60 * 24, int(os.getenv("RISK_HARD_BLOCK_MINUTES", "15")))))


def _infer_daily_loss_ratio(track: str) -> float:
    try:
        return float(repo.get_execution_daily_loss_ratio(track=track, lookback_hours=24))
    except Exception:
        try:
            totals = repo.build_pnl_attribution(track=track, lookback_hours=24).get("totals", {})
        except Exception:
            return 0.0
        net_pnl = float(totals.get("net_pnl", 0.0) or 0.0)
        gross_notional = abs(float(totals.get("gross_notional_signed", 0.0) or 0.0))
        if net_pnl >= 0 or gross_notional <= 1e-9:
            return 0.0
        return float(max(0.0, min(5.0, (-net_pnl) / gross_notional)))


def _infer_latest_trade_edge_ratio(track: str, strategy_id: Optional[str] = None) -> float:
    rows = repo.get_execution_edge_pnls(track=track, lookback_hours=24, limit=1, strategy_id=strategy_id)
    if not rows:
        return 0.0
    edge = float(rows[0].get("edge_pnl") or 0.0)
    notional = float(rows[0].get("notional") or 0.0)
    if notional <= 1e-9:
        return 0.0
    return float(max(-5.0, min(5.0, edge / notional)))


def _infer_intraday_drawdown_ratio(track: str, strategy_id: Optional[str] = None) -> float:
    rows = repo.get_execution_edge_pnls(track=track, lookback_hours=24, limit=5000, strategy_id=strategy_id)
    if not rows:
        return 0.0

    def _ts(row: Dict[str, Any]) -> float:
        dt = row.get("created_at")
        if isinstance(dt, datetime):
            return float(dt.timestamp())
        return 0.0

    ordered = sorted(rows, key=_ts)
    cumulative = 0.0
    peak = 0.0
    max_drawdown_amt = 0.0
    gross_notional = 0.0
    for row in ordered:
        edge = float(row.get("edge_pnl") or 0.0)
        notional = max(0.0, float(row.get("notional") or 0.0))
        cumulative += edge
        gross_notional += notional
        peak = max(peak, cumulative)
        max_drawdown_amt = max(max_drawdown_amt, peak - cumulative)
    if gross_notional <= 1e-9:
        return 0.0
    return float(max(0.0, min(5.0, max_drawdown_amt / gross_notional)))


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
            hour = _utcnow().hour
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


def _market_adaptation_limits() -> Dict[str, float]:
    return {
        "min_latest_volume": float(os.getenv("RISK_MIN_LATEST_VOLUME", "1.0")),
        "max_price_staleness_sec": float(os.getenv("RISK_MAX_PRICE_STALENESS_SEC", "900")),
        "min_order_notional_usd": float(os.getenv("RISK_MIN_ORDER_NOTIONAL_USD", "5.0")),
    }


def _order_market_fit_violations(orders: List[Dict[str, Any]], market_type: str) -> List[str]:
    limits = _market_adaptation_limits()
    out: List[str] = []
    seen: set[str] = set()
    for order in orders:
        qty = float(order.get("quantity") or 0.0)
        if qty <= 0.0:
            continue
        target = str(order.get("target") or "").upper()
        if not target:
            continue
        if str(market_type).strip().lower() == "perp_usdt" and not target.endswith("USDT"):
            code = f"market_type_mismatch:{target}:perp_usdt_requires_usdt_pair"
            if code not in seen:
                seen.add(code)
                out.append(code)
            continue
        if str(market_type).strip().lower() == "spot" and target.endswith("PERP"):
            code = f"market_type_mismatch:{target}:spot_symbol_invalid"
            if code not in seen:
                seen.add(code)
                out.append(code)
            continue
        try:
            px = repo.latest_price_snapshot(target)
        except Exception:
            px = None
        if not isinstance(px, dict):
            code = f"market_data_missing:{target}"
            if code not in seen:
                seen.add(code)
                out.append(code)
            continue
        price = float(px.get("price") or 0.0)
        latest_volume = float(px.get("volume") or 0.0)
        ts = px.get("timestamp")
        if isinstance(ts, datetime):
            ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            age_sec = max(0.0, (_utcnow() - ts_utc.astimezone(timezone.utc)).total_seconds())
            if age_sec > float(limits["max_price_staleness_sec"]):
                code = f"stale_price_data:{target}"
                if code not in seen:
                    seen.add(code)
                    out.append(code)
        if latest_volume < float(limits["min_latest_volume"]):
            code = f"insufficient_liquidity:{target}"
            if code not in seen:
                seen.add(code)
                out.append(code)
        if price <= 0.0 or (price * qty) < float(limits["min_order_notional_usd"]):
            code = f"order_notional_too_small:{target}"
            if code not in seen:
                seen.add(code)
                out.append(code)
    return out


def _order_risk_price(order: Dict[str, Any], cache: Dict[str, float]) -> float:
    est_price = float(order.get("est_price") or 0.0)
    if est_price > 0.0:
        return est_price
    target = str(order.get("target") or "").upper()
    if not target:
        return 0.0
    if target in cache:
        return cache[target]
    try:
        px = repo.latest_price_snapshot(target)
    except Exception:
        px = None
    val = float(px.get("price") or 0.0) if isinstance(px, dict) else 0.0
    cache[target] = val
    return val


def _infer_execution_risk_positions(orders: List[Dict[str, Any]], risk_equity_usd: Optional[float]) -> List[RebalancePosition]:
    return _executionapi_infer_execution_risk_positions(
        orders=orders,
        risk_equity_usd=risk_equity_usd,
        order_risk_price_fn=_order_risk_price,
        rebalance_position_cls=RebalancePosition,
        conversion_error_hook=lambda track: RISK_WEIGHT_CONVERSION_ERRORS_TOTAL.labels(track=track).inc(),
    )


def _order_to_risk_weight(order: Dict[str, Any], risk_equity_usd: Optional[float], signed_notional: Optional[float] = None) -> Optional[float]:
    return _executionapi_order_to_risk_weight(
        order=order,
        risk_equity_usd=risk_equity_usd,
        signed_notional=signed_notional,
        conversion_error_hook=lambda track: RISK_WEIGHT_CONVERSION_ERRORS_TOTAL.labels(track=track).inc(),
    )


def _normalize_execution_payload(execution: Dict[str, Any]) -> Dict[str, Any]:
    return _executionapi_normalize_execution_payload(execution=execution, now_iso=_utcnow_iso_z())


def _normalize_reject_reason(reason: str) -> str:
    return _executionapi_normalize_reject_reason(reason)


def _position_sizing_settings() -> Dict[str, float]:
    return {
        "entry_z": float(os.getenv("SIGNAL_ENTRY_Z_MIN", "0.02")),
        "exit_z": float(os.getenv("SIGNAL_EXIT_Z_MIN", "0.008")),
        "max_weight_base": float(os.getenv("POSITION_MAX_WEIGHT_BASE", "0.05")),
        "high_vol_mult": float(os.getenv("POSITION_MAX_WEIGHT_HIGH_VOL_MULT", "0.65")),
        "cost_lambda": float(os.getenv("COST_PENALTY_LAMBDA", "1.2")),
    }


def _cost_model_settings() -> Dict[str, float]:
    return {
        "fee_bps": float(os.getenv("COST_FEE_BPS", "5.0")),
        "slippage_bps": float(os.getenv("COST_SLIPPAGE_BPS", "3.0")),
        "impact_coeff": float(os.getenv("COST_IMPACT_COEFF", "120.0")),
    }


def _score_to_size(score: float, confidence: float, est_cost_bps: float, vol_bucket: str, sizing_cfg: Dict[str, float]) -> float:
    cfg = dict(sizing_cfg or {})
    s = abs(float(score))
    c = max(0.0, min(1.0, float(confidence)))
    if s < cfg["entry_z"]:
        return 0.0
    core = np.tanh((s - cfg["entry_z"]) / max(1e-9, cfg["entry_z"] * 4.0))
    conf_boost = 0.4 + 0.6 * c
    cost_penalty = max(0.1, 1.0 - cfg["cost_lambda"] * max(0.0, est_cost_bps) / 10000.0)
    vol_penalty = cfg["high_vol_mult"] if vol_bucket == "high" else 1.0
    return float(max(0.0, cfg["max_weight_base"] * core * conf_boost * cost_penalty * vol_penalty))


def _target_vol_bucket(target: str) -> str:
    rows = repo.load_price_history(target.upper(), lookback_days=2)
    prices = np.array([float(r.get("price") or 0.0) for r in rows if float(r.get("price") or 0.0) > 0], dtype=np.float64)
    if prices.size < 24:
        return "normal"
    rets = np.diff(prices) / np.clip(prices[:-1], 1e-12, None)
    sigma = float(np.std(rets[-96:] if rets.size > 96 else rets))
    return "high" if sigma > 0.02 else "normal"


def _bucket_limits() -> Dict[str, float]:
    return {
        "trend": float(os.getenv("RISK_BUCKET_TREND", "0.5")),
        "event": float(os.getenv("RISK_BUCKET_EVENT", "0.4")),
        "mean_reversion": float(os.getenv("RISK_BUCKET_MEAN_REVERSION", "0.35")),
    }


def _run_source_filters() -> Tuple[List[str], List[str]]:
    include_raw = os.getenv("BACKTEST_GATE_INCLUDE_SOURCES", "prod")
    exclude_raw = os.getenv("BACKTEST_GATE_EXCLUDE_SOURCES", "smoke,async_test,maintenance")
    include = [s.strip().lower() for s in include_raw.split(",") if s.strip()]
    exclude = [s.strip().lower() for s in exclude_raw.split(",") if s.strip()]
    return include, exclude


def _data_regime_filters() -> List[str]:
    allowed = {"prod_live", "maintenance_replay", "mixed"}
    raw = os.getenv("BACKTEST_GATE_DATA_REGIMES", "prod_live")
    out = [s.strip().lower() for s in raw.split(",") if s.strip()]
    out = [s for s in out if s in allowed]
    return out or ["prod_live"]


def _score_source_filter(value: Optional[str]) -> str:
    return _strategybacktest_score_source_filter(value)


def _default_gate_thresholds() -> Dict[str, Any]:
    return {
        "min_ic": float(os.getenv("GATE_MIN_IC", "0.0")),
        "min_pnl_after_cost": float(os.getenv("GATE_MIN_PNL_AFTER_COST", "0.0")),
        "max_drawdown": float(os.getenv("GATE_MAX_DRAWDOWN", "0.2")),
        "windows": int(max(1, int(os.getenv("GATE_WINDOWS", "3")))),
    }


def _single_run_gate_pass(
    *,
    metrics: Dict[str, Any],
    min_ic: float,
    min_pnl_after_cost: float,
    max_drawdown: float,
) -> bool:
    require_leakage_pass = _env_flag("GATE_REQUIRE_LEAKAGE_PASS", "1")
    allow_degraded_runs = _env_flag("GATE_ALLOW_DEGRADED_RUNS", "0")
    leakage_passed = bool((metrics.get("leakage_checks") or {}).get("passed", False))
    degraded = bool(metrics.get("degraded", False))
    if require_leakage_pass and not leakage_passed:
        return False
    if (not allow_degraded_runs) and degraded:
        return False
    ic = float(metrics.get("ic", 0.0) or 0.0)
    pnl = float(metrics.get("pnl_after_cost", 0.0) or 0.0)
    dd = float(metrics.get("max_drawdown", 1.0) or 1.0)
    return bool(ic >= min_ic and pnl >= min_pnl_after_cost and dd <= max_drawdown)


def _strict_asof_hard_fail_enabled() -> bool:
    return _env_flag("BACKTEST_STRICT_ASOF_HARD_FAIL", "0")


def _take_profit_mode() -> str:
    mode = str(os.getenv("RISK_TAKE_PROFIT_MODE", "")).strip().lower()
    if mode in {"soft", "hard"}:
        return mode
    return "hard" if _env_flag("RISK_TAKE_PROFIT_HARD_BLOCK", "0") else "soft"


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
    now = _utcnow()
    return int(max(0.0, (exp - now).total_seconds()))


def _evaluate_risk(
    proposed: List[RebalancePosition],
    current: List[RebalancePosition],
    realized_drawdown: float,
    max_sector_exposure_override: Optional[float] = None,
    max_style_exposure_override: Optional[float] = None,
) -> Tuple[List[RebalancePosition], List[str], float, float]:
    limits = _risk_limits()
    dd_limit = float(limits["max_realized_drawdown"])
    dd = max(0.0, float(realized_drawdown))
    warn_threshold, near_limit_threshold = _drawdown_thresholds(dd_limit)
    if dd_limit > 1e-9 and dd >= warn_threshold > 0:
        span = max(1e-9, dd_limit - warn_threshold)
        ratio = min(1.0, max(0.0, (dd - warn_threshold) / span))
        shrink = max(0.25, 1.0 - ratio * 0.75)
        limits["max_single_weight"] = max(0.01, limits["max_single_weight"] * shrink)
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

    if dd >= near_limit_threshold > 0:
        current_map = _position_map(current)
        reduced: List[RebalancePosition] = []
        blocked_additions = 0
        for p in capped:
            key = (p.target.upper(), p.track)
            curr_w = float(current_map.get(key, 0.0))
            next_w = float(p.weight)
            if abs(next_w) > abs(curr_w) + 1e-9:
                blocked_additions += 1
                next_w = curr_w
            reduced.append(
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=next_w,
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
            )
        if blocked_additions > 0:
            violations.append("drawdown_near_limit_reduce_only")
        capped = reduced

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
        "feature_version": FEATURE_VERSION,
    }

    outputs = {
        "p_next_round": {
            "6m": round(max(0.01, min(0.99, p_next_round + 0.05)), 4),
            "12m": round(p_next_round, 4),
            "24m": round(max(0.01, min(0.99, p_next_round - 0.08)), 4),
        },
        "p_exit_24m": round(p_exit, 4),
        "expected_moic_distribution": expected_moic,
        "as_of": _utcnow().isoformat(),
    }

    return {
        "target": company_name,
        "score": float(round(p_next_round, 4)),
        "confidence": float(round(min(0.95, 0.55 + len(context) * 0.04), 4)),
        "outputs": outputs,
        "explanation": explanation,
    }


def _build_liquid_prediction(symbol: str, horizon: str) -> Dict[str, object]:
    return _build_liquid_model_prediction(symbol, horizon)


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
    # Keep native score polarity; downstream auto-polarity logic can explicitly invert when justified.
    return float(feature_signal_score(payload))


def _to_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except Exception:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    return None


def _extract_feature_time(row: Dict[str, Any]) -> Optional[datetime]:
    for key in ("feature_available_at", "as_of_ts", "as_of", "event_time"):
        dt = _to_utc_datetime(row.get(key))
        if dt is not None:
            return dt.astimezone(timezone.utc)
    return None


def _extract_price_time(row: Dict[str, Any]) -> Optional[datetime]:
    for key in ("timestamp", "ts", "as_of"):
        dt = _to_utc_datetime(row.get(key))
        if dt is not None:
            return dt.astimezone(timezone.utc)
    return None


def _align_feature_price_rows(
    feature_rows: List[Dict[str, Any]],
    price_rows: List[Dict[str, Any]],
    alignment_mode: str,
    max_feature_staleness_hours: int,
    alignment_version: str,
    allow_legacy_fallback: bool = False,
) -> Dict[str, Any]:
    # Legacy/index fallback path is intentionally disabled; keep arg only for compatibility.
    _ = allow_legacy_fallback
    mode = str(alignment_mode or "strict_asof").strip().lower()
    if mode != "strict_asof":
        mode = "strict_asof"

    price_points: List[Tuple[datetime, int, float, float]] = []
    dropped_price_ts = 0
    for idx, row in enumerate(price_rows):
        ts = _extract_price_time(row)
        px = float(row.get("price") or 0.0)
        vol = float(row.get("volume") or 0.0)
        if ts is None or px <= 0.0:
            dropped_price_ts += 1
            continue
        price_points.append((ts, idx, px, vol))
    price_points.sort(key=lambda x: x[0])
    if len(price_points) < 3:
        total_steps = int(max(0, len(price_points) - 1))
        return {
            "feature_rows": [],
            "feature_indices": [],
            "rets": np.array([], dtype=np.float64),
            "vols": np.array([], dtype=np.float64),
            "step_timestamps": [],
            "audit": {
                "alignment_mode_requested": mode,
                "alignment_mode_applied": "strict_asof",
                "alignment_version": alignment_version,
                "total_price_steps": total_steps,
                "aligned_steps": 0,
                "dropped_missing_feature": total_steps,
                "dropped_stale_feature": 0,
                "dropped_missing_timestamps": int(dropped_price_ts + len(feature_rows)),
                "leakage_violations": 0,
                "max_feature_age_hours": None,
                "min_feature_age_hours": None,
                "fail_fast": True,
                "fail_fast_reason": "missing_price_timestamps",
            },
        }

    ft_candidates: List[Tuple[datetime, int]] = []
    for idx, row in enumerate(feature_rows):
        ts = _extract_feature_time(row)
        if ts is None:
            continue
        ft_candidates.append((ts, idx))
    ft_candidates.sort(key=lambda x: x[0])
    if not ft_candidates:
        total_steps = int(max(0, len(price_points) - 1))
        return {
            "feature_rows": [],
            "feature_indices": [],
            "rets": np.array([], dtype=np.float64),
            "vols": np.array([], dtype=np.float64),
            "step_timestamps": [],
            "audit": {
                "alignment_mode_requested": mode,
                "alignment_mode_applied": "strict_asof",
                "alignment_version": alignment_version,
                "total_price_steps": total_steps,
                "aligned_steps": 0,
                "dropped_missing_feature": total_steps,
                "dropped_stale_feature": 0,
                "dropped_missing_timestamps": int(dropped_price_ts + len(feature_rows)),
                "leakage_violations": 0,
                "max_feature_age_hours": None,
                "min_feature_age_hours": None,
                "fail_fast": True,
                "fail_fast_reason": "missing_feature_timestamps",
            },
        }

    ft_times = [x[0] for x in ft_candidates]
    aligned_features: List[Dict[str, Any]] = []
    aligned_feature_indices: List[int] = []
    aligned_rets: List[float] = []
    aligned_vols: List[float] = []
    aligned_step_ts: List[Optional[datetime]] = []
    stale_drops = 0
    missing_feature_drops = 0
    leakage_violations = 0
    feature_ages: List[float] = []
    staleness_hours = float(max(1, max_feature_staleness_hours))

    for i in range(len(price_points) - 1):
        decision_ts, _, px0, _ = price_points[i]
        step_ts, _, px1, vol1 = price_points[i + 1]
        idx = bisect_right(ft_times, decision_ts) - 1
        if idx < 0:
            missing_feature_drops += 1
            continue
        feat_ts, feat_idx = ft_candidates[idx]
        age_hours = max(0.0, (decision_ts - feat_ts).total_seconds() / 3600.0)
        if feat_ts > decision_ts:
            leakage_violations += 1
            continue
        if age_hours > staleness_hours:
            stale_drops += 1
            continue
        feature_ages.append(age_hours)
        aligned_features.append(feature_rows[feat_idx])
        aligned_feature_indices.append(int(feat_idx))
        aligned_rets.append(float((px1 - px0) / max(px0, 1e-12)))
        aligned_vols.append(float(max(1.0, vol1)))
        aligned_step_ts.append(step_ts)

    return {
        "feature_rows": aligned_features,
        "feature_indices": aligned_feature_indices,
        "rets": np.array(aligned_rets, dtype=np.float64),
        "vols": np.array(aligned_vols, dtype=np.float64),
        "step_timestamps": aligned_step_ts,
        "audit": {
            "alignment_mode_requested": mode,
            "alignment_mode_applied": "strict_asof",
            "alignment_version": alignment_version,
            "total_price_steps": int(max(0, len(price_points) - 1)),
            "aligned_steps": int(len(aligned_rets)),
            "dropped_missing_feature": int(missing_feature_drops),
            "dropped_stale_feature": int(stale_drops),
            "dropped_missing_timestamps": int(dropped_price_ts + (len(feature_rows) - len(ft_candidates))),
            "leakage_violations": int(leakage_violations),
            "max_feature_age_hours": round(float(max(feature_ages)), 6) if feature_ages else None,
            "min_feature_age_hours": round(float(min(feature_ages)), 6) if feature_ages else None,
        },
    }


def _run_model_replay_backtest(
    feature_rows: List[Dict[str, Any]],
    price_rows: List[Dict[str, Any]],
    fee_bps: float,
    slippage_bps: float,
    sizing_override: Optional[Dict[str, float]] = None,
    raw_series_override: Optional[np.ndarray] = None,
    signal_polarity_mode: str = "normal",
    calibration_ratio: float = 0.5,
    alignment_mode: str = "strict_asof",
    max_feature_staleness_hours: int = 24 * 14,
    alignment_version: str = "strict_asof_v1",
    aligned_bundle: Optional[Dict[str, Any]] = None,
    strict_asof_fail_fast: bool = False,
) -> Dict[str, Any]:
    cfg = _cost_model_settings()
    sizing = _position_sizing_settings()
    if sizing_override:
        for k in ("entry_z", "exit_z", "max_weight_base", "high_vol_mult", "cost_lambda"):
            if k in sizing_override and sizing_override[k] is not None:
                sizing[k] = float(sizing_override[k])
    fee_bps = float(fee_bps if fee_bps >= 0 else cfg["fee_bps"])
    slippage_bps = float(slippage_bps if slippage_bps >= 0 else cfg["slippage_bps"])
    impact_coeff = float(cfg["impact_coeff"])
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
    alignment = aligned_bundle or _align_feature_price_rows(
        feature_rows=feature_rows,
        price_rows=price_rows,
        alignment_mode=alignment_mode,
        max_feature_staleness_hours=max_feature_staleness_hours,
        alignment_version=alignment_version,
        allow_legacy_fallback=False,
    )
    aligned_features: List[Dict[str, Any]] = list(alignment.get("feature_rows") or [])
    rets_obj = alignment.get("rets")
    vols_obj = alignment.get("vols")
    aligned_rets = np.array(rets_obj if rets_obj is not None else [], dtype=np.float64).reshape(-1)
    aligned_vols = np.array(vols_obj if vols_obj is not None else [], dtype=np.float64).reshape(-1)
    aligned_idx = [int(x) for x in (alignment.get("feature_indices") or [])]
    aligned_step_ts = list(alignment.get("step_timestamps") or [])
    alignment_audit = dict(alignment.get("audit") or {})
    if bool(alignment_audit.get("fail_fast", False)) and bool(strict_asof_fail_fast):
        fail_reason = str(alignment_audit.get("fail_fast_reason") or "alignment_unavailable")
        return {
            "status": "failed",
            "reason": f"strict_asof_alignment_fail_fast:{fail_reason}",
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "lineage_coverage": 0.0,
            "alignment_audit": alignment_audit,
            "leakage_checks": {
                "passed": False,
                "leakage_violations": int(alignment_audit.get("leakage_violations", 0) or 0),
                "alignment_mode": str(alignment_audit.get("alignment_mode_applied") or alignment_mode),
                "alignment_version": str(alignment_audit.get("alignment_version") or alignment_version),
            },
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
        }
    n = int(min(len(aligned_features), aligned_rets.shape[0], aligned_vols.shape[0]))
    if n < 8:
        return {
            "status": "failed",
            "reason": "insufficient_aligned_samples",
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "lineage_coverage": 0.0,
            "alignment_audit": alignment_audit,
            "leakage_checks": {
                "passed": False,
                "leakage_violations": int(alignment_audit.get("leakage_violations", 0) or 0),
                "alignment_mode": str(alignment_audit.get("alignment_mode_applied") or alignment_mode),
                "alignment_version": str(alignment_audit.get("alignment_version") or alignment_version),
            },
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
        }

    polarity = 1.0
    max_abs_position = float(os.getenv("BACKTEST_MAX_ABS_POSITION", str(sizing["max_weight_base"])))
    max_step_turnover = float(os.getenv("BACKTEST_MAX_STEP_TURNOVER", "0.12"))
    cost_edge_mult = float(os.getenv("BACKTEST_COST_EDGE_MULT", "3.0"))

    scores: List[float] = []
    future_rets: List[float] = []
    step_pnls: List[float] = []
    prev_signal = 0.0
    eq = 1.0
    eq_curve = [eq]
    fee_cost = 0.0
    slippage_cost = 0.0
    impact_cost = 0.0
    wins = 0
    turnover = 0.0
    unique_lineages = set()
    regime_steps: Dict[str, Dict[str, float]] = {
        "bull": {"samples": 0.0, "wins": 0.0, "turnover": 0.0, "ret_sum": 0.0},
        "bear": {"samples": 0.0, "wins": 0.0, "turnover": 0.0, "ret_sum": 0.0},
        "sideways": {"samples": 0.0, "wins": 0.0, "turnover": 0.0, "ret_sum": 0.0},
    }
    if raw_series_override is not None:
        raw_series = np.array(raw_series_override, dtype=np.float64).reshape(-1)
        if raw_series.shape[0] == len(feature_rows) and aligned_idx:
            raw_series = raw_series[np.array(aligned_idx, dtype=np.int64)]
        if raw_series.shape[0] < n:
            raw_series = np.pad(raw_series, (0, n - raw_series.shape[0]), mode="constant")
        elif raw_series.shape[0] > n:
            raw_series = raw_series[:n]
    else:
        raw_series = np.array(
            [polarity * _feature_signal_score((aligned_features[i].get("feature_payload") or {})) for i in range(n)],
            dtype=np.float64,
        )
    polarity_mode = str(signal_polarity_mode or "normal").strip().lower()
    polarity_mode = polarity_mode if polarity_mode in {"normal", "auto_train_ic", "auto_train_pnl"} else "normal"
    polarity_selection = "disabled"
    polarity_train_ic = 0.0
    polarity_train_edge = 0.0
    if polarity_mode != "normal" and n >= 16:
        ratio = float(min(0.9, max(0.2, calibration_ratio)))
        cal_n = int(min(n - 1, max(16, round(n * ratio))))
        raw_cal = raw_series[:cal_n]
        ret_cal = aligned_rets[:cal_n]
        if raw_cal.size >= 8 and ret_cal.size >= 8 and float(np.std(raw_cal)) > 1e-12 and float(np.std(ret_cal)) > 1e-12:
            polarity_train_ic = float(np.corrcoef(raw_cal, ret_cal)[0, 1])
            polarity_train_edge = float(np.mean(raw_cal * ret_cal))
            ic_threshold = float(os.getenv("BACKTEST_POLARITY_IC_THRESHOLD", "0.005"))
            edge_threshold = float(os.getenv("BACKTEST_POLARITY_EDGE_THRESHOLD", "0.0"))
            if polarity_mode == "auto_train_pnl":
                use_auto = abs(polarity_train_edge) >= max(0.0, edge_threshold)
            else:
                use_auto = abs(polarity_train_ic) >= max(0.0, ic_threshold)
            if use_auto:
                if polarity_mode == "auto_train_pnl":
                    polarity = 1.0 if polarity_train_edge >= 0.0 else -1.0
                    polarity_selection = "auto_train_pnl"
                else:
                    polarity = 1.0 if polarity_train_ic >= 0.0 else -1.0
                    polarity_selection = "auto_train_ic"
            else:
                polarity_selection = "auto_threshold_hold"
    if polarity < 0:
        raw_series = -raw_series
    raw_abs_p75 = float(np.percentile(np.abs(raw_series), 75)) if raw_series.size else 0.0
    dynamic_scale = 1.0
    if raw_abs_p75 > 1e-9:
        target_scale = max(float(sizing["entry_z"]), 0.012)
        dynamic_scale = min(40.0, max(1.0, target_scale / raw_abs_p75))
    scaled_p75 = raw_abs_p75 * dynamic_scale
    base_entry = float(os.getenv("BACKTEST_ENTRY_Z_MIN", "0.004"))
    base_exit = float(os.getenv("BACKTEST_EXIT_Z_MIN", "0.0015"))
    adaptive_entry = scaled_p75 * 0.6 if scaled_p75 > 0 else base_entry
    backtest_entry_z = max(base_entry, min(float(sizing["entry_z"]), adaptive_entry))
    backtest_exit_z = max(base_exit, min(float(sizing["exit_z"]), backtest_entry_z * 0.5))

    def _regime_label(idx: int) -> str:
        st = max(0, idx - 96)
        window = aligned_rets[st:idx] if idx > st else np.array([], dtype=np.float64)
        if window.size < 12:
            return "sideways"
        mu = float(np.mean(window))
        sigma = float(np.std(window))
        thr = max(0.0008, sigma * 0.35)
        if mu > thr:
            return "bull"
        if mu < -thr:
            return "bear"
        return "sideways"

    for i in range(n):
        f = aligned_features[i].get("feature_payload") or {}
        lineage_id = str(aligned_features[i].get("lineage_id") or "").strip()
        if lineage_id:
            unique_lineages.add(lineage_id)
        raw = float(dynamic_scale * raw_series[i])
        vol_proxy = abs(float(f.get("vol_12", 0.0) or 0.0)) + abs(float(f.get("vol_48", 0.0) or 0.0))
        vol_bucket = "high" if vol_proxy > 0.015 else "normal"
        vol = float(max(1.0, aligned_vols[i]))
        impact_bps = min(30.0, impact_coeff / np.sqrt(vol))
        est_cost_bps = float(fee_bps + slippage_bps + impact_bps)
        edge_floor = max(backtest_entry_z, cost_edge_mult * est_cost_bps / 10000.0)
        if abs(raw) < edge_floor:
            raw = 0.0
        score_for_size = raw * (float(sizing["entry_z"]) / max(backtest_entry_z, 1e-9))
        desired_abs = _score_to_size(score_for_size, 1.0, est_cost_bps, vol_bucket, sizing_cfg=sizing)
        desired = desired_abs if raw >= 0 else -desired_abs
        desired = float(max(-max_abs_position, min(max_abs_position, desired)))
        if abs(raw) < backtest_exit_z:
            desired = 0.0
        elif abs(raw) < backtest_entry_z and abs(prev_signal) > 1e-6:
            desired = 0.5 * prev_signal
        step = desired - prev_signal
        if step > max_step_turnover:
            desired = prev_signal + max_step_turnover
        elif step < -max_step_turnover:
            desired = prev_signal - max_step_turnover
        nxt = float(aligned_rets[i])
        trade_amt = abs(desired - prev_signal)
        est_cost = (fee_bps + slippage_bps + impact_bps) / 10000.0 * trade_amt
        penalty = min(0.7, float(sizing["cost_lambda"]) * est_cost * 80.0)
        signal = desired * max(0.0, 1.0 - penalty)
        if abs(signal) < backtest_exit_z:
            signal = 0.0
        c_fee = fee_bps / 10000.0 * abs(signal - prev_signal)
        c_slip = slippage_bps / 10000.0 * abs(signal - prev_signal)
        c_imp = impact_bps / 10000.0 * abs(signal - prev_signal)
        pnl = signal * nxt - (c_fee + c_slip + c_imp)
        step_pnls.append(float(pnl))
        eq *= (1.0 + pnl)
        eq_curve.append(eq)
        fee_cost += c_fee
        slippage_cost += c_slip
        impact_cost += c_imp
        scores.append(signal)
        future_rets.append(nxt)
        if abs(signal) > 1e-6 and signal * nxt > 0:
            wins += 1
        trade_turnover = abs(signal - prev_signal)
        turnover += trade_turnover
        regime = _regime_label(i)
        bucket = regime_steps.get(regime, regime_steps["sideways"])
        bucket["samples"] += 1.0
        if abs(signal) > 1e-6 and signal * nxt > 0:
            bucket["wins"] += 1.0
        bucket["turnover"] += trade_turnover
        bucket["ret_sum"] += pnl
        prev_signal = signal

    s = np.array(scores, dtype=np.float64)
    y = np.array(future_rets, dtype=np.float64)
    ic = float(np.corrcoef(s, y)[0, 1]) if np.std(s) > 0 and np.std(y) > 0 else 0.0
    hit_rate = float(wins / max(1, len(scores)))
    sharpe_step_raw = 0.0
    if len(step_pnls) > 1:
        p = np.array(step_pnls, dtype=np.float64)
        p_std = float(np.std(p))
        if p_std > 1e-12:
            sharpe_step_raw = float((float(np.mean(p)) / p_std) * np.sqrt(24.0 * 365.0))
    daily_map: Dict[str, float] = {}
    for i, pnl in enumerate(step_pnls):
        ts = aligned_step_ts[i] if i < len(aligned_step_ts) else None
        if isinstance(ts, datetime):
            ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            day_key = ts_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")
        else:
            day_key = f"idx_{i // 288}"  # 5m bars fallback bucket
        daily_map[day_key] = daily_map.get(day_key, 0.0) + float(pnl)
    daily_vals = np.array(list(daily_map.values()), dtype=np.float64) if daily_map else np.array([], dtype=np.float64)
    observation_days = int(daily_vals.shape[0])
    vol_floor_applied = False
    sharpe_daily = 0.0
    if observation_days > 1:
        mu_d = float(np.mean(daily_vals))
        sd_d = float(np.std(daily_vals))
        if sd_d < 1e-9:
            sd_d = 1e-9
            vol_floor_applied = True
        sharpe_daily = float((mu_d / sd_d) * np.sqrt(365.0))
    active_curve = eq_curve
    peak = active_curve[0]
    max_dd = 0.0
    for v in active_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / max(peak, 1e-12))
    regime_breakdown: Dict[str, Dict[str, float]] = {}
    for name, vals in regime_steps.items():
        cnt = int(vals["samples"])
        if cnt <= 0:
            continue
        regime_breakdown[name] = {
            "samples": cnt,
            "hit_rate": round(float(vals["wins"] / max(1.0, vals["samples"])), 6),
            "turnover": round(float(vals["turnover"]), 6),
            "pnl_after_cost": round(float(vals["ret_sum"]), 6),
        }
    return {
        "status": "completed",
        "reason": "ok",
        "samples": int(len(scores)),
        "ic": round(ic, 6),
        "sharpe": round(sharpe_daily, 6),
        "sharpe_step_raw": round(sharpe_step_raw, 6),
        "sharpe_daily": round(sharpe_daily, 6),
        "sharpe_method": "daily_agg_v1",
        "observation_days": observation_days,
        "vol_floor_applied": bool(vol_floor_applied),
        "hit_rate": round(hit_rate, 6),
        "turnover": round(float(turnover), 6),
        "pnl_after_cost": round(float(active_curve[-1] - 1.0), 6),
        "max_drawdown": round(float(max_dd), 6),
        "lineage_coverage": round(float(len(unique_lineages) / max(1, len(scores))), 6),
        "alignment_audit": alignment_audit,
        "leakage_checks": {
            "passed": bool(int(alignment_audit.get("leakage_violations", 0) or 0) == 0),
            "leakage_violations": int(alignment_audit.get("leakage_violations", 0) or 0),
            "alignment_mode": str(alignment_audit.get("alignment_mode_applied") or alignment_mode),
            "alignment_version": str(alignment_audit.get("alignment_version") or alignment_version),
        },
        "signal_polarity": "inverted" if polarity < 0 else "normal",
        "polarity_selection": polarity_selection,
        "polarity_mode_requested": polarity_mode,
        "polarity_train_ic": round(float(polarity_train_ic), 6),
        "polarity_train_edge": round(float(polarity_train_edge), 8),
        "regime_breakdown": regime_breakdown,
        "cost_breakdown": {
            "fee": round(float(fee_cost), 8),
            "slippage": round(float(slippage_cost), 8),
            "impact": round(float(impact_cost), 8),
        },
    }


def _feature_vector_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    return np.array([float(payload.get(k, 0.0) or 0.0) for k in LIQUID_FEATURE_KEYS], dtype=np.float64)


def _align_feature_dim(x: np.ndarray, dim: int, pad_value: float = 0.0) -> np.ndarray:
    if x.shape[0] < dim:
        return np.concatenate([x, np.full((dim - x.shape[0],), float(pad_value), dtype=np.float64)], axis=0)
    if x.shape[0] > dim:
        return x[:dim]
    return x


def _load_tabular_model_artifact(target: str) -> Optional[Dict[str, Any]]:
    target = str(target or "").strip().upper()
    if not target:
        return None
    sym = target.split("_")[0].lower()
    if sym in _TABULAR_MODEL_CACHE:
        return _TABULAR_MODEL_CACHE[sym]
    path = _default_model_dir() / f"liquid_{sym}_lgbm_baseline_v2.json"
    if not path.exists():
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    if not isinstance(data, dict):
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    if str(data.get("track") or "").strip().lower() not in {"", "liquid"}:
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    if str(data.get("feature_version") or "").strip() not in {"", FEATURE_VERSION}:
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    if str(data.get("data_version") or "").strip() not in {"", DATA_VERSION}:
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    schema_version = str(data.get("feature_payload_schema_version") or "").strip()
    if schema_version and schema_version not in FEATURE_PAYLOAD_SCHEMA_COMPAT:
        _TABULAR_MODEL_CACHE[sym] = None
        return None

    artifact: Dict[str, Any] = {
        "feature_dim": int(data.get("feature_dim", len(LIQUID_FEATURE_KEYS)) or len(LIQUID_FEATURE_KEYS)),
        "x_mean": np.array(data.get("x_mean", []), dtype=np.float64).reshape(-1),
        "x_std": np.array(data.get("x_std", []), dtype=np.float64).reshape(-1),
    }
    if isinstance(data.get("weights"), list):
        artifact["weights"] = np.array(data.get("weights", []), dtype=np.float64).reshape(-1)
    if (
        str(data.get("model") or "").strip().lower() == "lightgbm"
        and HAS_LGB
        and isinstance(data.get("booster_model"), str)
        and str(data.get("booster_model")).strip()
    ):
        try:
            artifact["booster"] = lgb.Booster(model_str=str(data["booster_model"]))
        except Exception:
            pass
    if "weights" not in artifact and "booster" not in artifact:
        _TABULAR_MODEL_CACHE[sym] = None
        return None
    _TABULAR_MODEL_CACHE[sym] = artifact
    return artifact


def _load_tabular_model_weights(target: str) -> Optional[Dict[str, Any]]:
    # backward-compatible alias used by existing tests/tooling
    return _load_tabular_model_artifact(target)


def _predict_expected_return_tabular(target: str, payload: Dict[str, Any]) -> Optional[float]:
    model = _load_tabular_model_weights(target)
    if model is None:
        return None
    x = _feature_vector_from_payload(payload)
    dim = int(model.get("feature_dim", x.shape[0]) or x.shape[0])
    x = _align_feature_dim(x, dim, pad_value=0.0)
    x_mean = model["x_mean"]
    x_std = model["x_std"]
    if x_mean.size > 0 and x_std.size > 0:
        x_mean = _align_feature_dim(x_mean, dim, pad_value=0.0)
        x_std = _align_feature_dim(x_std, dim, pad_value=1.0)
        x = (x - x_mean) / np.clip(x_std, 1e-6, None)
    booster = model.get("booster")
    if booster is not None:
        pred = booster.predict(x.reshape(1, -1))
        arr = np.array(pred, dtype=np.float64).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            return float(arr[0])
    w = model.get("weights")
    if isinstance(w, np.ndarray) and w.size > 0:
        ww = _align_feature_dim(w.reshape(-1), dim, pad_value=0.0)
        return float(x @ ww)
    return None


def _run_model_inference_backtest(
    target: str,
    feature_rows: List[Dict[str, Any]],
    price_rows: List[Dict[str, Any]],
    fee_bps: float,
    slippage_bps: float,
    sizing_override: Optional[Dict[str, float]] = None,
    signal_polarity_mode: str = "normal",
    calibration_ratio: float = 0.5,
    alignment_mode: str = "strict_asof",
    max_feature_staleness_hours: int = 24 * 14,
    alignment_version: str = "strict_asof_v1",
    strict_asof_fail_fast: bool = False,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    alignment = _align_feature_price_rows(
        feature_rows=feature_rows,
        price_rows=price_rows,
        alignment_mode=alignment_mode,
        max_feature_staleness_hours=max_feature_staleness_hours,
        alignment_version=alignment_version,
        allow_legacy_fallback=False,
    )
    aligned_features = list(alignment.get("feature_rows") or [])
    n = int(len(aligned_features))
    if n <= 1:
        replay = _run_model_replay_backtest(
            feature_rows=feature_rows,
            price_rows=price_rows,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            sizing_override=sizing_override,
            signal_polarity_mode=signal_polarity_mode,
            calibration_ratio=calibration_ratio,
            alignment_mode=alignment_mode,
            max_feature_staleness_hours=max_feature_staleness_hours,
            alignment_version=alignment_version,
            aligned_bundle=alignment,
            strict_asof_fail_fast=strict_asof_fail_fast,
        )
        replay["model_inference_coverage"] = 0.0
        replay["fallback_used"] = False
        replay["score_source"] = "model"
        return replay

    model_raw: List[float] = []
    predicted = 0
    failed_points = 0
    service = _get_liquid_model_service()
    for i in range(n):
        payload = aligned_features[i].get("feature_payload") or {}
        try:
            pred = service.predict_from_feature_payload(
                target=target,
                payload=payload,
                horizon="1d",
                model_name=model_name,
                model_version=model_version,
                require_artifact=True,
            )
            exp = float(pred.get("expected_return") or 0.0)
        except Exception:
            exp = None
        if exp is not None and np.isfinite(float(exp)):
            predicted += 1
            model_raw.append(float(exp))
        else:
            failed_points += 1

    if failed_points > 0:
        return {
            "status": "failed",
            "reason": "model_inference_incomplete",
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "lineage_coverage": 0.0,
            "model_inference_coverage": round(float(predicted / max(1, n)), 6),
            "fallback_used": False,
            "score_source": "model",
            "alignment_audit": dict(alignment.get("audit") or {}),
            "leakage_checks": {
                "passed": False,
                "leakage_violations": int((alignment.get("audit") or {}).get("leakage_violations", 0) or 0),
                "alignment_mode": str((alignment.get("audit") or {}).get("alignment_mode_applied") or alignment_mode),
                "alignment_version": str((alignment.get("audit") or {}).get("alignment_version") or alignment_version),
            },
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
        }

    replay = _run_model_replay_backtest(
        feature_rows=feature_rows,
        price_rows=price_rows,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        sizing_override=sizing_override,
        raw_series_override=np.array(model_raw, dtype=np.float64),
        signal_polarity_mode=signal_polarity_mode,
        calibration_ratio=calibration_ratio,
        alignment_mode=alignment_mode,
        max_feature_staleness_hours=max_feature_staleness_hours,
        alignment_version=alignment_version,
        aligned_bundle=alignment,
        strict_asof_fail_fast=strict_asof_fail_fast,
    )
    replay["model_inference_coverage"] = round(float(predicted / max(1, n)), 6)
    replay["fallback_used"] = False
    replay["score_source"] = "model"
    return replay


def _default_model_by_track(track: str) -> Tuple[str, str]:
    if track == "liquid":
        return "liquid_ttm_ensemble", "v2.1"
    return "vc_survival_model", "v2.1"


def _parity_check(
    track: str,
    max_deviation: float = 0.10,
    min_completed_runs: int = 5,
    score_source: str = "model",
) -> Dict[str, Any]:
    score_source = _score_source_filter(score_source)
    include_sources, exclude_sources = _run_source_filters()
    data_regimes = _data_regime_filters()
    runs = repo.list_recent_backtest_runs(
        track=track,
        limit=500,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        data_regimes=data_regimes,
    )
    total_completed = sum(
        1
        for r in runs
        if isinstance(r.get("metrics"), dict)
        and str((r.get("metrics") or {}).get("status") or "").lower() == "completed"
        and r.get("superseded_by_run_id") is None
        and _score_source_filter((r.get("config") or {}).get("score_source")) == score_source
    )

    bt_target_7 = repo.get_backtest_target_pnl_window(
        track=track,
        window_hours=24 * 7,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        score_source=score_source,
        data_regimes=data_regimes,
    )
    bt_target_30 = repo.get_backtest_target_pnl_window(
        track=track,
        window_hours=24 * 30,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        score_source=score_source,
        data_regimes=data_regimes,
    )
    px_target_7 = repo.get_execution_target_realized_window(track=track, window_hours=24 * 7)
    px_target_30 = repo.get_execution_target_realized_window(track=track, window_hours=24 * 30)

    if total_completed < min_completed_runs:
        METRIC_GATE_STATUS.labels(track=track, metric="parity_30d").set(0.0)
        return {
            "status": "insufficient_observation",
            "passed": False,
            "completed_runs": total_completed,
            "min_completed_runs": min_completed_runs,
            "score_source": score_source,
            "data_regimes": data_regimes,
            "reason": "insufficient_completed_backtests",
        }

    matched_7 = sorted(set(bt_target_7.keys()) & set(px_target_7.keys()))
    matched_30 = sorted(set(bt_target_30.keys()) & set(px_target_30.keys()))
    orders_7 = int(sum((px_target_7.get(t) or {}).get("orders", 0.0) for t in matched_7))
    orders_30 = int(sum((px_target_30.get(t) or {}).get("orders", 0.0) for t in matched_30))
    if len(matched_30) < 2:
        METRIC_GATE_STATUS.labels(track=track, metric="parity_30d").set(0.0)
        return {
            "status": "insufficient_observation",
            "passed": False,
            "completed_runs": total_completed,
            "min_completed_runs": min_completed_runs,
            "score_source": score_source,
            "data_regimes": data_regimes,
            "reason": "insufficient_matched_targets",
            "matched_targets_count": len(matched_30),
            "paper_filled_orders_count": orders_30,
            "comparison_basis": "matched_filled_orders",
        }
    if orders_30 < 50:
        METRIC_GATE_STATUS.labels(track=track, metric="parity_30d").set(0.0)
        return {
            "status": "insufficient_observation",
            "passed": False,
            "completed_runs": total_completed,
            "min_completed_runs": min_completed_runs,
            "score_source": score_source,
            "data_regimes": data_regimes,
            "reason": "insufficient_paper_orders",
            "matched_targets_count": len(matched_30),
            "paper_filled_orders_count": orders_30,
            "comparison_basis": "matched_filled_orders",
        }

    def _window_pair(
        bt_map: Dict[str, Dict[str, float]], px_map: Dict[str, Dict[str, float]], matched: List[str]
    ) -> Tuple[float, float]:
        bt_sum = 0.0
        px_sum = 0.0
        total_w = 0.0
        for t in matched:
            bt_avg = float((bt_map.get(t) or {}).get("sum", 0.0)) / max(1.0, float((bt_map.get(t) or {}).get("count", 0.0)))
            px_w = float((px_map.get(t) or {}).get("sum_notional", 0.0))
            px_avg = float((px_map.get(t) or {}).get("sum_weighted", 0.0)) / max(1e-12, px_w)
            w = max(1.0, px_w)
            bt_sum += bt_avg * w
            px_sum += px_avg * w
            total_w += w
        if total_w <= 0:
            return 0.0, 0.0
        return bt_sum / total_w, px_sum / total_w

    bt_avg_7, px_avg_7 = _window_pair(bt_target_7, px_target_7, matched_7)
    bt_avg_30, px_avg_30 = _window_pair(bt_target_30, px_target_30, matched_30)
    d7 = abs(bt_avg_7 - px_avg_7)
    d30 = abs(bt_avg_30 - px_avg_30)
    parity_floor = float(os.getenv("PARITY_RETURN_FLOOR", "0.02"))
    rel7 = d7 / max(1e-6, parity_floor, abs(bt_avg_7), abs(px_avg_7))
    rel30 = d30 / max(1e-6, parity_floor, abs(bt_avg_30), abs(px_avg_30))
    passed = rel30 <= max_deviation
    METRIC_GATE_STATUS.labels(track=track, metric="parity_30d").set(1.0 if passed else 0.0)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "track": track,
        "score_source": score_source,
        "data_regimes": data_regimes,
        "max_deviation": max_deviation,
        "completed_runs": total_completed,
        "matched_targets_count": len(matched_30),
        "paper_filled_orders_count": orders_30,
        "comparison_basis": "matched_filled_orders",
        "gate_window": "30d",
        "alert_window": "7d",
        "window_details": {
            "7d": {
                "backtest_avg_pnl_after_cost": round(bt_avg_7, 9),
                "paper_realized_return": round(px_avg_7, 9),
                "matched_targets": len(matched_7),
                "paper_filled_orders": orders_7,
                "abs_delta": round(d7, 9),
                "relative_deviation": round(rel7, 9),
            },
            "30d": {
                "backtest_avg_pnl_after_cost": round(bt_avg_30, 9),
                "paper_realized_return": round(px_avg_30, 9),
                "matched_targets": len(matched_30),
                "paper_filled_orders": orders_30,
                "abs_delta": round(d30, 9),
                "relative_deviation": round(rel30, 9),
            },
        },
    }


def _evaluate_gate(
    track: str,
    min_ic: float,
    min_pnl_after_cost: float,
    max_drawdown: float,
    windows: int,
    score_source: str = "model",
) -> Tuple[bool, str, Dict[str, float], int]:
    return _strategybacktest_evaluate_gate(
        track=track,
        min_ic=min_ic,
        min_pnl_after_cost=min_pnl_after_cost,
        max_drawdown=max_drawdown,
        windows=windows,
        score_source=score_source,
        repo=repo,
        env_flag_fn=_env_flag,
        run_source_filters_fn=_run_source_filters,
        data_regime_filters_fn=_data_regime_filters,
    )


def _strategy_bucket(track: str, score: float, confidence: float, target: str) -> str:
    return _strategysignal_strategy_bucket(
        track=track,
        score=score,
        confidence=confidence,
        target=target,
        latest_feature_payload_fn=lambda sym: _latest_liquid_feature_payload(sym, max_lookback_days=14),
    )


def _parse_horizon_float_map(raw: str, defaults: Dict[str, float]) -> Dict[str, float]:
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
        if key in out:
            out[key] = float(val)
    return out


def _liquid_cost_profile(cost_profile: str, stack: Dict[str, Any]) -> Dict[str, float]:
    cfg = stack.get("cost_config") if isinstance(stack.get("cost_config"), dict) else {}
    multipliers = cfg.get("horizon_multipliers") if isinstance(cfg.get("horizon_multipliers"), dict) else {}
    fee_bps = float(cfg.get("fee_bps", 5.0) or 5.0)
    slippage_bps = float(cfg.get("slippage_bps", 3.0) or 3.0)
    impact_base_bps = float(cfg.get("impact_base_bps", 2.0) or 2.0)
    base_cost = max(0.0, (fee_bps + slippage_bps + impact_base_bps) / 10000.0)
    defaults = {
        "1h": base_cost * float(multipliers.get("1h", 1.0) if multipliers else 1.0),
        "4h": base_cost * float(multipliers.get("4h", 1.6) if multipliers else 1.6),
        "1d": base_cost * float(multipliers.get("1d", 2.4) if multipliers else 2.4),
        "7d": base_cost * float(multipliers.get("7d", 3.2) if multipliers else 3.2),
    }
    env_key = f"SIGNAL_COST_PROFILE_{str(cost_profile or 'standard').strip().upper()}"
    return _parse_horizon_float_map(os.getenv(env_key, ""), defaults)


def _horizon_signal_thresholds() -> Dict[str, Dict[str, float]]:
    score_defaults = {"1h": 0.20, "4h": 0.16, "1d": 0.10, "7d": 0.08}
    conf_defaults = {"1h": 0.55, "4h": 0.52, "1d": 0.50, "7d": 0.48}
    return {
        "score_entry": _parse_horizon_float_map(os.getenv("SIGNAL_SCORE_ENTRY_BY_HORIZON", ""), score_defaults),
        "confidence_min": _parse_horizon_float_map(os.getenv("SIGNAL_CONFIDENCE_MIN_BY_HORIZON", ""), conf_defaults),
    }


def _liquid_multi_horizon_signal(
    *,
    pred_outputs: Dict[str, Any],
    horizon: str,
    cost_profile: str,
) -> Dict[str, Any]:
    horizon_key = str(horizon or "1h").strip().lower() or "1h"
    expected = pred_outputs.get("expected_return_horizons") if isinstance(pred_outputs.get("expected_return_horizons"), dict) else {}
    conf = pred_outputs.get("signal_confidence_horizons") if isinstance(pred_outputs.get("signal_confidence_horizons"), dict) else {}
    vol = pred_outputs.get("vol_forecast_horizons") if isinstance(pred_outputs.get("vol_forecast_horizons"), dict) else {}
    if not expected:
        expected = {"1h": float(pred_outputs.get("expected_return", 0.0) or 0.0)}
    if not conf:
        conf = {"1h": float(pred_outputs.get("signal_confidence", 0.0) or 0.0)}
    if not vol:
        vol = {"1h": float(pred_outputs.get("vol_forecast", 0.0) or 0.0)}
    if horizon_key not in expected:
        horizon_key = "1h" if "1h" in expected else sorted(expected.keys())[0]
    costs = _liquid_cost_profile(cost_profile, pred_outputs.get("stack") if isinstance(pred_outputs.get("stack"), dict) else {})
    thresholds = _horizon_signal_thresholds()
    eps = max(1e-6, float(os.getenv("SIGNAL_VOL_EPS", "1e-4") or 1e-4))
    edge_map: Dict[str, float] = {}
    score_map: Dict[str, float] = {}
    action_map: Dict[str, str] = {}
    for h, ret in expected.items():
        hh = str(h).strip().lower()
        edge = float(ret or 0.0) - float(costs.get(hh, costs.get("1h", 0.0)))
        vv = float(vol.get(hh, vol.get("1h", 0.0)) or 0.0)
        sc = float(edge / max(vv, eps))
        cf = float(conf.get(hh, conf.get("1h", 0.0)) or 0.0)
        score_th = float(thresholds["score_entry"].get(hh, thresholds["score_entry"]["1h"]))
        conf_th = float(thresholds["confidence_min"].get(hh, thresholds["confidence_min"]["1h"]))
        if cf < conf_th:
            act = "hold"
        elif sc >= score_th:
            act = "buy"
        elif sc <= -score_th:
            act = "sell"
        else:
            act = "hold"
        edge_map[hh] = edge
        score_map[hh] = sc
        action_map[hh] = act
    selected_score = float(score_map.get(horizon_key, 0.0))
    selected_conf = float(conf.get(horizon_key, 0.0) or 0.0)
    selected_action = str(action_map.get(horizon_key, "hold"))
    return {
        "selected_horizon": horizon_key,
        "score": selected_score,
        "confidence": selected_conf,
        "action": selected_action,
        "edge_horizons": edge_map,
        "score_horizons": score_map,
        "action_horizons": action_map,
        "cost_horizons": costs,
        "expected_return_horizons": {str(k).lower(): float(v) for k, v in expected.items()},
        "signal_confidence_horizons": {str(k).lower(): float(v) for k, v in conf.items()},
        "vol_forecast_horizons": {str(k).lower(): float(v) for k, v in vol.items()},
    }


def _ks_statistic(a: List[float], b: List[float]) -> float:
    return _driftops_ks_statistic(a, b)


def _psi(reference: List[float], current: List[float], bins: int = 10) -> float:
    return _driftops_psi(reference, current, bins=bins)


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
    return _driftops_feature_drift_score(reference_payloads, current_payloads)


@router.post("/ingest/events", response_model=IngestEventsResponse)
async def ingest_events(payload: IngestEventsRequest) -> IngestEventsResponse:
    for ev in payload.events:
        if "market_scope" not in ev.payload:
            ev.payload["market_scope"] = ev.market_scope
    accepted, inserted, deduplicated, event_ids = repo.ingest_events(payload.events)
    INGEST_EVENTS_TOTAL.labels(status="accepted").inc(accepted)
    INGEST_EVENTS_TOTAL.labels(status="inserted").inc(inserted)
    INGEST_EVENTS_TOTAL.labels(status="deduplicated").inc(deduplicated)
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
        feature_set_id=FEATURE_VERSION,
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
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    mh = _liquid_multi_horizon_signal(
        pred_outputs=outputs,
        horizon=payload.horizon,
        cost_profile="standard",
    )
    prediction_id = repo.insert_prediction(
        track="liquid",
        target=payload.symbol.upper(),
        score=float(mh["score"]),
        confidence=float(mh["confidence"]),
        outputs={
            **outputs,
            "expected_return_horizons": mh["expected_return_horizons"],
            "signal_confidence_horizons": mh["signal_confidence_horizons"],
            "vol_forecast_horizons": mh["vol_forecast_horizons"],
            "score_horizons": mh["score_horizons"],
            "edge_horizons": mh["edge_horizons"],
            "cost_horizons": mh["cost_horizons"],
            "selected_horizon": mh["selected_horizon"],
        },
        explanation=result["explanation"],
        horizon=payload.horizon,
        feature_set_id=FEATURE_VERSION,
    )

    compat_horizon = str(mh.get("selected_horizon") or "1h")
    expected_h = dict(mh.get("expected_return_horizons") or {})
    conf_h = dict(mh.get("signal_confidence_horizons") or {})
    vol_h = dict(mh.get("vol_forecast_horizons") or {})
    return {
        "prediction_id": prediction_id,
        "track": "liquid",
        "target": payload.symbol.upper(),
        # Multi-horizon primary fields.
        "expected_return": expected_h,
        "signal_confidence": conf_h,
        "vol_forecast": vol_h,
        # Backward-compatible scalar fields.
        "expected_return_legacy": float(expected_h.get(compat_horizon, 0.0) or 0.0),
        "signal_confidence_legacy": float(conf_h.get(compat_horizon, 0.0) or 0.0),
        "vol_forecast_legacy": float(vol_h.get(compat_horizon, 0.0) or 0.0),
        "score_horizons": dict(mh.get("score_horizons") or {}),
        "edge_horizons": dict(mh.get("edge_horizons") or {}),
        "cost_horizons": dict(mh.get("cost_horizons") or {}),
        "action_horizons": dict(mh.get("action_horizons") or {}),
        "selected_horizon": compat_horizon,
        "stack": dict(outputs.get("stack") or {}),
        "degraded": bool(outputs.get("degraded", False)),
        "degraded_reasons": list(outputs.get("degraded_reasons") or []),
        "outputs": outputs,  # backward-compat window
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
    universe_resolve: Dict[str, Any] = {}
    if not targets:
        if payload.track == "liquid":
            env_targets = _default_liquid_targets()
            if payload.use_universe_snapshot:
                resolve_asof = payload.universe_asof or (_utcnow() - timedelta(days=max(1, int(payload.lookback_days))))
                universe_resolve = repo.resolve_asset_universe_asof(
                    track="liquid",
                    as_of=resolve_asof,
                    fallback_targets=env_targets,
                )
                targets = [s.strip().upper() for s in universe_resolve.get("symbols", []) if str(s).strip()]
            else:
                targets = env_targets
                universe_resolve = {
                    "track": "liquid",
                    "as_of": None,
                    "symbols": targets,
                    "source": "env_default",
                    "universe_version": "env_default",
                    "snapshot_at": None,
                }
        else:
            targets = ["OpenAI", "Anthropic", "Scale AI"]
    elif payload.track == "liquid":
        universe_resolve = {
            "track": "liquid",
            "as_of": None,
            "symbols": [s.strip().upper() for s in targets if str(s).strip()],
            "source": "payload_targets",
            "universe_version": "manual",
            "snapshot_at": None,
        }
    targets = [str(s).strip().upper() for s in targets if str(s).strip()]

    run_name = f"{payload.track}-wf-{_utcnow().strftime('%Y%m%d%H%M%S')}"
    cost_cfg = _cost_model_settings()
    effective_fee_bps = float(payload.fee_bps if payload.fee_bps >= 0 else cost_cfg["fee_bps"])
    effective_slippage_bps = float(payload.slippage_bps if payload.slippage_bps >= 0 else cost_cfg["slippage_bps"])
    config = payload.model_dump(mode="json")
    score_source = _score_source_filter(payload.score_source)
    data_regime = str(payload.data_regime).strip().lower()
    if "data_regime" not in payload.model_fields_set:
        if payload.run_source == "prod":
            data_regime = "prod_live"
        elif payload.run_source == "maintenance":
            data_regime = "maintenance_replay"
        else:
            data_regime = "mixed"
    if data_regime not in {"prod_live", "maintenance_replay", "mixed"}:
        data_regime = "mixed"
    if data_regime == "prod_live" and payload.run_source != "prod":
        data_regime = "mixed"
    if payload.run_source == "prod" and str(payload.alignment_mode).strip().lower() != "strict_asof":
        raise HTTPException(status_code=400, detail="prod run_source requires alignment_mode=strict_asof")
    strict_mode_requested = str(payload.alignment_mode).strip().lower() == "strict_asof"
    strict_asof_fail_fast = bool(strict_mode_requested)
    config["score_source"] = score_source
    config["data_regime"] = data_regime
    config["targets"] = targets
    if payload.track == "liquid":
        config["universe_resolve"] = universe_resolve
    config["fee_bps"] = effective_fee_bps
    config["slippage_bps"] = effective_slippage_bps
    config["strict_asof_hard_fail"] = bool(strict_asof_fail_fast)
    config["strict_asof_gate_mode"] = "hard" if strict_asof_fail_fast else "soft"
    gate_cfg = _default_gate_thresholds()
    run_id = repo.create_backtest_run(run_name=run_name, track=payload.track, config=config, run_source=payload.run_source)
    model_name = payload.model_name or _default_model_by_track(payload.track)[0]
    model_version = payload.model_version or _default_model_by_track(payload.track)[1]
    sizing_override: Dict[str, float] = {}
    if payload.signal_entry_z_min is not None:
        sizing_override["entry_z"] = float(payload.signal_entry_z_min)
    if payload.signal_exit_z_min is not None:
        sizing_override["exit_z"] = float(payload.signal_exit_z_min)
    if payload.position_max_weight_base is not None:
        sizing_override["max_weight_base"] = float(payload.position_max_weight_base)
    if payload.position_max_weight_high_vol_mult is not None:
        sizing_override["high_vol_mult"] = float(payload.position_max_weight_high_vol_mult)
    if payload.cost_penalty_lambda is not None:
        sizing_override["cost_lambda"] = float(payload.cost_penalty_lambda)
    missing_model_targets: List[str] = []
    model_artifact_ok = True
    if score_source == "model" and payload.require_model_artifact:
        if payload.track == "liquid":
            service = _get_liquid_model_service()
            missing_model_targets = [t for t in targets if not service.has_required_artifacts(target=t, model_name=model_name)]
            model_artifact_ok = len(missing_model_targets) == 0
        else:
            model_artifact_ok = repo.model_artifact_exists(
                model_name=model_name,
                track=payload.track,
                model_version=model_version,
            )
    if score_source == "model" and payload.require_model_artifact and not model_artifact_ok:
        agg = {
            "status": "failed",
            "reason": "model_artifact_missing",
            "run_source": payload.run_source,
            "data_regime": data_regime,
            "score_source": score_source,
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
            "model_inference_coverage": 0.0,
            "fallback_used": False,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
            "gate_passed": False,
            "universe_resolve": universe_resolve,
            "strict_asof_fail_fast": bool(strict_asof_fail_fast),
            "degraded": False,
            "degraded_reasons": [],
            "missing_model_targets": missing_model_targets[:50],
        }
        repo.finish_backtest_run(run_id, agg)
        BACKTEST_FAILED_RUNS_TOTAL.labels(track=payload.track, reason=str(agg["reason"])).inc()
        METRIC_GATE_STATUS.labels(track=payload.track, metric="backtest_completed").set(0.0)
        return BacktestRunResponse(
            run_id=run_id,
            run_name=run_name,
            track=payload.track,
            status="failed",
            metrics=agg,
            config=config,
        )

    all_metrics: List[Dict[str, float]] = []
    calibration_ratio = float(payload.train_days) / float(max(1, payload.lookback_days))
    polarity_mode = str(payload.signal_polarity_mode or "normal").strip().lower()
    for target in targets:
        feature_rows = repo.load_feature_history(
            target=target,
            track=payload.track,
            lookback_days=payload.lookback_days,
            data_version=payload.data_version,
        )
        price_rows = repo.load_price_history(target, lookback_days=payload.lookback_days)
        if score_source == "model":
            m = _run_model_inference_backtest(
                target=target,
                feature_rows=feature_rows,
                price_rows=price_rows,
                fee_bps=effective_fee_bps,
                slippage_bps=effective_slippage_bps,
                sizing_override=sizing_override or None,
                signal_polarity_mode=polarity_mode,
                calibration_ratio=calibration_ratio,
                alignment_mode=payload.alignment_mode,
                max_feature_staleness_hours=payload.max_feature_staleness_hours,
                alignment_version=payload.alignment_version,
                strict_asof_fail_fast=strict_asof_fail_fast,
                model_name=model_name,
                model_version=model_version,
            )
        else:
            m = _run_model_replay_backtest(
                feature_rows=feature_rows,
                price_rows=price_rows,
                fee_bps=effective_fee_bps,
                slippage_bps=effective_slippage_bps,
                sizing_override=sizing_override or None,
                signal_polarity_mode=polarity_mode,
                calibration_ratio=calibration_ratio,
                alignment_mode=payload.alignment_mode,
                max_feature_staleness_hours=payload.max_feature_staleness_hours,
                alignment_version=payload.alignment_version,
                strict_asof_fail_fast=strict_asof_fail_fast,
            )
            m["score_source"] = "heuristic"
            m["model_inference_coverage"] = 0.0
            m["fallback_used"] = False
        m["target"] = target
        all_metrics.append(m)

    samples = sum(int(m.get("samples", 0)) for m in all_metrics)
    alignment_modes = sorted(
        {
            str((m.get("alignment_audit") or {}).get("alignment_mode_applied") or "").strip()
            for m in all_metrics
            if str((m.get("alignment_audit") or {}).get("alignment_mode_applied") or "").strip()
        }
    )
    if not alignment_modes:
        alignment_mode_applied = str(payload.alignment_mode)
    elif len(alignment_modes) == 1:
        alignment_mode_applied = alignment_modes[0]
    else:
        alignment_mode_applied = f"mixed:{','.join(alignment_modes)}"
    if samples == 0:
        failed_reasons = [str(m.get("reason") or "") for m in all_metrics if str(m.get("reason") or "").strip()]
        strict_fail = next((r for r in failed_reasons if r.startswith("strict_asof_alignment_fail_fast:")), "")
        if strict_fail:
            agg_reason = strict_fail
        elif any(r == "insufficient_aligned_samples" for r in failed_reasons):
            agg_reason = "insufficient_aligned_samples"
        elif any(r == "insufficient_features" for r in failed_reasons):
            agg_reason = "insufficient_features"
        else:
            agg_reason = "insufficient_prices"
        agg = {
            "status": "failed",
            "reason": agg_reason,
            "run_source": payload.run_source,
            "data_regime": data_regime,
            "score_source": score_source,
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
            "model_inference_coverage": 0.0,
            "fallback_used": False,
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
            "strict_asof_fail_fast": bool(strict_asof_fail_fast),
            "failed_target_reasons": failed_reasons[:20],
            "alignment_audit": {
                "alignment_mode_requested": payload.alignment_mode,
                "alignment_mode_applied": alignment_mode_applied,
                "alignment_version": payload.alignment_version,
                "total_price_steps": int(sum(int((m.get("alignment_audit") or {}).get("total_price_steps", 0) or 0) for m in all_metrics)),
                "aligned_steps": int(sum(int((m.get("alignment_audit") or {}).get("aligned_steps", 0) or 0) for m in all_metrics)),
                "dropped_missing_feature": int(sum(int((m.get("alignment_audit") or {}).get("dropped_missing_feature", 0) or 0) for m in all_metrics)),
                "dropped_stale_feature": int(sum(int((m.get("alignment_audit") or {}).get("dropped_stale_feature", 0) or 0) for m in all_metrics)),
                "dropped_missing_timestamps": int(sum(int((m.get("alignment_audit") or {}).get("dropped_missing_timestamps", 0) or 0) for m in all_metrics)),
                "leakage_violations": int(sum(int((m.get("alignment_audit") or {}).get("leakage_violations", 0) or 0) for m in all_metrics)),
                "fail_fast_count": int(sum(1 for m in all_metrics if bool((m.get("alignment_audit") or {}).get("fail_fast", False)))),
            },
            "leakage_checks": {
                "passed": bool(all(bool((m.get("leakage_checks") or {}).get("passed", False)) for m in all_metrics)),
                "leakage_violations": int(sum(int((m.get("leakage_checks") or {}).get("leakage_violations", 0) or 0) for m in all_metrics)),
                "alignment_mode": alignment_mode_applied,
                "alignment_version": payload.alignment_version,
            },
            "gate_passed": False,
            "universe_resolve": universe_resolve,
        }
        agg["degraded"] = bool(not agg["leakage_checks"]["passed"] or strict_asof_fail_fast)
        agg["degraded_reasons"] = [
            reason
            for reason in [
                "leakage_or_alignment_failed" if not agg["leakage_checks"]["passed"] else "",
                "strict_asof_hard_fail_enabled" if strict_asof_fail_fast else "",
            ]
            if reason
        ]
    else:
        per_target: Dict[str, Dict[str, float]] = {}
        regime_agg: Dict[str, Dict[str, float]] = {}
        for m in all_metrics:
            target = str(m.get("target") or "").upper()
            if not target:
                continue
            per_target[target] = {
                "pnl_after_cost": round(float(m.get("pnl_after_cost", 0.0) or 0.0), 6),
                "max_drawdown": round(float(m.get("max_drawdown", 0.0) or 0.0), 6),
                "sharpe": round(float(m.get("sharpe_daily", m.get("sharpe", 0.0)) or 0.0), 6),
                "sharpe_step_raw": round(float(m.get("sharpe_step_raw", 0.0) or 0.0), 6),
                "sharpe_daily": round(float(m.get("sharpe_daily", 0.0) or 0.0), 6),
                "observation_days": int(m.get("observation_days", 0) or 0),
                "vol_floor_applied": bool(m.get("vol_floor_applied", False)),
                "samples": int(m.get("samples", 0) or 0),
                "leakage_violations": int(((m.get("leakage_checks") or {}).get("leakage_violations", 0) or 0)),
                "alignment_mode": str(((m.get("leakage_checks") or {}).get("alignment_mode") or "")),
            }
            rb = m.get("regime_breakdown") or {}
            if isinstance(rb, dict):
                for name, vals in rb.items():
                    if not isinstance(vals, dict):
                        continue
                    cur = regime_agg.setdefault(
                        str(name),
                        {"samples": 0.0, "wins_weighted": 0.0, "turnover": 0.0, "pnl_after_cost": 0.0},
                    )
                    cnt = float(vals.get("samples") or 0.0)
                    hit = float(vals.get("hit_rate") or 0.0)
                    cur["samples"] += cnt
                    cur["wins_weighted"] += hit * cnt
                    cur["turnover"] += float(vals.get("turnover") or 0.0)
                    cur["pnl_after_cost"] += float(vals.get("pnl_after_cost") or 0.0)
        agg = {
            "status": "completed",
            "run_source": payload.run_source,
            "data_regime": data_regime,
            "score_source": score_source,
            "targets": targets,
            "samples": int(samples),
            "ic": round(float(np.mean([m["ic"] for m in all_metrics])), 6),
            "sharpe": round(float(np.mean([float(m.get("sharpe_daily", m.get("sharpe", 0.0)) or 0.0) for m in all_metrics])), 6),
            "sharpe_step_raw": round(float(np.mean([float(m.get("sharpe_step_raw", 0.0) or 0.0) for m in all_metrics])), 6),
            "sharpe_daily": round(float(np.mean([float(m.get("sharpe_daily", 0.0) or 0.0) for m in all_metrics])), 6),
            "sharpe_method": "daily_agg_v1",
            "observation_days": int(min([int(m.get("observation_days", 0) or 0) for m in all_metrics] or [0])),
            "vol_floor_applied": bool(any(bool(m.get("vol_floor_applied", False)) for m in all_metrics)),
            "hit_rate": round(float(np.mean([m["hit_rate"] for m in all_metrics])), 6),
            "turnover": round(float(np.mean([m["turnover"] for m in all_metrics])), 6),
            "pnl_after_cost": round(float(np.mean([m["pnl_after_cost"] for m in all_metrics])), 6),
            "max_drawdown": round(float(np.mean([m["max_drawdown"] for m in all_metrics])), 6),
            "model_name": model_name,
            "model_version": model_version,
            "lineage_coverage": round(float(np.mean([float(m.get("lineage_coverage", 0.0)) for m in all_metrics])), 6),
            "model_inference_coverage": round(float(np.mean([float(m.get("model_inference_coverage", 0.0) or 0.0) for m in all_metrics])), 6),
            "fallback_used": bool(any(bool(m.get("fallback_used", False)) for m in all_metrics)),
            "strict_asof_fail_fast": bool(strict_asof_fail_fast),
            "cost_breakdown": {
                "fee": round(float(np.mean([float((m.get("cost_breakdown") or {}).get("fee", 0.0)) for m in all_metrics])), 8),
                "slippage": round(float(np.mean([float((m.get("cost_breakdown") or {}).get("slippage", 0.0)) for m in all_metrics])), 8),
                "impact": round(float(np.mean([float((m.get("cost_breakdown") or {}).get("impact", 0.0)) for m in all_metrics])), 8),
            },
            "per_target": per_target,
            "regime_breakdown": {
                name: {
                    "samples": int(vals["samples"]),
                    "hit_rate": round(float(vals["wins_weighted"] / max(1.0, vals["samples"])), 6),
                    "turnover": round(float(vals["turnover"]), 6),
                    "pnl_after_cost": round(float(vals["pnl_after_cost"]), 6),
                }
                for name, vals in regime_agg.items()
                if vals["samples"] > 0
            },
            "alignment_audit": {
                "alignment_mode_requested": payload.alignment_mode,
                "alignment_mode_applied": alignment_mode_applied,
                "alignment_version": payload.alignment_version,
                "total_price_steps": int(sum(int((m.get("alignment_audit") or {}).get("total_price_steps", 0) or 0) for m in all_metrics)),
                "aligned_steps": int(sum(int((m.get("alignment_audit") or {}).get("aligned_steps", 0) or 0) for m in all_metrics)),
                "dropped_missing_feature": int(sum(int((m.get("alignment_audit") or {}).get("dropped_missing_feature", 0) or 0) for m in all_metrics)),
                "dropped_stale_feature": int(sum(int((m.get("alignment_audit") or {}).get("dropped_stale_feature", 0) or 0) for m in all_metrics)),
                "dropped_missing_timestamps": int(sum(int((m.get("alignment_audit") or {}).get("dropped_missing_timestamps", 0) or 0) for m in all_metrics)),
                "leakage_violations": int(sum(int((m.get("alignment_audit") or {}).get("leakage_violations", 0) or 0) for m in all_metrics)),
            },
            "leakage_checks": {
                "passed": bool(all(bool((m.get("leakage_checks") or {}).get("passed", False)) for m in all_metrics)),
                "leakage_violations": int(sum(int((m.get("leakage_checks") or {}).get("leakage_violations", 0) or 0) for m in all_metrics)),
                "alignment_mode": alignment_mode_applied,
                "alignment_version": payload.alignment_version,
            },
            "universe_resolve": universe_resolve,
        }
        degraded_reasons: List[str] = []
        if bool(agg.get("fallback_used", False)):
            degraded_reasons.append("legacy_or_heuristic_fallback_used")
        if not bool((agg.get("leakage_checks") or {}).get("passed", False)):
            degraded_reasons.append("leakage_or_alignment_failed")
        if float(agg.get("model_inference_coverage", 0.0) or 0.0) < 1.0:
            degraded_reasons.append("model_inference_partial_coverage")
        agg["degraded"] = bool(degraded_reasons)
        agg["degraded_reasons"] = sorted(set(degraded_reasons))
        agg["gate_passed"] = False

    MODEL_COVERAGE_RATIO.labels(track=payload.track).set(float(agg.get("model_inference_coverage", 0.0) or 0.0))
    if str(agg.get("reason") or "").startswith("strict_asof_alignment_fail_fast:"):
        STRICT_ASOF_FAIL_COUNT.labels(track=payload.track).inc()
    repo.finish_backtest_run(run_id, agg)
    if str(agg.get("status")) == "failed":
        BACKTEST_FAILED_RUNS_TOTAL.labels(track=payload.track, reason=str(agg.get("reason") or "unknown")).inc()
        METRIC_GATE_STATUS.labels(track=payload.track, metric="backtest_completed").set(0.0)
    else:
        METRIC_GATE_STATUS.labels(track=payload.track, metric="backtest_completed").set(1.0)

    gate_model_name = model_name
    gate_model_version = model_version
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=float(gate_cfg["min_ic"]),
        min_pnl_after_cost=float(gate_cfg["min_pnl_after_cost"]),
        max_drawdown=float(gate_cfg["max_drawdown"]),
        windows=int(gate_cfg["windows"]),
        score_source=score_source,
    )
    agg["gate_passed"] = bool(passed) if str(agg.get("status") or "").lower() == "completed" else False
    agg["gate_reason"] = str(reason)
    agg["gate_summary"] = dict(summary)
    agg["gate_windows_checked"] = int(checked)
    repo.finish_backtest_run(run_id, agg)
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
        created_at=_utcnow(),
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
        model_name = str(pred.get("model_name") or "liquid_ttm_ensemble")
        model_version = str(pred.get("model_version") or "v2.1")
    else:
        pred = _build_vc_prediction(payload.target, 12)
        model_name = "vc_survival_model"
        model_version = "v2.1"

    action = "hold"
    score = float(pred["score"])
    confidence = float(pred["confidence"])
    pred_outputs = pred.get("outputs") if isinstance(pred.get("outputs"), dict) else {}
    if payload.track == "liquid":
        mh = _liquid_multi_horizon_signal(
            pred_outputs=pred_outputs,
            horizon=payload.horizon,
            cost_profile=payload.cost_profile,
        )
        score = float(mh["score"])
        confidence = float(mh["confidence"])
        action = str(mh["action"])
        predicted_score = float(score)
        predicted_conf = float(confidence)
    else:
        mh = {}
        predicted_score = float(pred_outputs.get("expected_return", score) or score)
        predicted_conf = float(pred_outputs.get("signal_confidence", confidence) or confidence)
    divergence = abs(score - predicted_score) + abs(confidence - predicted_conf)
    SIGNAL_PREDICT_DIVERGENCE.labels(track=payload.track, target=payload.target.upper()).set(float(divergence))
    bucket = _strategy_bucket(payload.track, score, confidence, payload.target)

    if payload.policy == "ensemble-v1" and payload.track != "liquid":
        # slow model: historical/event score, fast model: recent price-volume pulse
        px = repo.latest_price_snapshot(payload.target)
        fast = 0.0
        if px:
            p = float(px.get("price") or 0.0)
            v = float(px.get("volume") or 0.0)
            fast = float(np.tanh(np.log1p(max(v, 0.0)) * 0.03 + np.log1p(max(p, 0.0)) * 0.01))
        score = round(0.7 * score + 0.3 * fast, 6)
        confidence = round(min(0.95, confidence + 0.03), 4)

    if payload.track == "liquid":
        if confidence < float(payload.min_confidence):
            action = "hold"
    else:
        if confidence >= payload.min_confidence:
            if score > 0.01:
                action = "buy"
            elif score < -0.01:
                action = "sell"

    reason = f"policy={payload.policy};model={model_name}:{model_version};score={round(score, 6)}"
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
            "score_horizons": dict(mh.get("score_horizons") or {}),
            "edge_horizons": dict(mh.get("edge_horizons") or {}),
            "cost_horizons": dict(mh.get("cost_horizons") or {}),
            "action_horizons": dict(mh.get("action_horizons") or {}),
            "selected_horizon": str(mh.get("selected_horizon") or payload.horizon),
            "expected_return_horizons": dict(mh.get("expected_return_horizons") or {}),
            "signal_confidence_horizons": dict(mh.get("signal_confidence_horizons") or {}),
            "vol_forecast_horizons": dict(mh.get("vol_forecast_horizons") or {}),
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
        created_at=_utcnow(),
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
    include_sources, exclude_sources = _run_source_filters()
    runs = repo.list_recent_backtest_runs(
        track=payload.track,
        limit=payload.max_recent_losses,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
    )
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


@router.get("/models/liquid/registry")
async def get_liquid_registry(symbol: str = Query("", description="optional symbol filter")):
    reg = load_liquid_registry()
    sym = str(symbol or "").strip().upper()
    if not sym:
        return reg
    symbols = reg.get("symbols") if isinstance(reg.get("symbols"), dict) else {}
    return {
        "status": reg.get("status", "ok"),
        "updated_at": reg.get("updated_at"),
        "symbol": sym,
        "entry": symbols.get(sym, {}),
    }


@router.post("/models/liquid/registry/activate")
async def activate_liquid_registry(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    horizon = str(payload.get("horizon") or "1h").strip().lower()
    model_name = str(payload.get("model_name") or "").strip()
    model_version = str(payload.get("model_version") or "").strip()
    actor = str(payload.get("operator") or "manual")
    if not symbol or not model_name or not model_version:
        raise HTTPException(status_code=400, detail="symbol/model_name/model_version required")
    out = upsert_liquid_active(symbol=symbol, horizon=horizon, model_name=model_name, model_version=model_version, actor=actor)
    return {"status": "ok", "symbol": symbol, "horizon": horizon, "entry": out}


@router.post("/models/liquid/registry/candidate")
async def set_liquid_registry_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    horizon = str(payload.get("horizon") or "1h").strip().lower()
    model_name = str(payload.get("model_name") or "").strip()
    model_version = str(payload.get("model_version") or "").strip()
    actor = str(payload.get("operator") or "manual")
    if not symbol or not model_name or not model_version:
        raise HTTPException(status_code=400, detail="symbol/model_name/model_version required")
    out = upsert_liquid_candidate(symbol=symbol, horizon=horizon, model_name=model_name, model_version=model_version, actor=actor)
    cand = get_liquid_candidate_model(symbol=symbol, horizon=horizon)
    return {"status": "ok", "symbol": symbol, "horizon": horizon, "candidate": cand, "entry": out}


@router.post("/models/liquid/registry/promote-candidate")
async def promote_liquid_registry_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    horizon = str(payload.get("horizon") or "1h").strip().lower()
    actor = str(payload.get("operator") or "manual")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")
    out = promote_liquid_candidate(symbol=symbol, horizon=horizon, actor=actor)
    return {"status": "ok", "symbol": symbol, "horizon": horizon, **out}


@router.post("/models/liquid/registry/rollback")
async def rollback_liquid_registry(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    horizon = str(payload.get("horizon") or "1h").strip().lower()
    actor = str(payload.get("operator") or "manual")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")
    out = rollback_liquid_active(symbol=symbol, horizon=horizon, actor=actor)
    return {"status": "ok", "symbol": symbol, "horizon": horizon, **out}


@router.post("/models/parity/check")
async def check_backtest_paper_parity(payload: ParityCheckRequest):
    res = _parity_check(
        track=payload.track,
        max_deviation=payload.max_deviation,
        min_completed_runs=payload.min_completed_runs,
        score_source=payload.score_source,
    )
    if not bool(res.get("passed", False)):
        repo.save_risk_event(
            decision_id=f"parity-{uuid.uuid4().hex[:12]}",
            severity="warning",
            code="backtest_paper_parity_failed",
            message=str(res.get("status") or "failed"),
            payload=res,
        )
    return res


@router.get("/risk/limits", response_model=RiskLimitsResponse)
async def get_risk_limits() -> RiskLimitsResponse:
    limits = {**_risk_limits(), **_risk_runtime_limits()}
    return RiskLimitsResponse(updated_at=_utcnow(), **limits)


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
        if "daily_loss_exceeded" not in violations:
            violations.append("daily_loss_exceeded")
        hard_block = True
    if payload.consecutive_losses > int(runtime_limits["max_consecutive_losses"]):
        if "consecutive_loss_exceeded" not in violations:
            violations.append("consecutive_loss_exceeded")
        hard_block = True
    if payload.latest_trade_edge_ratio is not None:
        latest_trade_edge = float(payload.latest_trade_edge_ratio)
        if latest_trade_edge <= -float(runtime_limits["single_trade_stop_loss_pct"]):
            if "single_trade_stop_loss_triggered" not in violations:
                violations.append("single_trade_stop_loss_triggered")
            hard_block = True
        if latest_trade_edge >= float(runtime_limits["single_trade_take_profit_pct"]):
            if "single_trade_take_profit_reached" not in violations:
                violations.append("single_trade_take_profit_reached")
    if payload.intraday_drawdown is not None and float(payload.intraday_drawdown) > float(runtime_limits["intraday_drawdown_halt_pct"]):
        if "intraday_drawdown_halt" not in violations:
            violations.append("intraday_drawdown_halt")
        hard_block = True
    if hard_block:
        hard_reasons = [
            v
            for v in violations
            if v in {"drawdown_exceeded", "daily_loss_exceeded", "consecutive_loss_exceeded", "single_trade_stop_loss_triggered", "intraday_drawdown_halt"}
        ]
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
    soft_only_violations = {"single_trade_take_profit_reached"} if _take_profit_mode() == "soft" else set()
    if kill_block:
        approved = False
        if kill_block not in violations:
            violations = [*violations, kill_block]
    elif any(v not in soft_only_violations for v in violations):
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
            updated_at=_utcnow(),
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
        updated_at=state.get("updated_at") or _utcnow(),
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
        updated_at=state.get("updated_at") or _utcnow(),
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
        updated_at=row.get("updated_at") or _utcnow(),
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
        updated_at=row.get("updated_at") or _utcnow(),
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

    allocator_mode = str(os.getenv("PORTFOLIO_ALLOCATOR_MODE", "allocator_v2")).strip().lower() or "allocator_v2"
    target_positions: List[RebalancePosition] = []
    bucket_violations: List[str] = []
    allocator_meta: Dict[str, Any] = {}
    if allocator_mode == "legacy":
        raw: List[Tuple[str, str, float, str]] = []
        sizing_cfg = _position_sizing_settings()
        for sig in payload.signals:
            side = 0.0
            if sig.action == "buy":
                side = 1.0
            elif sig.action == "sell":
                side = -1.0
            vol_bucket = _target_vol_bucket(sig.target)
            est_cost_bps = 8.0 if sig.track == "liquid" else 12.0
            size = _score_to_size(
                float(sig.score),
                float(sig.confidence),
                est_cost_bps=est_cost_bps,
                vol_bucket=vol_bucket,
                sizing_cfg=sizing_cfg,
            )
            strength = side * size
            bucket = _strategy_bucket(sig.track, float(sig.score), float(sig.confidence), sig.target)
            raw.append((sig.target.upper(), sig.track, strength, bucket))

        gross_strength = sum(abs(x[2]) for x in raw)
        if gross_strength <= 0:
            raise HTTPException(status_code=400, detail="no actionable signals")

        bucket_caps = _bucket_limits()
        pre_bucket_exposure: Dict[str, float] = {}
        for target, track, strength, bucket in raw:
            weight = (strength / gross_strength) * payload.risk_budget
            if abs(weight) < sizing_cfg["exit_z"]:
                weight = 0.0
            pre_bucket_exposure[bucket] = pre_bucket_exposure.get(bucket, 0.0) + abs(weight)
            target_positions.append(RebalancePosition(target=target, track=track, weight=weight, style_bucket=bucket))

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
    else:
        sig_inputs: List[AllocatorSignal] = []
        for sig in payload.signals:
            bucket = _strategy_bucket(sig.track, float(sig.score), float(sig.confidence), sig.target)
            sig_inputs.append(
                AllocatorSignal(
                    target=sig.target.upper(),
                    track=sig.track,
                    horizon=sig.horizon,
                    action=sig.action,
                    score=float(sig.score),
                    confidence=float(sig.confidence),
                    strategy_bucket=bucket,
                )
            )
        alloc = allocate_targets(
            signals=sig_inputs,
            risk_budget=float(payload.risk_budget),
            load_price_history=lambda target, lookback_days: repo.load_price_history(target, lookback_days=lookback_days),
        )
        weights = dict(alloc.get("weights") or {})
        allocator_meta = alloc
        if not weights:
            raise HTTPException(status_code=400, detail="no actionable signals")
        for sig in sig_inputs:
            if sig.target not in weights:
                continue
            # One target appears once in output position; preserve a representative bucket label for attribution.
            if any(p.target == sig.target for p in target_positions):
                continue
            target_positions.append(
                RebalancePosition(
                    target=sig.target,
                    track=sig.track,  # type: ignore[arg-type]
                    weight=float(weights.get(sig.target, 0.0) or 0.0),
                    style_bucket=f"{sig.strategy_bucket}:{sig.horizon}",
                )
            )

    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed=target_positions,
        current=payload.current_positions,
        realized_drawdown=payload.realized_drawdown,
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
                "metadata": {
                    "weight_delta": delta,
                    "allocator_mode": allocator_mode,
                    "allocator_contributions": (allocator_meta.get("contributions") or {}).get(target, {}),
                },
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
        allocator_details={
            "mode": allocator_mode,
            "bucket_exposure": dict(allocator_meta.get("bucket_exposure") or {}),
            "raw_strength": dict(allocator_meta.get("raw_strength") or {}),
            "dedupe_factor": dict(allocator_meta.get("dedupe_factor") or {}),
        },
    )


@router.post("/execution/orders", response_model=SubmitExecutionOrdersResponse)
async def submit_execution_orders(payload: SubmitExecutionOrdersRequest) -> SubmitExecutionOrdersResponse:
    if not payload.orders:
        raise HTTPException(status_code=400, detail="orders cannot be empty")
    resolved_venue = payload.venue
    if payload.adapter == "bitget_live" and payload.venue == "coinbase":
        resolved_venue = "bitget"
    if payload.adapter == "coinbase_live" and payload.venue == "bitget":
        resolved_venue = "coinbase"
    decision_id = uuid.uuid4().hex
    execution_params = {
        "market_type": payload.market_type,
        "product_type": payload.product_type,
        "leverage": payload.leverage,
        "reduce_only": payload.reduce_only,
        "position_mode": payload.position_mode,
        "margin_mode": payload.margin_mode,
    }
    order_payloads = [
        {
            "target": o.target.upper(),
            "track": o.track,
            "side": o.side,
            "quantity": o.quantity,
            "est_price": o.est_price,
            "strategy_id": o.strategy_id,
            "metadata": {**o.metadata, "execution_params": execution_params},
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
        venue=resolved_venue,
        time_in_force=payload.time_in_force,
        max_slippage_bps=payload.max_slippage_bps,
        orders=order_payloads,
    )
    return SubmitExecutionOrdersResponse(
        decision_id=decision_id,
        adapter=payload.adapter,
        venue=resolved_venue,
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
    resolved_venue = payload.venue
    if payload.adapter == "bitget_live" and payload.venue == "coinbase":
        resolved_venue = "bitget"
    if payload.adapter == "coinbase_live" and payload.venue == "bitget":
        resolved_venue = "coinbase"
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
    market_fit_violations = _order_market_fit_violations(scaled_orders, payload.market_type)
    if market_fit_violations:
        for code in market_fit_violations:
            repo.save_risk_event(
                decision_id=f"risk-precheck-{uuid.uuid4().hex[:12]}",
                severity="warning",
                code=code,
                message="execution market adaptation precheck blocked",
                payload={"decision_id": payload.decision_id, "market_type": payload.market_type},
            )
        raise HTTPException(status_code=423, detail=f"risk_blocked:{','.join(market_fit_violations)}")
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
    max_strategy_streak = 0
    pair_latest_edges: List[Tuple[str, str, float]] = []
    pair_intraday_drawdowns: List[Tuple[str, str, float]] = []
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
        max_strategy_streak = max(max_strategy_streak, streak)
        pair_latest_edges.append((pair_track, pair_strategy, _infer_latest_trade_edge_ratio(track=pair_track, strategy_id=pair_strategy)))
        pair_intraday_drawdowns.append((pair_track, pair_strategy, _infer_intraday_drawdown_ratio(track=pair_track, strategy_id=pair_strategy)))
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
    if pair_latest_edges:
        tp_track, tp_strategy, best_edge_ratio = max(pair_latest_edges, key=lambda x: x[2])
        if best_edge_ratio >= float(runtime_limits["single_trade_take_profit_pct"]):
            code = f"single_trade_take_profit_reached:{tp_strategy}"
            repo.save_risk_event(
                decision_id=f"risk-precheck-{uuid.uuid4().hex[:12]}",
                severity="warning",
                code=code,
                message="execution take-profit precheck triggered",
                payload={"decision_id": payload.decision_id, "edge_ratio": best_edge_ratio, "track": tp_track},
            )
            if _take_profit_mode() == "hard":
                raise HTTPException(status_code=423, detail=f"risk_blocked:{code}")
    # enforce risk check before execution; execution is blocked if risk fails.
    inferred_positions = _infer_execution_risk_positions(
        scaled_orders,
        risk_equity_usd=payload.risk_equity_usd,
    )
    track = str(scaled_orders[0].get("track") or "liquid")
    realized_daily_loss = _infer_daily_loss_ratio(track=track)
    best_latest_edge_ratio = max((x[2] for x in pair_latest_edges), default=0.0)
    max_intraday_drawdown = max((x[2] for x in pair_intraday_drawdowns), default=0.0)
    risk_resp = await risk_check(
        RiskCheckRequest(
            proposed_positions=inferred_positions,
            current_positions=[],
            realized_drawdown=0.0,
            daily_loss=realized_daily_loss,
            consecutive_losses=max_strategy_streak,
            latest_trade_edge_ratio=best_latest_edge_ratio,
            intraday_drawdown=max_intraday_drawdown,
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
            "venue": resolved_venue,
            "market_type": payload.market_type,
            "product_type": payload.product_type,
            "leverage": payload.leverage,
            "reduce_only": payload.reduce_only,
            "position_mode": payload.position_mode,
            "margin_mode": payload.margin_mode,
            "limit_timeout_sec": payload.limit_timeout_sec,
            "max_retries": payload.max_retries,
            "fee_bps": payload.fee_bps,
        },
    )
    filled = 0
    rejected = 0
    reject_breakdown: Dict[str, int] = {}
    merged = []
    for order, res in zip(scaled_orders, results):
        normalized_res = _normalize_execution_payload(res)
        exec_trace = build_execution_trace(order, normalized_res)
        status = str(normalized_res.get("status") or "rejected")
        EXECUTION_ORDERS_TOTAL.labels(adapter=payload.adapter, status=status).inc()
        if status in {"filled", "partially_filled"}:
            filled += 1
        elif status == "rejected":
            rejected += 1
            reason_cat = str(normalized_res.get("reject_reason_category") or "other")
            reject_breakdown[reason_cat] = reject_breakdown.get(reason_cat, 0) + 1
            EXECUTION_REJECTS_TOTAL.labels(adapter=payload.adapter, reason=reason_cat).inc()
            reason_raw = str(normalized_res.get("reject_reason") or "").lower()
            if reason_cat == "other" and "bitget" in reason_raw:
                BITGET_UNCLASSIFIED_ERRORS_TOTAL.labels(adapter=payload.adapter).inc()
        repo.update_order_execution(
            order["id"],
            status=status,
            metadata={
                "execution": normalized_res,
                "execution_trace": exec_trace,
                "execution_policy": str(normalized_res.get("execution_policy") or ""),
            },
        )
        merged.append(
            {
                **order,
                "execution": normalized_res,
                "execution_trace": exec_trace,
                "execution_policy": str(normalized_res.get("execution_policy") or ""),
            }
        )

    resp = ExecuteOrdersResponse(
        decision_id=payload.decision_id,
        adapter=payload.adapter,
        total=len(orders),
        filled=filled,
        rejected=rejected,
        reject_breakdown=reject_breakdown,
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
        generated_at=_utcnow(),
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
        evaluated_at=_utcnow(),
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
    include_sources, exclude_sources = _run_source_filters()
    data_regimes = _data_regime_filters()
    runs = repo.list_recent_backtest_runs(
        track=payload.track,
        limit=payload.windows,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        data_regimes=data_regimes,
    )
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
            updated_at=_utcnow(),
        )
    return RolloutStateResponse(
        track=track,  # type: ignore[arg-type]
        model_name=str(state.get("model_name") or _default_model_by_track(track)[0]),
        model_version=str(state.get("model_version") or _default_model_by_track(track)[1]),
        stage_pct=int(state.get("stage_pct") or 10),
        status=str(state.get("status") or "shadow"),
        hard_limits=state.get("hard_limits") or {},
        metrics=state.get("metrics") or {},
        updated_at=state.get("updated_at") or _utcnow(),
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
        generated_at=_utcnow(),
    )


@router.post("/tasks/pnl-attribution", response_model=AsyncTaskSubmitResponse)
async def submit_pnl_attribution_task(payload: PnLAttributionRequest) -> AsyncTaskSubmitResponse:
    task_id = enqueue_task("pnl_attribution", payload.model_dump())
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type="pnl_attribution",
        status="queued",
        created_at=_utcnow(),
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
        created_at=item.get("created_at") or _utcnow(),
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
        generated_at=_utcnow(),
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
        generated_at=_utcnow(),
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
