from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


EntityType = Literal["company", "investor", "asset"]
EventType = Literal["funding", "mna", "product", "regulatory", "market"]
TrackType = Literal["vc", "liquid"]


class Entity(BaseModel):
    id: Optional[int] = None
    entity_type: EntityType
    name: str
    symbol: Optional[str] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    id: Optional[int] = None
    event_type: EventType
    title: str
    occurred_at: datetime
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    source_timezone: str = "UTC"
    source_tier: int = Field(default=3, ge=1, le=5)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    event_importance: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    entity_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    latency_ms: Optional[int] = Field(default=None, ge=0)
    dedup_cluster_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    entities: List[Entity] = Field(default_factory=list)


class IngestEventsRequest(BaseModel):
    events: List[Event]


class IngestEventsResponse(BaseModel):
    accepted: int
    inserted: int
    deduplicated: int
    event_ids: List[int]


class FeatureVector(BaseModel):
    entity_id: Optional[int] = None
    symbol: Optional[str] = None
    as_of: datetime
    features: Dict[str, float]
    feature_version: str


class VCPredictRequest(BaseModel):
    company_name: str
    horizon_months: Literal[6, 12, 24] = 12
    country: Optional[str] = None
    sector: Optional[str] = None


class LiquidPredictRequest(BaseModel):
    symbol: str
    horizon: Literal["1h", "1d", "7d"] = "1d"


class PredictionExplanation(BaseModel):
    top_event_contributors: List[Dict[str, Any]] = Field(default_factory=list)
    top_feature_contributors: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_links: List[str] = Field(default_factory=list)
    model_version: str
    feature_version: str


class Prediction(BaseModel):
    id: Optional[int] = None
    track: TrackType
    target: str
    score: float
    confidence: float
    outputs: Dict[str, Any]
    created_at: datetime
    explanation: PredictionExplanation


class PortfolioPosition(BaseModel):
    target: str
    track: TrackType
    score: float
    risk: float = Field(ge=0.0)


class PortfolioScoreRequest(BaseModel):
    positions: List[PortfolioPosition]
    risk_budget: float = Field(default=1.0, gt=0.0)


class PortfolioScoreResponse(BaseModel):
    alpha_score: float
    expected_return: float
    expected_volatility: float
    recommendations: List[Dict[str, Any]]


class BacktestRunRequest(BaseModel):
    track: TrackType
    targets: List[str] = Field(default_factory=list)
    horizon: Literal["1h", "1d", "7d"] = "1d"
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    data_version: str = Field(default="v1", min_length=1, max_length=64)
    lookback_days: int = Field(default=90, ge=14, le=730)
    train_days: int = Field(default=35, ge=7, le=365)
    test_days: int = Field(default=7, ge=1, le=90)
    fee_bps: float = Field(default=5.0, ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=3.0, ge=0.0, le=1000.0)


class PnLAttributionRequest(BaseModel):
    track: TrackType = "liquid"
    lookback_hours: int = Field(default=24 * 7, ge=1, le=24 * 365)


class BacktestRunResponse(BaseModel):
    run_id: int
    run_name: str
    track: TrackType
    status: Literal["completed", "failed", "running"]
    metrics: Dict[str, Any]
    config: Dict[str, Any]


class AsyncTaskSubmitResponse(BaseModel):
    task_id: str
    task_type: str
    status: Literal["queued"]
    created_at: datetime


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SignalGenerateRequest(BaseModel):
    track: TrackType
    target: str
    horizon: Literal["1h", "1d", "7d"] = "1d"
    policy: str = "baseline-v2"
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    strategy_id: str = Field(default="default-liquid-v1", min_length=1, max_length=64)
    cost_profile: str = Field(default="standard", min_length=1, max_length=64)
    risk_profile: str = Field(default="balanced", min_length=1, max_length=64)


class SignalGenerateResponse(BaseModel):
    signal_id: int
    track: TrackType
    target: str
    horizon: str
    action: Literal["buy", "sell", "hold"]
    score: float
    confidence: float
    reason: str
    policy: str
    strategy_bucket: Literal["trend", "event", "mean_reversion"]
    created_at: datetime


class SignalInput(BaseModel):
    target: str
    track: TrackType
    action: Literal["buy", "sell", "hold"]
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: Literal["1h", "1d", "7d"] = "1d"


class RebalancePosition(BaseModel):
    target: str
    weight: float
    track: TrackType
    sector: Optional[str] = None
    style_bucket: Optional[str] = None


class PortfolioRebalanceRequest(BaseModel):
    signals: List[SignalInput]
    current_positions: List[RebalancePosition] = Field(default_factory=list)
    capital: float = Field(default=1.0, gt=0.0)
    risk_budget: float = Field(default=1.0, gt=0.0)


class PortfolioRebalanceResponse(BaseModel):
    decision_id: str
    target_positions: List[RebalancePosition]
    expected_turnover: float
    orders: List[Dict[str, Any]]
    risk_ok: bool
    risk_violations: List[str]


class RiskLimitsResponse(BaseModel):
    max_single_weight: float
    max_gross_exposure: float
    max_turnover_per_rebalance: float
    max_realized_drawdown: float
    max_sector_exposure: float
    max_style_exposure: float
    updated_at: datetime


class RiskCheckRequest(BaseModel):
    proposed_positions: List[RebalancePosition]
    current_positions: List[RebalancePosition] = Field(default_factory=list)
    realized_drawdown: float = Field(default=0.0, ge=0.0)
    daily_loss: float = Field(default=0.0, ge=0.0)
    consecutive_losses: int = Field(default=0, ge=0)
    strategy_id: str = Field(default="global", min_length=1, max_length=64)
    max_sector_exposure_override: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    max_style_exposure_override: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class RiskCheckResponse(BaseModel):
    approved: bool
    violations: List[str]
    adjusted_positions: List[RebalancePosition]
    gross_exposure: float
    expected_turnover: float
    hard_block: bool = False
    kill_switch_state: Literal["armed", "triggered"] = "armed"
    risk_budget_used: float = Field(default=0.0, ge=0.0)


class KillSwitchTriggerRequest(BaseModel):
    track: TrackType
    strategy_id: str = Field(default="global", min_length=1, max_length=64)
    reason: str = Field(default="manual_trigger", min_length=1, max_length=256)
    duration_minutes: Optional[int] = Field(default=None, ge=1, le=60 * 24 * 7)


class KillSwitchResetRequest(BaseModel):
    track: TrackType
    strategy_id: str = Field(default="global", min_length=1, max_length=64)
    reason: str = Field(default="manual_reset", min_length=1, max_length=256)


class KillSwitchStateResponse(BaseModel):
    track: TrackType
    strategy_id: str
    state: Literal["armed", "triggered"] = "armed"
    reason: str
    triggered: bool
    updated_at: datetime
    expires_at: Optional[datetime] = None
    remaining_seconds: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionOpeningStatusResponse(BaseModel):
    track: TrackType
    strategy_id: str
    can_open_new_positions: bool
    state: Literal["armed", "triggered"] = "armed"
    block_reason: str = "none"
    updated_at: datetime
    expires_at: Optional[datetime] = None
    remaining_seconds: int = 0


class ModelGateRequest(BaseModel):
    track: TrackType
    model_name: str
    model_version: str
    min_ic: float = Field(default=0.0)
    min_pnl_after_cost: float = Field(default=0.0)
    max_drawdown: float = Field(default=0.2, gt=0.0)
    windows: int = Field(default=3, ge=1, le=20)


class ModelGateResponse(BaseModel):
    passed: bool
    track: TrackType
    model_name: str
    model_version: str
    reason: str
    windows_checked: int
    metrics_summary: Dict[str, Any]


class RollbackCheckRequest(BaseModel):
    track: TrackType
    model_name: str
    model_version: str
    max_recent_losses: int = Field(default=5, ge=1, le=50)
    min_recent_hit_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    max_recent_drawdown: float = Field(default=0.25, ge=0.0, le=1.0)


class RollbackCheckResponse(BaseModel):
    rollback_triggered: bool
    reason: str
    from_model: str
    to_model: str
    windows_failed: int = Field(default=0, ge=0)
    trigger_rule: str = "none"
    metrics: Dict[str, Any]


class ExecuteOrdersRequest(BaseModel):
    decision_id: str
    adapter: Literal["paper", "coinbase_live"] = "paper"
    time_in_force: Literal["GTC", "IOC", "FOK"] = "IOC"
    max_slippage_bps: float = Field(default=20.0, ge=0.0, le=2000.0)
    venue: str = Field(default="coinbase", min_length=1, max_length=64)
    max_orders: int = Field(default=100, ge=1, le=1000)
    limit_timeout_sec: float = Field(default=2.0, ge=0.1, le=30.0)
    max_retries: int = Field(default=1, ge=0, le=10)
    fee_bps: float = Field(default=5.0, ge=0.0, le=1000.0)


class ExecuteOrdersResponse(BaseModel):
    decision_id: str
    adapter: str
    total: int
    filled: int
    rejected: int
    orders: List[Dict[str, Any]]


class ExecutionOrderInput(BaseModel):
    target: str
    track: TrackType = "liquid"
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0.0)
    est_price: Optional[float] = Field(default=None, gt=0.0)
    strategy_id: str = Field(default="default-liquid-v1", min_length=1, max_length=64)
    decision_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubmitExecutionOrdersRequest(BaseModel):
    adapter: Literal["paper", "coinbase_live"] = "paper"
    venue: str = Field(default="coinbase", min_length=1, max_length=64)
    time_in_force: Literal["GTC", "IOC", "FOK"] = "IOC"
    max_slippage_bps: float = Field(default=20.0, ge=0.0, le=2000.0)
    orders: List[ExecutionOrderInput]


class SubmitExecutionOrdersResponse(BaseModel):
    decision_id: str
    adapter: str
    venue: str
    accepted_orders: int
    order_ids: List[int]


class ExecutionOrderStatusResponse(BaseModel):
    order_id: int
    decision_id: str
    target: str
    side: str
    quantity: float
    status: str
    track: str
    venue: str
    adapter: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradeAuditResponse(BaseModel):
    decision_id: str
    signals: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    pnl: Dict[str, Any]
    generated_at: datetime


class DriftEvaluateRequest(BaseModel):
    track: TrackType
    lookback_hours: int = Field(default=48, ge=1, le=24 * 30)
    reference_hours: int = Field(default=24 * 14, ge=24, le=24 * 180)
    psi_threshold: float = Field(default=0.25, ge=0.0, le=10.0)
    ks_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


class DriftEvaluateResponse(BaseModel):
    track: TrackType
    drift_detected: bool
    action: Literal["keep", "warn", "rollback_check"]
    metrics: Dict[str, float]
    reason: str
    evaluated_at: datetime


class AutoGateEvaluateRequest(BaseModel):
    track: TrackType
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    min_ic: float = Field(default=0.0)
    min_pnl_after_cost: float = Field(default=0.0)
    max_drawdown: float = Field(default=0.2, gt=0.0)
    windows: int = Field(default=3, ge=1, le=20)
    auto_promote: bool = True


class AutoGateEvaluateResponse(BaseModel):
    passed: bool
    promoted: bool
    track: TrackType
    model_name: str
    model_version: str
    reason: str
    windows_checked: int
    metrics_summary: Dict[str, Any]


class RolloutAdvanceRequest(BaseModel):
    track: TrackType
    model_name: str
    model_version: str
    current_stage_pct: int = Field(default=10, ge=10, le=100)
    next_stage_pct: int = Field(default=30, ge=10, le=100)
    min_hit_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    min_pnl_after_cost: float = Field(default=0.0)
    max_drawdown: float = Field(default=0.25, ge=0.0, le=1.0)
    windows: int = Field(default=3, ge=1, le=30)


class RolloutAdvanceResponse(BaseModel):
    track: TrackType
    model_name: str
    model_version: str
    current_stage_pct: int
    next_stage_pct: int
    promoted: bool
    reason: str
    hard_limits: Dict[str, float]
    metrics: Dict[str, float]


class RolloutStateResponse(BaseModel):
    track: TrackType
    model_name: str
    model_version: str
    stage_pct: int
    status: str
    hard_limits: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class PnLAttributionResponse(BaseModel):
    track: TrackType
    lookback_hours: int
    totals: Dict[str, float]
    by_target: List[Dict[str, Any]]
    generated_at: datetime


class SchedulerAuditLogRequest(BaseModel):
    track: TrackType
    action: str = Field(min_length=1, max_length=64)
    window: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    decision: Dict[str, Any] = Field(default_factory=dict)


class DataQualitySampleRequest(BaseModel):
    limit: int = Field(default=200, ge=1, le=1000)
    min_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DataQualityAuditUpdate(BaseModel):
    audit_id: int
    reviewer: str
    verdict: Literal["correct", "incorrect", "uncertain"]
    note: Optional[str] = None


class DataQualityStatsResponse(BaseModel):
    lookback_days: int
    totals: Dict[str, float]
    by_source: List[Dict[str, Any]]
    generated_at: datetime


class DataQualityConsistencyResponse(BaseModel):
    lookback_days: int
    total_review_logs: int
    multi_review_events: int
    pairwise_agreement: float
    reviewer_pairs: List[Dict[str, Any]]
    generated_at: datetime


class LineageConsistencyRequest(BaseModel):
    track: TrackType
    target: Optional[str] = None
    lineage_id: str = Field(min_length=6, max_length=64)
    data_version: Optional[str] = Field(default=None, min_length=1, max_length=64)
    strict: bool = True
    max_mismatch_keys: int = Field(default=20, ge=1, le=200)
    tolerance: float = Field(default=1e-6, ge=0.0, le=1.0)


class LineageConsistencyResponse(BaseModel):
    passed: bool
    track: TrackType
    target: Optional[str] = None
    lineage_id: str
    data_version: Optional[str] = None
    compared_snapshots: int
    max_abs_diff: float
    mean_abs_diff: float
    mismatch_keys: List[str] = Field(default_factory=list)
    reason: str
