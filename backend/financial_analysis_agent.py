from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


RiskMode = Literal["aggressive", "neutral", "defensive"]


class RiskFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fee: float
    slippage: float
    impact: float
    funding: float
    infra: float
    total: float


class RegimeView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    regime_probs: Dict[str, float]
    key_features: Dict[str, float]


class SuggestedThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    band: float
    pos_scale: float
    max_leverage: float

    @field_validator("pos_scale")
    @classmethod
    def _validate_pos_scale(cls, v: float) -> float:
        out = float(v)
        if out < 0.0 or out > 1.0:
            raise ValueError("pos_scale_out_of_range")
        return out


class Explanations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_drivers: List[str] = Field(default_factory=list)
    counterfactuals: List[str] = Field(default_factory=list)


class FinancialAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_summary: str
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    cost_breakdown: CostBreakdown
    regime_view: RegimeView
    suggested_risk_mode: RiskMode
    suggested_thresholds: SuggestedThresholds
    explanations: Explanations
    regime_hint: Optional[Dict[str, float]] = None
    prompt_hash: str
    output_hash: str
    ts: str

    @field_validator("regime_hint")
    @classmethod
    def _validate_hint(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        keys = ("trend", "meanrev", "liquidation", "neutral")
        out = {k: float(v.get(k, 0.0) or 0.0) for k in keys}
        s = sum(out.values())
        if s <= 0:
            raise ValueError("regime_hint_sum_non_positive")
        out = {k: max(0.0, min(1.0, val / s)) for k, val in out.items()}
        return out


@dataclass(frozen=True)
class FinancialAnalysisConfig:
    enabled: bool = True
    router_hint_enabled: bool = False


class FinancialAnalysisAgent:
    def __init__(self, config: FinancialAnalysisConfig | None = None):
        self.config = config or FinancialAnalysisConfig()

    @staticmethod
    def _hash_payload(payload: Mapping[str, Any]) -> str:
        body = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def analyze(
        self,
        *,
        model_pred: Mapping[str, Any],
        expert_weights: Mapping[str, float] | None,
        regime_features: Mapping[str, float] | None,
        cost_profile: Mapping[str, Any] | None,
        account_state: Mapping[str, Any] | None,
        position_state: Mapping[str, Any] | None,
        recent_trades_summary: Mapping[str, Any] | None,
        ts: str | None = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("financial_analysis_agent_disabled")
        rp = dict(regime_features or {})
        cp = dict(cost_profile or {})
        pred = dict(model_pred or {})
        ew = dict(expert_weights or {})
        acct = dict(account_state or {})
        pos = dict(position_state or {})
        tr = dict(recent_trades_summary or {})
        now_ts = str(ts or self._now_iso())

        mu = self._safe_float(pred.get("mu"))
        sigma = max(1e-8, self._safe_float(pred.get("sigma")))
        q10 = self._safe_float((pred.get("quantiles") or {}).get("p10")) if isinstance(pred.get("quantiles"), dict) else 0.0
        q90 = self._safe_float((pred.get("quantiles") or {}).get("p90")) if isinstance(pred.get("quantiles"), dict) else 0.0
        uncertainty = max(sigma, abs(q90 - q10))

        fee = self._safe_float(cp.get("fee_bps", cp.get("fee", 0.0)))
        slippage = self._safe_float(cp.get("slippage_bps", cp.get("slippage", 0.0)))
        impact = self._safe_float(cp.get("impact_bps", cp.get("impact", 0.0)))
        funding = self._safe_float(cp.get("funding_bps", cp.get("funding", 0.0)))
        infra = self._safe_float(cp.get("infra_bps", cp.get("infra", 0.0)))
        total_cost = fee + slippage + impact + funding + infra

        liquidation_score = max(
            self._safe_float(rp.get("liquidation_proxy")),
            self._safe_float(ew.get("liquidation_risk", ew.get("liquidation", 0.0))),
        )
        spread_proxy = self._safe_float(rp.get("spread_proxy"))
        trend_score = self._safe_float(rp.get("trend_strength"))
        funding_z = abs(self._safe_float(rp.get("funding_zscore")))

        risk_flags: List[RiskFlag] = []
        if liquidation_score > 0.8:
            risk_flags.append(RiskFlag(code="LIQUIDATION_STRESS", severity="high", message="liquidation stress is elevated"))
        if spread_proxy > 15.0:
            risk_flags.append(RiskFlag(code="SPREAD_WIDE", severity="medium", message="spread proxy indicates worse execution"))
        if funding_z > 2.0:
            risk_flags.append(RiskFlag(code="CROWDING_EXTREME", severity="medium", message="funding zscore indicates crowding"))
        if total_cost > 30.0:
            risk_flags.append(RiskFlag(code="COST_HEAVY", severity="high", message="total transaction cost too high"))

        if risk_flags:
            mode: RiskMode = "defensive"
        elif mu > 0 and trend_score > 0:
            mode = "aggressive" if uncertainty < abs(mu) else "neutral"
        else:
            mode = "neutral"

        if mode == "aggressive":
            band, pos_scale, max_lev = 0.6, 1.0, 2.5
        elif mode == "defensive":
            band, pos_scale, max_lev = 1.4, 0.6, 1.2
        else:
            band, pos_scale, max_lev = 1.0, 1.0, 2.0

        top_drivers = [
            f"mu={mu:.6f}",
            f"sigma={sigma:.6f}",
            f"total_cost_bps={total_cost:.3f}",
            f"trend_strength={trend_score:.4f}",
            f"liquidation_proxy={liquidation_score:.4f}",
        ]
        counterfactuals = [
            "if spread_proxy drops 30%, trading band can tighten",
            "if funding_zscore normalizes, mean-reversion risk overlay can relax",
            "if uncertainty halves, position scale can increase",
        ]

        regime_probs = {
            "trend": max(0.0, min(1.0, self._safe_float(ew.get("trend", 0.0)))),
            "meanrev": max(0.0, min(1.0, self._safe_float(ew.get("mean_reversion", ew.get("meanrev", 0.0))))),
            "liquidation": max(0.0, min(1.0, self._safe_float(ew.get("liquidation_risk", ew.get("liquidation", 0.0))))),
            "neutral": max(0.0, min(1.0, self._safe_float(ew.get("neutral", 0.0)))),
        }
        s = sum(regime_probs.values())
        if s > 0:
            regime_probs = {k: float(v / s) for k, v in regime_probs.items()}
        else:
            regime_probs = {"trend": 0.25, "meanrev": 0.25, "liquidation": 0.25, "neutral": 0.25}

        regime_hint: Optional[Dict[str, float]] = None
        if self.config.router_hint_enabled:
            regime_hint = dict(regime_probs)

        prompt_obj = {
            "model_pred": pred,
            "expert_weights": ew,
            "regime_features": rp,
            "cost_profile": cp,
            "account_state": acct,
            "position_state": pos,
            "recent_trades_summary": tr,
            "ts": now_ts,
            "router_hint_enabled": self.config.router_hint_enabled,
        }
        prompt_hash = self._hash_payload(prompt_obj)
        out_obj = FinancialAnalysisOutput(
            analysis_summary=f"edge={mu:.6f}, uncertainty={uncertainty:.6f}, cost_bps={total_cost:.3f}, mode={mode}",
            risk_flags=risk_flags,
            cost_breakdown=CostBreakdown(
                fee=fee,
                slippage=slippage,
                impact=impact,
                funding=funding,
                infra=infra,
                total=total_cost,
            ),
            regime_view=RegimeView(
                regime_probs=regime_probs,
                key_features={
                    "trend_strength": trend_score,
                    "funding_zscore": self._safe_float(rp.get("funding_zscore")),
                    "spread_proxy": spread_proxy,
                    "depth_proxy": self._safe_float(rp.get("depth_proxy")),
                    "liquidation_proxy": self._safe_float(rp.get("liquidation_proxy")),
                },
            ),
            suggested_risk_mode=mode,
            suggested_thresholds=SuggestedThresholds(
                band=float(band),
                pos_scale=float(pos_scale),
                max_leverage=float(max_lev),
            ),
            explanations=Explanations(
                top_drivers=top_drivers,
                counterfactuals=counterfactuals,
            ),
            regime_hint=regime_hint,
            prompt_hash=prompt_hash,
            output_hash="",
            ts=now_ts,
        )
        output_hash = self._hash_payload(out_obj.model_dump(mode="json", exclude={"output_hash"}))
        out_obj.output_hash = output_hash
        return out_obj.model_dump(mode="json")

    @staticmethod
    def validate_router_hint(*, hint: Mapping[str, Any], expected_ts: str) -> Dict[str, float]:
        out = FinancialAnalysisOutput.model_validate(
            {
                "analysis_summary": "x",
                "risk_flags": [],
                "cost_breakdown": {"fee": 0, "slippage": 0, "impact": 0, "funding": 0, "infra": 0, "total": 0},
                "regime_view": {"regime_probs": {}, "key_features": {}},
                "suggested_risk_mode": "neutral",
                "suggested_thresholds": {"band": 1, "pos_scale": 1, "max_leverage": 1},
                "explanations": {"top_drivers": [], "counterfactuals": []},
                "regime_hint": dict(hint),
                "prompt_hash": "x",
                "output_hash": "x",
                "ts": expected_ts,
            }
        )
        if str(out.ts) != str(expected_ts):
            raise RuntimeError(f"agent_hint_stale:{out.ts}:{expected_ts}")
        assert out.regime_hint is not None
        return {k: float(v) for k, v in out.regime_hint.items()}
