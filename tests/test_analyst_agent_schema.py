from __future__ import annotations

import pytest

from backend.financial_analysis_agent import FinancialAnalysisAgent, FinancialAnalysisConfig, FinancialAnalysisOutput


def test_financial_analysis_agent_schema_and_hint_validation():
    agent = FinancialAnalysisAgent(FinancialAnalysisConfig(enabled=True, router_hint_enabled=True))
    out = agent.analyze(
        model_pred={"mu": 0.01, "sigma": 0.02, "quantiles": {"p10": -0.01, "p90": 0.03}, "direction": 0.6},
        expert_weights={"trend": 0.3, "mean_reversion": 0.2, "liquidation_risk": 0.4, "neutral": 0.1},
        regime_features={"trend_strength": 0.5, "funding_zscore": 2.5, "spread_proxy": 5.0, "liquidation_proxy": 0.9},
        cost_profile={"fee_bps": 5.0, "slippage_bps": 3.0, "impact_bps": 2.0, "funding_bps": 1.0, "infra_bps": 0.2},
        account_state={"equity": 10000.0},
        position_state={"qty": 0.0},
        recent_trades_summary={"count": 3},
        ts="2026-03-01T00:00:00+00:00",
    )
    parsed = FinancialAnalysisOutput.model_validate(out)
    assert parsed.analysis_summary
    assert parsed.cost_breakdown.total > 0
    assert parsed.suggested_risk_mode in {"aggressive", "neutral", "defensive"}
    assert 0.0 <= parsed.suggested_thresholds.pos_scale <= 1.0
    assert parsed.regime_hint is not None
    hint = FinancialAnalysisAgent.validate_router_hint(hint=parsed.regime_hint, expected_ts=parsed.ts)
    assert abs(sum(hint.values()) - 1.0) < 1e-6


def test_financial_analysis_output_forbids_extra_fields():
    payload = {
        "analysis_summary": "x",
        "risk_flags": [],
        "cost_breakdown": {"fee": 0, "slippage": 0, "impact": 0, "funding": 0, "infra": 0, "total": 0},
        "regime_view": {"regime_probs": {}, "key_features": {}},
        "suggested_risk_mode": "neutral",
        "suggested_thresholds": {"band": 1.0, "pos_scale": 0.8, "max_leverage": 1.0},
        "explanations": {"top_drivers": [], "counterfactuals": []},
        "prompt_hash": "p",
        "output_hash": "o",
        "ts": "2026-03-01T00:00:00+00:00",
        "extra_field_should_fail": 1,
    }
    with pytest.raises(Exception):
        FinancialAnalysisOutput.model_validate(payload)
