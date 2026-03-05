#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.financial_analysis_agent import FinancialAnalysisAgent, FinancialAnalysisConfig
from backend.decision_trace.models import DecisionTrace
from cost.cost_profile import compute_cost_breakdown_bps, load_cost_profile


def main() -> int:
    ap = argparse.ArgumentParser(description="Run decision smoke from inference output and emit DecisionTrace")
    ap.add_argument("--inference-json", default="artifacts/smoke/inference_smoke.json")
    ap.add_argument("--horizon", default="1h")
    ap.add_argument("--cost-profile", default="standard")
    ap.add_argument("--out-json", default="artifacts/smoke/decision_trace_smoke.json")
    args = ap.parse_args()

    pred = json.loads(Path(str(args.inference_json)).read_text(encoding="utf-8"))
    symbol = str(pred.get("symbol") or "UNKNOWN").upper()
    h = str(args.horizon).lower()
    mu_map = dict((pred.get("predictions") or {}).get("mu") or {})
    sigma_map = dict((pred.get("predictions") or {}).get("sigma") or {})
    q_map = dict((pred.get("predictions") or {}).get("quantiles") or {})
    d_map = dict((pred.get("predictions") or {}).get("direction_prob") or {})
    ew_map = dict((pred.get("predictions") or {}).get("expert_weights") or {})
    rp_map = dict((pred.get("predictions") or {}).get("regime_probs") or {})
    mu = float(mu_map.get(h, 0.0) or 0.0)
    sigma = float(sigma_map.get(h, 0.0) or 0.0)
    qh = dict(q_map.get(h) or {})
    risk = max(1e-8, sigma, abs(float(qh.get("p90", 0.0) or 0.0) - float(qh.get("p10", 0.0) or 0.0))
    )

    cost_bd = compute_cost_breakdown_bps(
        horizon=h,
        profile=load_cost_profile(str(args.cost_profile)),
        market_state={"realized_vol": sigma},
        liquidity_features={"liquidity_score": 0.8},
        turnover_estimate=0.5,
    )
    edge = mu - float(cost_bd["total_bps"]) / 10000.0
    score = edge / risk
    band = 1.0
    action = "buy" if score >= band else ("sell" if score <= -band else "hold")

    agent = FinancialAnalysisAgent(FinancialAnalysisConfig(enabled=True, router_hint_enabled=False))
    analyst = agent.analyze(
        model_pred={
            "mu": mu,
            "sigma": sigma,
            "quantiles": qh,
            "direction": float(d_map.get(h, 0.0) or 0.0),
        },
        expert_weights=dict(ew_map.get(h) or {}),
        regime_features={"spread_proxy": 0.0, "depth_proxy": 0.0, "liquidation_proxy": 0.0},
        cost_profile={
            "fee_bps": float(cost_bd["fee_bps"]),
            "slippage_bps": float(cost_bd["slippage_bps"]),
            "impact_bps": float(cost_bd["impact_bps"]),
            "funding_bps": float(cost_bd["funding_bps"]),
            "infra_bps": float(cost_bd["infra_bps"]),
        },
        account_state={"equity": 10000.0, "free_margin": 8000.0},
        position_state={"symbol": symbol, "qty": 0.0},
        recent_trades_summary={"count": 0},
        ts=datetime.now(timezone.utc).isoformat(),
    )

    trace = DecisionTrace(
        decision_id=f"smoke-{uuid.uuid4().hex[:12]}",
        symbol=symbol,
        action=action,
        target_pos=0.2 if action == "buy" else (-0.2 if action == "sell" else 0.0),
        delta_pos=0.2 if action in {"buy", "sell"} else 0.0,
        mu={h: mu},
        sigma={h: sigma},
        quantiles={h: qh},
        direction_prob={h: float(d_map.get(h, 0.0) or 0.0)},
        expert_weights={h: dict(ew_map.get(h) or {})},
        regime_probs={h: dict(rp_map.get(h) or {})},
        cost={h: float(cost_bd["total_bps"]) / 10000.0},
        cost_breakdown={
            h: {
                "fee": float(cost_bd["fee_bps"]) / 10000.0,
                "slippage": float(cost_bd["slippage_bps"]) / 10000.0,
                "impact": float(cost_bd["impact_bps"]) / 10000.0,
                "funding": float(cost_bd["funding_bps"]) / 10000.0,
                "infra": float(cost_bd["infra_bps"]) / 10000.0,
                "total": float(cost_bd["total_bps"]) / 10000.0,
            }
        },
        score={h: float(score)},
        thresholds={h: float(band)},
        risk={"mode": "smoke"},
        account={"equity": 10000.0},
        position={"target_pos": 0.2 if action != "hold" else 0.0},
        analyst=analyst,
        reason_codes=["smoke_test"],
    )
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out_json": str(out_path), "action": action, "score": score}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
