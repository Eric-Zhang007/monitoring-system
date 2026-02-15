#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
if command -v jq >/dev/null 2>&1; then
  PRETTY="jq ."
else
  PRETTY="cat"
fi

echo "[1/17] health"
curl -fsS "${API_BASE}/health" | eval "$PRETTY"

echo "[2/17] ingest events"
INGEST_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/ingest/events" \
  -H 'Content-Type: application/json' \
  -d '{
    "events": [
      {
        "event_type": "funding",
        "title": "Test Startup raised Series A",
        "occurred_at": "2026-02-14T12:00:00Z",
        "source_url": "https://example.com/a",
        "source_name": "test-script",
        "source_timezone": "UTC",
        "source_tier": 2,
        "confidence_score": 0.88,
        "event_importance": 0.9,
        "novelty_score": 0.8,
        "entity_confidence": 0.85,
        "latency_ms": 500,
        "dedup_cluster_id": "test-cluster-a",
        "payload": {"amount_usd": 12000000},
        "entities": [
          {"entity_type": "company", "name": "Test Startup", "country": "US", "sector": "AI", "metadata": {}},
          {"entity_type": "investor", "name": "Test Capital", "country": "US", "sector": "VC", "metadata": {}}
        ]
      }
    ]
  }')
echo "$INGEST_RESP" | eval "$PRETTY"

echo "[3/17] vc predict"
VC_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/predict/vc" \
  -H 'Content-Type: application/json' \
  -d '{"company_name":"Test Startup","horizon_months":12}')
echo "$VC_RESP" | eval "$PRETTY"
PRED_ID=$(echo "$VC_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['prediction_id'])")

echo "[4/17] explanation"
curl -fsS "${API_BASE}/api/v2/predictions/${PRED_ID}/explanation" | eval "$PRETTY"

echo "[5/17] portfolio score"
curl -fsS -X POST "${API_BASE}/api/v2/portfolio/score" \
  -H 'Content-Type: application/json' \
  -d '{
    "positions": [
      {"target":"Test Startup","track":"vc","score":0.72,"risk":0.4},
      {"target":"BTC","track":"liquid","score":0.03,"risk":0.7}
    ],
    "risk_budget": 1.0
  }' | eval "$PRETTY"

echo "[6/17] risk limits"
curl -fsS "${API_BASE}/api/v2/risk/limits" | eval "$PRETTY"

echo "[7/17] risk check"
curl -fsS -X POST "${API_BASE}/api/v2/risk/check" \
  -H 'Content-Type: application/json' \
  -d '{
    "proposed_positions": [
      {"target":"BTC","track":"liquid","weight":0.35},
      {"target":"ETH","track":"liquid","weight":0.28}
    ],
    "current_positions": [
      {"target":"BTC","track":"liquid","weight":0.10}
    ],
    "realized_drawdown": 0.05
  }' | eval "$PRETTY"

echo "[8/17] signal generate (ensemble-v1)"
SIG_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/signals/generate" \
  -H 'Content-Type: application/json' \
  -d '{"track":"vc","target":"Test Startup","horizon":"1d","policy":"ensemble-v1","min_confidence":0.2}')
echo "$SIG_RESP" | eval "$PRETTY"

echo "[9/17] portfolio rebalance"
RB_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/portfolio/rebalance" \
  -H 'Content-Type: application/json' \
  -d '{
    "signals": [
      {"target":"BTC","track":"liquid","action":"buy","score":0.08,"confidence":0.8,"horizon":"1d"},
      {"target":"ETH","track":"liquid","action":"sell","score":-0.03,"confidence":0.7,"horizon":"1d"}
    ],
    "current_positions": [
      {"target":"BTC","track":"liquid","weight":0.05}
    ],
    "capital": 1.0,
    "risk_budget": 0.6
  }')
echo "$RB_RESP" | eval "$PRETTY"
DECISION_ID=$(echo "$RB_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['decision_id'])")

echo "[10/17] execution run"
curl -fsS -X POST "${API_BASE}/api/v2/execution/run" \
  -H 'Content-Type: application/json' \
  -d "{\"decision_id\":\"${DECISION_ID}\",\"adapter\":\"paper\",\"max_orders\":50}" | eval "$PRETTY"

echo "[11/17] backtest run"
BT_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/backtest/run" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","targets":["BTC"],"horizon":"1d","lookback_days":30,"train_days":14,"test_days":3,"fee_bps":5,"slippage_bps":3}')
echo "$BT_RESP" | eval "$PRETTY"
RUN_ID=$(echo "$BT_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['run_id'])")

echo "[12/17] backtest detail"
curl -fsS "${API_BASE}/api/v2/backtest/${RUN_ID}" | eval "$PRETTY"

echo "[13/17] model gate evaluate"
curl -fsS -X POST "${API_BASE}/api/v2/models/gate/evaluate" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","model_name":"liquid_ttm_ensemble","model_version":"v2.1","min_ic":0.0,"min_pnl_after_cost":0.0,"max_drawdown":0.2,"windows":1}' | eval "$PRETTY"

echo "[14/17] model rollback check"
curl -fsS -X POST "${API_BASE}/api/v2/models/rollback/check" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","model_name":"liquid_ttm_ensemble","model_version":"v2.1","max_recent_losses":3,"min_recent_hit_rate":0.4,"max_recent_drawdown":0.25}' | eval "$PRETTY"

echo "[15/17] data quality sample"
DQ_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/data-quality/sample" \
  -H 'Content-Type: application/json' \
  -d '{"limit":20,"min_quality_score":0.0}')
echo "$DQ_RESP" | eval "$PRETTY"
AUDIT_ID=$(echo "$DQ_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); items=d.get('items',[]); print(items[0]['id'] if items else '')")

echo "[16/17] data quality audit update"
if [ -n "$AUDIT_ID" ]; then
  curl -fsS -X POST "${API_BASE}/api/v2/data-quality/audit" \
    -H 'Content-Type: application/json' \
    -d "{\"audit_id\":${AUDIT_ID},\"reviewer\":\"smoke-test\",\"verdict\":\"correct\",\"note\":\"auto-check\"}" | eval "$PRETTY"
else
  echo "{\"status\":\"skip\",\"reason\":\"no_audit_items\"}" | eval "$PRETTY"
fi

echo "[17/17] data quality stats"
curl -fsS "${API_BASE}/api/v2/data-quality/stats?lookback_days=7" | eval "$PRETTY"

echo "V2 API smoke test passed (phase0-4 enhanced)"
