#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
if command -v jq >/dev/null 2>&1; then
  PRETTY="jq ."
else
  PRETTY="cat"
fi

assert_json() {
  local json="$1"
  local expr="$2"
  echo "$json" | python3 -c 'import json,sys; data=json.load(sys.stdin); expr=sys.argv[1]; assert eval(expr, {}, {"data": data}), f"assert failed: {expr}"' "$expr"
}

echo "[1/27] health"
curl -fsS "${API_BASE}/health" | eval "$PRETTY"

echo "[1.1/27] pre-reset kill switch (liquid/global)"
curl -sS -X POST "${API_BASE}/api/v2/risk/kill-switch/reset" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","strategy_id":"global","reason":"smoke_pre_reset"}' | eval "$PRETTY"

echo "[2/27] ingest events"
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

echo "[3/27] vc predict"
VC_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/predict/vc" \
  -H 'Content-Type: application/json' \
  -d '{"company_name":"Test Startup","horizon_months":12}')
echo "$VC_RESP" | eval "$PRETTY"
assert_json "$VC_RESP" "'prediction_id' in data and isinstance(data['prediction_id'], int)"
PRED_ID=$(echo "$VC_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['prediction_id'])")

echo "[4/27] explanation"
curl -fsS "${API_BASE}/api/v2/predictions/${PRED_ID}/explanation" | eval "$PRETTY"

echo "[5/27] portfolio score"
curl -fsS -X POST "${API_BASE}/api/v2/portfolio/score" \
  -H 'Content-Type: application/json' \
  -d '{
    "positions": [
      {"target":"Test Startup","track":"vc","score":0.72,"risk":0.4},
      {"target":"BTC","track":"liquid","score":0.03,"risk":0.7}
    ],
    "risk_budget": 1.0
  }' | eval "$PRETTY"

echo "[6/27] risk limits"
curl -fsS "${API_BASE}/api/v2/risk/limits" | eval "$PRETTY"

echo "[7/27] risk check"
RISK_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/risk/check" \
  -H 'Content-Type: application/json' \
  -d '{
    "proposed_positions": [
      {"target":"BTC","track":"liquid","weight":0.35},
      {"target":"ETH","track":"liquid","weight":0.28}
    ],
    "current_positions": [
      {"target":"BTC","track":"liquid","weight":0.10}
    ],
    "realized_drawdown": 0.05,
    "daily_loss": 0.0,
    "consecutive_losses": 0
  }')
echo "$RISK_RESP" | eval "$PRETTY"
assert_json "$RISK_RESP" "'approved' in data and 'violations' in data and isinstance(data['violations'], list)"

echo "[8/27] signal generate (ensemble-v1)"
SIG_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/signals/generate" \
  -H 'Content-Type: application/json' \
  -d '{"track":"vc","target":"Test Startup","horizon":"1d","policy":"ensemble-v1","min_confidence":0.2}')
echo "$SIG_RESP" | eval "$PRETTY"

echo "[9/27] portfolio rebalance"
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

echo "[10/27] execution run"
EXEC_RUN_BODY=$(mktemp)
EXEC_RUN_CODE=$(curl -sS -o "$EXEC_RUN_BODY" -w "%{http_code}" -X POST "${API_BASE}/api/v2/execution/run" \
  -H 'Content-Type: application/json' \
  -d "{\"decision_id\":\"${DECISION_ID}\",\"adapter\":\"paper\",\"time_in_force\":\"IOC\",\"max_slippage_bps\":20,\"venue\":\"coinbase\",\"max_orders\":50,\"limit_timeout_sec\":2.0,\"max_retries\":1,\"fee_bps\":5.0}")
cat "$EXEC_RUN_BODY" | eval "$PRETTY"
if [ "$EXEC_RUN_CODE" = "423" ]; then
  assert_json "$(cat "$EXEC_RUN_BODY")" "'detail' in data and ('risk_blocked:' in data['detail'] or 'kill_switch_triggered:' in data['detail'])"
elif [ "$EXEC_RUN_CODE" = "200" ]; then
  assert_json "$(cat "$EXEC_RUN_BODY")" "'orders' in data and isinstance(data['orders'], list)"
else
  echo "unexpected execution status: $EXEC_RUN_CODE"
  exit 1
fi
rm -f "$EXEC_RUN_BODY"

echo "[11/27] execution submit orders"
SUBMIT_EXEC_BODY=$(mktemp)
SUBMIT_EXEC_CODE=$(curl -sS -o "$SUBMIT_EXEC_BODY" -w "%{http_code}" -X POST "${API_BASE}/api/v2/execution/orders" \
  -H 'Content-Type: application/json' \
  -d '{
    "adapter":"paper",
    "venue":"coinbase",
    "time_in_force":"IOC",
    "max_slippage_bps":15,
    "orders":[
      {"target":"BTC","track":"liquid","side":"buy","quantity":0.01,"est_price":50000,"strategy_id":"smoke-v1","metadata":{"source":"smoke"}}
    ]
  }')
cat "$SUBMIT_EXEC_BODY" | eval "$PRETTY"
if [ "$SUBMIT_EXEC_CODE" = "423" ]; then
  assert_json "$(cat "$SUBMIT_EXEC_BODY")" "'detail' in data and 'kill_switch_triggered:' in data['detail']"
  ORDER_ID=""
elif [ "$SUBMIT_EXEC_CODE" = "200" ]; then
  assert_json "$(cat "$SUBMIT_EXEC_BODY")" "'order_ids' in data and isinstance(data['order_ids'], list)"
  ORDER_ID=$(cat "$SUBMIT_EXEC_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); ids=d.get('order_ids',[]); print(ids[0] if ids else '')")
else
  echo "unexpected execution submit status: $SUBMIT_EXEC_CODE"
  exit 1
fi
rm -f "$SUBMIT_EXEC_BODY"

echo "[12/27] execution order detail"
if [ -n "$ORDER_ID" ]; then
  curl -fsS "${API_BASE}/api/v2/execution/orders/${ORDER_ID}" | eval "$PRETTY"
else
  echo "{\"status\":\"skip\",\"reason\":\"no_order_ids\"}" | eval "$PRETTY"
fi

echo "[13/27] backtest run"
BT_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/backtest/run" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","targets":["BTC"],"horizon":"1d","model_name":"liquid_ttm_ensemble","model_version":"v2.1","data_version":"v1","lookback_days":30,"train_days":14,"test_days":3,"fee_bps":5,"slippage_bps":3}')
echo "$BT_RESP" | eval "$PRETTY"
assert_json "$BT_RESP" "'metrics' in data and isinstance(data['metrics'], dict)"
assert_json "$BT_RESP" "'model_name' in data['metrics'] and 'model_version' in data['metrics'] and 'lineage_coverage' in data['metrics'] and 'cost_breakdown' in data['metrics']"
RUN_ID=$(echo "$BT_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['run_id'])")

echo "[14/27] backtest detail"
curl -fsS "${API_BASE}/api/v2/backtest/${RUN_ID}" | eval "$PRETTY"

echo "[15/27] model gate evaluate"
curl -fsS -X POST "${API_BASE}/api/v2/models/gate/evaluate" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","model_name":"liquid_ttm_ensemble","model_version":"v2.1","min_ic":0.0,"min_pnl_after_cost":0.0,"max_drawdown":0.2,"windows":1}' | eval "$PRETTY"

echo "[16/27] model gate auto evaluate"
curl -fsS -X POST "${API_BASE}/api/v2/models/gate/auto-evaluate" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","windows":1,"auto_promote":false}' | eval "$PRETTY"

echo "[17/27] model drift evaluate"
curl -fsS -X POST "${API_BASE}/api/v2/models/drift/evaluate" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","lookback_hours":24,"reference_hours":72}' | eval "$PRETTY"

echo "[18/27] model rollback check"
RBK_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/models/rollback/check" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","model_name":"liquid_ttm_ensemble","model_version":"v2.1","max_recent_losses":3,"min_recent_hit_rate":0.4,"max_recent_drawdown":0.25}')
echo "$RBK_RESP" | eval "$PRETTY"
assert_json "$RBK_RESP" "'rollback_triggered' in data and 'windows_failed' in data and 'trigger_rule' in data"

echo "[19/27] pnl attribution"
curl -fsS "${API_BASE}/api/v2/metrics/pnl-attribution?track=liquid&lookback_hours=168" | eval "$PRETTY"

echo "[20/27] data quality sample"
DQ_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/data-quality/sample" \
  -H 'Content-Type: application/json' \
  -d '{"limit":20,"min_quality_score":0.0}')
echo "$DQ_RESP" | eval "$PRETTY"
AUDIT_ID=$(echo "$DQ_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); items=d.get('items',[]); print(items[0]['id'] if items else '')")

echo "[21/27] data quality lineage check"
LINEAGE_RESP=$(curl -fsS -X POST "${API_BASE}/api/v2/data-quality/lineage/check" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","lineage_id":"smoke-lineage-001","data_version":"v1","strict":true,"max_mismatch_keys":10,"tolerance":0.000001}')
echo "$LINEAGE_RESP" | eval "$PRETTY"
assert_json "$LINEAGE_RESP" "'mismatch_keys' in data and isinstance(data['mismatch_keys'], list)"

echo "[22/27] data quality audit update"
if [ -n "$AUDIT_ID" ]; then
  curl -fsS -X POST "${API_BASE}/api/v2/data-quality/audit" \
    -H 'Content-Type: application/json' \
    -d "{\"audit_id\":${AUDIT_ID},\"reviewer\":\"smoke-test\",\"verdict\":\"correct\",\"note\":\"auto-check\"}" | eval "$PRETTY"
else
  echo "{\"status\":\"skip\",\"reason\":\"no_audit_items\"}" | eval "$PRETTY"
fi

echo "[23/27] data quality stats + prometheus metrics"
curl -fsS "${API_BASE}/api/v2/data-quality/stats?lookback_days=7" | eval "$PRETTY"
curl -fsS "${API_BASE}/metrics" | sed -n '1,20p'

echo "[24/27] kill switch trigger"
curl -fsS -X POST "${API_BASE}/api/v2/risk/kill-switch/trigger" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","strategy_id":"global","reason":"smoke_test","duration_minutes":5}' | eval "$PRETTY"

echo "[25/27] kill switch state"
curl -fsS "${API_BASE}/api/v2/risk/kill-switch?track=liquid&strategy_id=global" | eval "$PRETTY"

echo "[25.1/27] opening status"
OPENING_STATUS=$(curl -fsS "${API_BASE}/api/v2/risk/opening-status?track=liquid&strategy_id=global")
echo "$OPENING_STATUS" | eval "$PRETTY"
assert_json "$OPENING_STATUS" "'can_open_new_positions' in data and 'remaining_seconds' in data"

echo "[26/27] kill switch reset"
curl -fsS -X POST "${API_BASE}/api/v2/risk/kill-switch/reset" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","strategy_id":"global","reason":"smoke_test_reset"}' | eval "$PRETTY"

echo "[27/27] execution audit"
curl -fsS "${API_BASE}/api/v2/execution/audit/${DECISION_ID}" | eval "$PRETTY"

echo "[28/29] rollout advance"
curl -fsS -X POST "${API_BASE}/api/v2/models/rollout/advance" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","model_name":"liquid_ttm_ensemble","model_version":"v2.1","current_stage_pct":10,"next_stage_pct":30,"windows":1}' | eval "$PRETTY"

echo "[29/29] data quality consistency"
curl -fsS "${API_BASE}/api/v2/data-quality/consistency?lookback_days=30" | eval "$PRETTY"

echo "[30/31] async backtest task submit"
BT_TASK=$(curl -fsS -X POST "${API_BASE}/api/v2/tasks/backtest" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","targets":["BTC"],"horizon":"1d","lookback_days":30,"train_days":14,"test_days":3,"fee_bps":5,"slippage_bps":3}')
echo "$BT_TASK" | eval "$PRETTY"
assert_json "$BT_TASK" "'task_id' in data and data.get('task_type') == 'backtest_run'"
BT_TASK_ID=$(echo "$BT_TASK" | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
BT_TASK_STATUS=$(curl -fsS "${API_BASE}/api/v2/tasks/${BT_TASK_ID}")
echo "$BT_TASK_STATUS" | eval "$PRETTY"
assert_json "$BT_TASK_STATUS" "data.get('status') in ['queued','running','completed','failed']"

echo "[31/31] async pnl-attribution task submit"
PNL_TASK=$(curl -fsS -X POST "${API_BASE}/api/v2/tasks/pnl-attribution" \
  -H 'Content-Type: application/json' \
  -d '{"track":"liquid","lookback_hours":24}')
echo "$PNL_TASK" | eval "$PRETTY"
assert_json "$PNL_TASK" "'task_id' in data and data.get('task_type') == 'pnl_attribution'"

echo "V2 API smoke test passed (phase0-5 enhanced)"
