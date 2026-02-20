from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_REQUESTS_TOTAL = Counter(
    "ms_http_requests_total",
    "Total HTTP requests by method/path/status.",
    ["method", "path", "status"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "ms_http_request_latency_seconds",
    "HTTP request latency by method/path.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

EXECUTION_ORDERS_TOTAL = Counter(
    "ms_execution_orders_total",
    "Order execution outcomes by adapter/status.",
    ["adapter", "status"],
)

EXECUTION_REJECT_RATE = Gauge(
    "ms_execution_reject_rate",
    "Recent execution reject rate by track.",
    ["track"],
)

EXECUTION_REJECTS_TOTAL = Counter(
    "ms_execution_rejects_total",
    "Execution rejects by adapter and normalized reason.",
    ["adapter", "reason"],
)

SIGNAL_LATENCY_SECONDS = Histogram(
    "ms_signal_latency_seconds",
    "Signal generation latency.",
    ["track"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2),
)

EXECUTION_LATENCY_SECONDS = Histogram(
    "ms_execution_latency_seconds",
    "Execution runtime latency.",
    ["adapter"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)

DATA_FRESHNESS_SECONDS = Gauge(
    "ms_data_freshness_seconds",
    "Data freshness in seconds by target.",
    ["target"],
)

MODEL_DRIFT_EVENTS_TOTAL = Counter(
    "ms_model_drift_events_total",
    "Drift evaluation decisions by track/action.",
    ["track", "action"],
)

BACKTEST_FAILED_RUNS_TOTAL = Counter(
    "ms_backtest_failed_runs_total",
    "Backtest failed runs by track/reason.",
    ["track", "reason"],
)

INGEST_EVENTS_TOTAL = Counter(
    "ms_ingest_events_total",
    "Ingest events API outcomes by status.",
    ["status"],
)

METRIC_GATE_STATUS = Gauge(
    "ms_metric_gate_status",
    "Metric gate pass/fail status by track and metric (1=pass,0=fail).",
    ["track", "metric"],
)

RISK_HARD_BLOCKS_TOTAL = Counter(
    "ms_risk_hard_blocks_total",
    "Hard risk blocks triggered by track.",
    ["track"],
)

MODEL_COVERAGE_RATIO = Gauge(
    "ms_model_coverage_ratio",
    "Backtest model inference coverage ratio by track.",
    ["track"],
)

SIGNAL_PREDICT_DIVERGENCE = Gauge(
    "ms_signal_predict_divergence",
    "Absolute divergence between signal and predict outputs.",
    ["track", "target"],
)

RISK_WEIGHT_CONVERSION_ERRORS_TOTAL = Counter(
    "ms_risk_weight_conversion_errors_total",
    "Count of execution orders that could not be converted to risk weight.",
    ["track"],
)

BITGET_UNCLASSIFIED_ERRORS_TOTAL = Counter(
    "ms_bitget_unclassified_errors_total",
    "Count of Bitget rejects that could not be normalized to a specific bucket.",
    ["adapter"],
)

STRICT_ASOF_FAIL_COUNT = Counter(
    "ms_strict_asof_fail_count",
    "Count of strict as-of hard-fail backtests by track.",
    ["track"],
)

WEBSOCKET_ACTIVE_CONNECTIONS = Gauge(
    "ms_websocket_active_connections",
    "Current number of active websocket connections.",
)

WEBSOCKET_DROPPED_MESSAGES_TOTAL = Counter(
    "ms_websocket_dropped_messages_total",
    "Dropped websocket messages due to backpressure/slow consumers.",
    ["reason"],
)

CONNECTOR_FETCH_TOTAL = Counter(
    "ms_connector_fetch_total",
    "Connector fetch outcomes by connector and status.",
    ["connector", "status"],
)

CONNECTOR_EMPTY_RESULTS_TOTAL = Counter(
    "ms_connector_empty_results_total",
    "Connector empty-result count by connector.",
    ["connector"],
)

CONNECTOR_RATE_LIMIT_TOTAL = Counter(
    "ms_connector_rate_limit_total",
    "Connector rate-limit events by connector.",
    ["connector"],
)

CONNECTOR_FETCH_LATENCY_SECONDS = Histogram(
    "ms_connector_fetch_latency_seconds",
    "Connector fetch latency by connector.",
    ["connector"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30),
)

CONNECTOR_COOLDOWN_SKIPS_TOTAL = Counter(
    "ms_connector_cooldown_skips_total",
    "Connector fetch skips caused by circuit-breaker cooldown.",
    ["connector"],
)

MULTIMODAL_TEXT_COVERAGE_RATIO = Gauge(
    "ms_multimodal_text_coverage_ratio",
    "Estimated text coverage ratio for multimodal track.",
    ["track"],
)

MULTIMODAL_GATE_OPENNESS = Gauge(
    "ms_multimodal_gate_openness",
    "Estimated gate openness (0~1) for multimodal fusion.",
    ["track"],
)

MULTIMODAL_TEXT_CONTRIBUTION_ABS = Gauge(
    "ms_multimodal_text_contribution_abs",
    "Absolute text contribution estimate from residual branch.",
    ["track"],
)

MULTIMODAL_READY_BACKBONES = Gauge(
    "ms_multimodal_ready_backbones",
    "Count of ready backbones in phase-D experiments.",
    ["track"],
)

MULTIMODAL_CANDIDATE_GATE_PASSED = Gauge(
    "ms_multimodal_candidate_gate_passed",
    "Candidate gate status (1=passed,0=failed).",
    ["track"],
)


def observe_http_request(method: str, path: str, status: int, elapsed_seconds: float) -> None:
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed_seconds)


def render_metrics() -> bytes:
    return generate_latest()


def observe_signal_latency(track: str, elapsed_seconds: float) -> None:
    SIGNAL_LATENCY_SECONDS.labels(track=track).observe(elapsed_seconds)


def observe_execution_latency(adapter: str, elapsed_seconds: float) -> None:
    EXECUTION_LATENCY_SECONDS.labels(adapter=adapter).observe(elapsed_seconds)


def set_multimodal_health_metrics(
    *,
    track: str,
    text_coverage_ratio: float,
    gate_openness: float,
    text_contribution_abs: float,
    ready_backbones: int,
    candidate_gate_passed: bool,
) -> None:
    t = str(track or "liquid")
    MULTIMODAL_TEXT_COVERAGE_RATIO.labels(track=t).set(float(max(0.0, min(1.0, text_coverage_ratio))))
    MULTIMODAL_GATE_OPENNESS.labels(track=t).set(float(max(0.0, min(1.0, gate_openness))))
    MULTIMODAL_TEXT_CONTRIBUTION_ABS.labels(track=t).set(float(max(0.0, text_contribution_abs)))
    MULTIMODAL_READY_BACKBONES.labels(track=t).set(float(max(0, int(ready_backbones))))
    MULTIMODAL_CANDIDATE_GATE_PASSED.labels(track=t).set(1.0 if bool(candidate_gate_passed) else 0.0)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "EXECUTION_ORDERS_TOTAL",
    "EXECUTION_REJECT_RATE",
    "EXECUTION_REJECTS_TOTAL",
    "SIGNAL_LATENCY_SECONDS",
    "EXECUTION_LATENCY_SECONDS",
    "DATA_FRESHNESS_SECONDS",
    "MODEL_DRIFT_EVENTS_TOTAL",
    "BACKTEST_FAILED_RUNS_TOTAL",
    "INGEST_EVENTS_TOTAL",
    "METRIC_GATE_STATUS",
    "RISK_HARD_BLOCKS_TOTAL",
    "MODEL_COVERAGE_RATIO",
    "SIGNAL_PREDICT_DIVERGENCE",
    "RISK_WEIGHT_CONVERSION_ERRORS_TOTAL",
    "BITGET_UNCLASSIFIED_ERRORS_TOTAL",
    "STRICT_ASOF_FAIL_COUNT",
    "WEBSOCKET_ACTIVE_CONNECTIONS",
    "WEBSOCKET_DROPPED_MESSAGES_TOTAL",
    "CONNECTOR_FETCH_TOTAL",
    "CONNECTOR_EMPTY_RESULTS_TOTAL",
    "CONNECTOR_RATE_LIMIT_TOTAL",
    "CONNECTOR_FETCH_LATENCY_SECONDS",
    "CONNECTOR_COOLDOWN_SKIPS_TOTAL",
    "MULTIMODAL_TEXT_COVERAGE_RATIO",
    "MULTIMODAL_GATE_OPENNESS",
    "MULTIMODAL_TEXT_CONTRIBUTION_ABS",
    "MULTIMODAL_READY_BACKBONES",
    "MULTIMODAL_CANDIDATE_GATE_PASSED",
    "observe_http_request",
    "observe_signal_latency",
    "observe_execution_latency",
    "set_multimodal_health_metrics",
    "render_metrics",
]
