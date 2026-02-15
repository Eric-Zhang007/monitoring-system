from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "monitoring"))

hc = pytest.importorskip("health_check")


def test_evaluate_slo_includes_p50_p99_and_availability():
    metrics = """
ms_signal_latency_seconds_bucket{le="0.1"} 80
ms_signal_latency_seconds_bucket{le="0.2"} 100
ms_signal_latency_seconds_bucket{le="+Inf"} 100
ms_execution_latency_seconds_bucket{le="0.2"} 60
ms_execution_latency_seconds_bucket{le="0.3"} 100
ms_execution_latency_seconds_bucket{le="+Inf"} 100
ms_http_requests_total{method="GET",path="/a",status="200"} 990
ms_http_requests_total{method="GET",path="/a",status="500"} 10
""".strip()
    slo = hc.evaluate_slo_from_metrics(metrics)
    assert "p50_seconds" in slo["signal_latency"]
    assert "p99_seconds" in slo["signal_latency"]
    assert "api_availability" in slo
    assert slo["api_availability"]["value"] == 0.99


def test_evaluate_collector_slo_pass():
    metrics = """
ms_collector_connector_fetch_total{connector="gdelt",status="success"} 95
ms_collector_connector_fetch_total{connector="gdelt",status="empty"} 3
ms_collector_connector_fetch_total{connector="gdelt",status="failure"} 2
ms_collector_source_publish_to_ingest_seconds_bucket{connector="gdelt",le="30"} 80
ms_collector_source_publish_to_ingest_seconds_bucket{connector="gdelt",le="60"} 96
ms_collector_source_publish_to_ingest_seconds_bucket{connector="gdelt",le="120"} 100
ms_collector_source_publish_to_ingest_seconds_bucket{connector="gdelt",le="+Inf"} 100
""".strip()
    slo = hc.evaluate_collector_slo_from_metrics(metrics)
    assert slo["overall"]["status"] == "pass"
    assert slo["connector_success_rate"]["status"] == "pass"
    assert slo["connector_success_rate"]["success_rate"] == 0.98
    assert slo["source_publish_to_ingest"]["status"] == "pass"
    assert slo["source_publish_to_ingest"]["p95_seconds"] == 60.0


def test_evaluate_collector_slo_insufficient_observation():
    slo = hc.evaluate_collector_slo_from_metrics("")
    assert slo["overall"]["status"] == "insufficient_observation"
    assert slo["connector_success_rate"]["status"] == "insufficient_observation"
    assert slo["source_publish_to_ingest"]["status"] == "insufficient_observation"
