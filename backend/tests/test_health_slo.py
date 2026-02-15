from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "monitoring"))

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
