#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from _metrics_test_logger import record_metrics_test


def _extract_alert_blocks(text: str) -> dict[str, str]:
    lines = text.splitlines()
    blocks: dict[str, list[str]] = {}
    cur_name = None
    cur_lines: list[str] = []

    for line in lines:
        m = re.match(r"^\s*-\s*alert:\s*(\S+)\s*$", line)
        if m:
            if cur_name:
                blocks[cur_name] = cur_lines
            cur_name = m.group(1).strip()
            cur_lines = [line]
            continue
        if cur_name:
            cur_lines.append(line)
    if cur_name:
        blocks[cur_name] = cur_lines
    return {k: "\n".join(v) for k, v in blocks.items()}


def main() -> int:
    started_at = datetime.now(timezone.utc)
    ap = argparse.ArgumentParser(description="Validate phase4/5 alert thresholds and routing contracts")
    ap.add_argument("--file", default="monitoring/alerts.yml")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        out = {
            "passed": False,
            "error": f"file_not_found:{path}",
            "evaluated_at": started_at.isoformat(),
            "window_start": None,
            "window_end": None,
        }
        print(json.dumps(out))
        record_metrics_test(test_name="validate_phase45_alerts", payload=out, window_start=None, window_end=None)
        return 2

    text = path.read_text(encoding="utf-8")
    blocks = _extract_alert_blocks(text)

    checks = {
        "ExecutionRejectRateCritical": [
            'track="liquid"',
            "> 0.01",
            "for: 5m",
            "severity: P1",
            "route: trading_execution",
        ],
        "ApiAvailabilityLow": [
            "< 0.999",
            "for: 5m",
            "severity: P1",
            "route: platform_reliability",
        ],
        "ExecutionRejectReasonSkew": [
            "ms_execution_rejects_total",
            "for: 5m",
            "severity: P2",
        ],
        "SignalLatencyP99Degraded": [
            "histogram_quantile(0.99",
            "> 0.25",
            "for: 10m",
            "severity: P2",
        ],
        "CollectorConnectorFailureSpike": [
            'status="failure"',
            "> 10",
            "for: 5m",
            "severity: P2",
        ],
        "CollectorConnectorRateLimited": [
            "ms_collector_connector_rate_limit_total",
            "> 10",
            "for: 5m",
            "severity: P2",
        ],
        "CollectorConnectorSuccessRateLow": [
            "ms_collector_connector_fetch_total",
            "< 0.95",
            "for: 10m",
            "severity: P2",
        ],
        "CollectorSourcePublishToIngestP95Degraded": [
            "histogram_quantile(0.95",
            "ms_collector_source_publish_to_ingest_seconds_bucket",
            "> 120",
            "for: 10m",
            "severity: P2",
        ],
    }

    details = {}
    passed = True
    for alert_name, expect in checks.items():
        block = blocks.get(alert_name, "")
        missing = [s for s in expect if s not in block]
        ok = len(missing) == 0
        details[alert_name] = {"ok": ok, "missing": missing}
        passed = passed and ok

    out = {
        "passed": passed,
        "file": str(path),
        "checks": details,
        "evaluated_at": started_at.isoformat(),
        "window_start": None,
        "window_end": None,
    }
    print(json.dumps(out, ensure_ascii=False))
    record_metrics_test(
        test_name="validate_phase45_alerts",
        payload=out,
        window_start=None,
        window_end=None,
        extra={"argv": {"file": str(path)}},
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
