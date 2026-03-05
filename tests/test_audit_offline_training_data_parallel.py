from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_offline_training_data_parallel.py"
    spec = importlib.util.spec_from_file_location("audit_offline_training_data_parallel", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parallel_audit_merges_symbol_readiness(monkeypatch, tmp_path: Path):
    mod = _load_module()

    def _fake_resolve(_db: str, _track: str, _asof, _top_n: int):  # noqa: ANN001
        return ["BTC", "ETH", "SOL"]

    def _fake_run_one(_base, symbol: str, idx: int):  # noqa: ANN001
        status = {"BTC": "READY", "ETH": "DEGRADED", "SOL": "BLOCKED"}[symbol]
        return {
            "symbol": symbol,
            "task_id": f"task-{idx}",
            "row": {
                "status": status,
                "blocked_reasons": ["x"] if status == "BLOCKED" else [],
                "degraded_reasons": ["y"] if status == "DEGRADED" else [],
            },
        }

    monkeypatch.setattr(mod, "_resolve_symbols", _fake_resolve)
    monkeypatch.setattr(mod, "_run_one", _fake_run_one)

    out_file = tmp_path / "readiness.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_offline_training_data_parallel.py",
            "--database-url",
            "postgresql://unused",
            "--track",
            "liquid",
            "--symbols",
            "",
            "--as-of",
            "2026-01-01T00:00:00Z",
            "--start",
            "2025-01-01T00:00:00Z",
            "--end",
            "2025-02-01T00:00:00Z",
            "--parallel-workers",
            "2",
            "--output",
            str(out_file),
        ],
    )

    assert mod.main() == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["summary"] == {"READY": 1, "DEGRADED": 1, "BLOCKED": 1}
    assert payload["blocked_symbols"] == ["SOL"]
    assert payload["gates"]["ready"] is False
