from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "watch_ingest_task.py"
    spec = importlib.util.spec_from_file_location("watch_ingest_task", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_checkpoint_updated_at_from_payload(tmp_path: Path):
    mod = _load_module()
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(json.dumps({"updated_at": "2026-03-02T03:00:00Z", "completed_chunks": {}}), encoding="utf-8")
    dt = mod._parse_checkpoint_updated_at(ckpt)
    assert dt is not None
    assert dt.isoformat().startswith("2026-03-02T03:00:00")


def test_checkpoint_summary_handles_missing(tmp_path: Path):
    mod = _load_module()
    missing = tmp_path / "missing.json"
    out = mod._checkpoint_summary(missing)
    assert out["exists"] is False
