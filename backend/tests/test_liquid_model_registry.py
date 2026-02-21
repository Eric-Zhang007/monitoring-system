from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))

import liquid_model_registry as reg_mod  # noqa: E402


def test_registry_activate_and_rollback(monkeypatch, tmp_path):
    reg_path = tmp_path / "liquid_horizon_registry.json"
    monkeypatch.setenv("LIQUID_HORIZON_REGISTRY_FILE", str(reg_path))
    reg_mod.upsert_active(symbol="BTC", horizon="1h", model_name="m1", model_version="v1", actor="u")
    reg_mod.upsert_active(symbol="BTC", horizon="1h", model_name="m2", model_version="v2", actor="u")
    cur = reg_mod.get_active_model("BTC", "1h")
    assert cur == ("m2", "v2")
    out = reg_mod.rollback_active(symbol="BTC", horizon="1h", actor="u")
    assert out["rolled_back"] is True
    cur2 = reg_mod.get_active_model("BTC", "1h")
    assert cur2 == ("m1", "v1")


def test_registry_candidate_and_promote(monkeypatch, tmp_path):
    reg_path = tmp_path / "liquid_horizon_registry.json"
    monkeypatch.setenv("LIQUID_HORIZON_REGISTRY_FILE", str(reg_path))
    reg_mod.upsert_active(symbol="BTC", horizon="4h", model_name="active_m1", model_version="v1", actor="u")
    reg_mod.upsert_candidate(symbol="BTC", horizon="4h", model_name="cand_m2", model_version="v2", actor="u")
    cand = reg_mod.get_candidate_model("BTC", "4h")
    assert cand == ("cand_m2", "v2")
    out = reg_mod.promote_candidate(symbol="BTC", horizon="4h", actor="u")
    assert out["promoted"] is True
    cur = reg_mod.get_active_model("BTC", "4h")
    assert cur == ("cand_m2", "v2")
