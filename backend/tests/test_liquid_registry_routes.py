from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import v2_router as router_mod  # noqa: E402


def test_registry_candidate_promote_and_rollback_routes(monkeypatch, tmp_path):
    reg_path = tmp_path / "liquid_horizon_registry.json"
    monkeypatch.setenv("LIQUID_HORIZON_REGISTRY_FILE", str(reg_path))

    out_a = asyncio.run(
        router_mod.activate_liquid_registry(
            {
                "symbol": "BTC",
                "horizon": "1h",
                "model_name": "m_base",
                "model_version": "v1",
                "operator": "tester",
            }
        )
    )
    assert out_a["status"] == "ok"

    out_c = asyncio.run(
        router_mod.set_liquid_registry_candidate(
            {
                "symbol": "BTC",
                "horizon": "1h",
                "model_name": "m_cand",
                "model_version": "v2",
                "operator": "tester",
            }
        )
    )
    assert out_c["status"] == "ok"
    assert tuple(out_c.get("candidate") or ()) == ("m_cand", "v2")

    out_p = asyncio.run(
        router_mod.promote_liquid_registry_candidate(
            {
                "symbol": "BTC",
                "horizon": "1h",
                "operator": "tester",
            }
        )
    )
    assert out_p["status"] == "ok"
    assert out_p["promoted"] is True

    out_r = asyncio.run(
        router_mod.rollback_liquid_registry(
            {
                "symbol": "BTC",
                "horizon": "1h",
                "operator": "tester",
            }
        )
    )
    assert out_r["status"] == "ok"
    assert out_r["rolled_back"] is True
