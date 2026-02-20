from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

import eval_multimodal_oos as eval_mod  # noqa: E402


def test_eval_multimodal_oos_main_writes_artifact(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    n_rows = 1400
    keys = ["ret_1", "funding_rate", "social_buzz", "event_density"]
    X = rng.normal(size=(n_rows, len(keys))).astype(np.float64)
    y = rng.normal(size=(n_rows,)).astype(np.float64)

    monkeypatch.setattr(eval_mod, "LIQUID_FEATURE_KEYS", keys)
    monkeypatch.setattr(eval_mod, "_load_matrix", lambda *args, **kwargs: (X, y))

    out_path = tmp_path / "multimodal_eval.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_multimodal_oos.py",
            "--database-url",
            "postgresql://unused",
            "--start",
            "2025-01-01T00:00:00Z",
            "--end",
            "2025-12-31T00:00:00Z",
            "--symbols",
            "BTC",
            "--ablations",
            "full",
            "--out",
            str(out_path),
        ],
    )

    rc = eval_mod.main()
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert int(payload["folds"]) >= 1
    assert str(payload["primary_ablation"]) == "full"
