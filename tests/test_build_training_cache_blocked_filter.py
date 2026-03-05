from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.build_training_cache as build_cache_mod


def test_build_training_cache_excludes_blocked_symbols(monkeypatch, tmp_path: Path):
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps({"symbols": ["BTC", "ETH", "SOL"], "snapshot_hash": "u_hash_1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"blocked_symbols": ["ETH"]}, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "cache"

    def _fake_build(cfg):  # noqa: ANN001
        payload = json.loads(Path(cfg.universe_snapshot_file).read_text(encoding="utf-8"))
        assert payload["symbols"] == ["BTC", "SOL"]
        np.savez_compressed(out_dir / "index.npz", symbol_id=np.array([0], dtype=np.int32), t_idx=np.array([1], dtype=np.int32), end_ts=np.array([1], dtype=np.int64))
        return {"cache_hash": "cache_x"}

    monkeypatch.setattr(build_cache_mod, "build_training_cache_from_db", _fake_build)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_training_cache.py",
            "--universe-snapshot",
            str(universe),
            "--start",
            "2026-01-01T00:00:00Z",
            "--end",
            "2026-01-10T00:00:00Z",
            "--output-dir",
            str(out_dir),
            "--readiness-file",
            str(readiness),
            "--exclude-blocked",
            "1",
        ],
    )
    assert build_cache_mod.main() == 0


def test_build_training_cache_fails_fast_when_readiness_missing(monkeypatch, tmp_path: Path):
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps({"symbols": ["BTC", "ETH"], "snapshot_hash": "u_hash_2"}, ensure_ascii=False),
        encoding="utf-8",
    )
    out_dir = tmp_path / "cache"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_training_cache.py",
            "--universe-snapshot",
            str(universe),
            "--start",
            "2026-01-01T00:00:00Z",
            "--end",
            "2026-01-10T00:00:00Z",
            "--output-dir",
            str(out_dir),
            "--readiness-file",
            str(tmp_path / "missing.json"),
            "--exclude-blocked",
            "1",
        ],
    )
    with pytest.raises(RuntimeError, match="readiness_file_missing"):
        build_cache_mod.main()


def test_build_training_cache_passes_incremental_config(monkeypatch, tmp_path: Path):
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps({"symbols": ["BTC"], "snapshot_hash": "u_hash_3"}, ensure_ascii=False),
        encoding="utf-8",
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"blocked_symbols": []}, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "cache"

    def _fake_build(cfg):  # noqa: ANN001
        assert bool(cfg.incremental) is True
        assert int(cfg.incremental_warmup_steps) == 480
        np.savez_compressed(out_dir / "index.npz", symbol_id=np.array([0], dtype=np.int32), t_idx=np.array([1], dtype=np.int32), end_ts=np.array([1], dtype=np.int64))
        return {"cache_hash": "cache_inc"}

    monkeypatch.setattr(build_cache_mod, "build_training_cache_from_db", _fake_build)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_training_cache.py",
            "--universe-snapshot",
            str(universe),
            "--start",
            "2026-01-01T00:00:00Z",
            "--end",
            "2026-01-10T00:00:00Z",
            "--output-dir",
            str(out_dir),
            "--readiness-file",
            str(readiness),
            "--exclude-blocked",
            "1",
            "--incremental",
            "1",
            "--incremental-warmup-steps",
            "480",
        ],
    )
    assert build_cache_mod.main() == 0
