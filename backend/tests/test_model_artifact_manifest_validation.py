from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v2_repository import V2Repository


class _FakeCursor:
    def __init__(self, artifact_path: str):
        self._artifact_path = artifact_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, _sql, _params):
        return None

    def fetchone(self):
        return {"artifact_path": self._artifact_path}


class _FakeConn:
    def __init__(self, artifact_path: str):
        self._artifact_path = artifact_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def cursor(self):
        return _FakeCursor(self._artifact_path)


class _FakeConnectCtx:
    def __init__(self, artifact_path: str):
        self._artifact_path = artifact_path

    def __enter__(self):
        return _FakeConn(self._artifact_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _build_repo(artifact_path: str) -> V2Repository:
    repo = object.__new__(V2Repository)
    repo._connect = lambda: _FakeConnectCtx(artifact_path)  # type: ignore[attr-defined]
    return repo


def test_model_artifact_manifest_requires_fields(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_name": "liquid_ttm_ensemble",
                "model_version": "v2.1",
                "track": "liquid",
                "type": "ensemble_manifest",
                "created_at": "2026-02-15T00:00:00Z",
                "feature_version": "feature-store-v2.1",
                "data_version": "v1",
            }
        ),
        encoding="utf-8",
    )
    repo = _build_repo(str(model_path))
    assert repo.model_artifact_exists("liquid_ttm_ensemble", "liquid", "v2.1") is True


def test_model_artifact_manifest_rejects_missing_required_field(tmp_path: Path):
    model_path = tmp_path / "bad.json"
    model_path.write_text(
        json.dumps(
            {
                "model_name": "liquid_ttm_ensemble",
                "model_version": "v2.1",
                "track": "liquid",
                "type": "ensemble_manifest",
                "created_at": "2026-02-15T00:00:00Z",
                "feature_version": "feature-store-v2.1",
            }
        ),
        encoding="utf-8",
    )
    repo = _build_repo(str(model_path))
    assert repo.model_artifact_exists("liquid_ttm_ensemble", "liquid", "v2.1") is False
