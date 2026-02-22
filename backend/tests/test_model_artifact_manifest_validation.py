from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "backend"))
from v2_repository import V2Repository
from schema.schema_hash import compute_schema_hash


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


def _write_manifest_dir(root: Path, model_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    snap = root / "schema_snapshot.yaml"
    snap.write_text(
        json.dumps(
            {
                "schema_name": "x",
                "schema_version": "main",
                "bucket_interval": "5m",
                "features": [
                    {
                        "name": "f1",
                        "dtype": "float32",
                        "group": "g",
                        "source": "s",
                        "asof": "<=t",
                        "impute": "zero",
                        "required": False,
                        "is_synthetic_allowed": False,
                        "comment": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "weights.pt").write_bytes(b"abc")
    (root / "training_report.json").write_text("{}", encoding="utf-8")
    schema_hash = compute_schema_hash(snap)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "schema_hash": schema_hash,
                "lookback": 16,
                "feature_dim": 1,
                "data_version": "main",
                "metrics_summary": {},
                "files": {
                    "weights": "weights.pt",
                    "schema_snapshot": "schema_snapshot.yaml",
                    "training_report": "training_report.json",
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_model_artifact_manifest_requires_dir_contract(tmp_path: Path):
    model_dir = _write_manifest_dir(tmp_path / "model", "liquid_main")
    repo = _build_repo(str(model_dir))
    assert repo.model_artifact_exists("liquid_main", "liquid", "main") is True


def test_model_artifact_manifest_rejects_missing_file(tmp_path: Path):
    model_dir = _write_manifest_dir(tmp_path / "bad", "liquid_main")
    (model_dir / "weights.pt").unlink()
    repo = _build_repo(str(model_dir))
    assert repo.model_artifact_exists("liquid_main", "liquid", "main") is False
