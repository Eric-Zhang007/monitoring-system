from __future__ import annotations

import json
from pathlib import Path

from mlops_artifacts.validate import validate_manifest_dir
from schema.schema_hash import compute_schema_hash


def test_manifest_validation_ok(tmp_path: Path):
    mdir = tmp_path / "m1"
    mdir.mkdir(parents=True, exist_ok=True)

    schema_snapshot = mdir / "schema_snapshot.yaml"
    schema_snapshot.write_text(
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
                        "asof": "<= t",
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
    (mdir / "weights.pt").write_bytes(b"abc")
    (mdir / "training_report.json").write_text("{}", encoding="utf-8")

    schema_hash = compute_schema_hash(schema_snapshot)
    manifest = {
        "model_id": "liquid_main",
        "schema_hash": schema_hash,
        "lookback": 64,
        "feature_dim": 1,
        "data_version": "main",
        "metrics_summary": {},
        "files": {
            "weights": "weights.pt",
            "schema_snapshot": "schema_snapshot.yaml",
            "training_report": "training_report.json",
        },
    }
    (mdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out = validate_manifest_dir(mdir)
    assert out["model_id"] == "liquid_main"
