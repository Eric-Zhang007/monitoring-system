from __future__ import annotations

import json
from pathlib import Path

from schema.schema_hash import compute_schema_hash


def test_schema_hash_stable_when_feature_order_changes(tmp_path: Path):
    base = {
        "schema_name": "x",
        "schema_version": "main",
        "bucket_interval": "5m",
        "features": [
            {
                "name": "b",
                "dtype": "float32",
                "group": "g",
                "source": "s",
                "asof": "<= t",
                "impute": "zero",
                "required": False,
                "is_synthetic_allowed": False,
                "comment": "",
            },
            {
                "name": "a",
                "dtype": "float32",
                "group": "g",
                "source": "s",
                "asof": "<= t",
                "impute": "zero",
                "required": False,
                "is_synthetic_allowed": False,
                "comment": "",
            },
        ],
    }
    p1 = tmp_path / "s1.yaml"
    p2 = tmp_path / "s2.yaml"
    p1.write_text(json.dumps(base), encoding="utf-8")
    base["features"] = list(reversed(base["features"]))
    p2.write_text(json.dumps(base), encoding="utf-8")

    assert compute_schema_hash(p1) == compute_schema_hash(p2)
