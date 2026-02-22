from __future__ import annotations

import json
from pathlib import Path

from schema.codegen_feature_contract import render_contract
from schema.schema_hash import compute_schema_hash


def test_codegen_contract_matches_schema(tmp_path: Path):
    schema = {
        "schema_name": "x",
        "schema_version": "main",
        "bucket_interval": "5m",
        "features": [
            {
                "name": "f1",
                "dtype": "float32",
                "group": "price",
                "source": "market_bars",
                "asof": "<= bucket_end_ts",
                "impute": "zero",
                "required": True,
                "is_synthetic_allowed": False,
                "comment": "",
            }
        ],
    }
    sp = tmp_path / "schema.yaml"
    sp.write_text(json.dumps(schema), encoding="utf-8")

    code = render_contract(str(sp))
    assert "FEATURE_KEYS" in code
    assert "SCHEMA_HASH" in code
    assert compute_schema_hash(sp) in code
