from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FIELDS = {
    "name",
    "dtype",
    "group",
    "source",
    "asof",
    "impute",
    "required",
    "is_synthetic_allowed",
    "comment",
}


class SchemaValidationError(ValueError):
    pass


def _parse_schema_text(text: str) -> Dict[str, Any]:
    # Prefer stdlib-only parsing. JSON is a valid YAML subset.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        if isinstance(obj, dict):
            return obj
    except Exception as exc:
        raise SchemaValidationError(f"schema_parse_failed:{exc}") from exc

    raise SchemaValidationError("schema_parse_failed:non_dict_root")


def load_schema(schema_path: str | Path) -> Dict[str, Any]:
    path = Path(schema_path)
    if not path.exists():
        raise SchemaValidationError(f"schema_missing:{path}")
    data = _parse_schema_text(path.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise SchemaValidationError("schema_features_must_be_non_empty_list")

    seen = set()
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(features):
        if not isinstance(item, dict):
            raise SchemaValidationError(f"feature_item_not_dict:{idx}")
        miss = REQUIRED_FIELDS - set(item.keys())
        if miss:
            raise SchemaValidationError(f"feature_missing_fields:{idx}:{sorted(miss)}")
        name = str(item.get("name") or "").strip()
        if not name:
            raise SchemaValidationError(f"feature_name_empty:{idx}")
        if name in seen:
            raise SchemaValidationError(f"feature_name_duplicate:{name}")
        seen.add(name)
        normalized.append(
            {
                "name": name,
                "dtype": str(item["dtype"]).strip(),
                "group": str(item["group"]).strip(),
                "source": str(item["source"]).strip(),
                "asof": str(item["asof"]).strip(),
                "impute": str(item["impute"]).strip(),
                "required": bool(item["required"]),
                "is_synthetic_allowed": bool(item["is_synthetic_allowed"]),
                "comment": str(item.get("comment") or "").strip(),
            }
        )

    out = {
        "schema_name": str(data.get("schema_name") or "liquid_feature_schema").strip(),
        "schema_version": str(data.get("schema_version") or "main").strip(),
        "bucket_interval": str(data.get("bucket_interval") or "5m").strip(),
        "features": sorted(normalized, key=lambda x: x["name"]),
    }
    return out


def canonical_schema_json(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def compute_schema_hash(schema_path: str | Path) -> str:
    schema = load_schema(schema_path)
    payload = canonical_schema_json(schema).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute schema hash")
    ap.add_argument("--schema", default="schema/liquid_feature_schema.yaml")
    args = ap.parse_args()
    print(compute_schema_hash(args.schema))
