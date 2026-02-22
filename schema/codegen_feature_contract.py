from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from schema.schema_hash import compute_schema_hash, load_schema
except Exception:  # pragma: no cover
    from schema_hash import compute_schema_hash, load_schema


def _py_repr_list(items: List[Any]) -> str:
    return "[\n" + "\n".join(f"    {repr(x)}," for x in items) + "\n]"


def _py_repr_dict(items: Dict[str, Any]) -> str:
    rows = "\n".join(f"    {repr(k)}: {repr(v)}," for k, v in items.items())
    return "{\n" + rows + "\n}"


def render_contract(schema_path: str) -> str:
    schema = load_schema(schema_path)
    schema_hash = compute_schema_hash(schema_path)
    features: List[Dict[str, Any]] = list(schema["features"])

    keys = [f["name"] for f in features]
    dtype_map = {f["name"]: f["dtype"] for f in features}
    group_map = {f["name"]: f["group"] for f in features}
    source_map = {f["name"]: f["source"] for f in features}
    impute_map = {f["name"]: f["impute"] for f in features}
    required = sorted([f["name"] for f in features if bool(f["required"])])
    synth_allowed = {f["name"]: bool(f["is_synthetic_allowed"]) for f in features}

    return f'''# AUTO-GENERATED FILE. DO NOT EDIT BY HAND.
# Source: {schema_path}

from __future__ import annotations

from typing import Dict, List, Set

SCHEMA_PATH = {schema_path!r}
SCHEMA_NAME = {schema["schema_name"]!r}
SCHEMA_VERSION = {schema["schema_version"]!r}
BUCKET_INTERVAL = {schema["bucket_interval"]!r}
SCHEMA_HASH = {schema_hash!r}

FEATURE_KEYS: List[str] = {_py_repr_list(keys)}
FEATURE_INDEX: Dict[str, int] = { _py_repr_dict({k: i for i, k in enumerate(keys)}) }
DTYPE_MAP: Dict[str, str] = { _py_repr_dict(dtype_map) }
GROUP_MAP: Dict[str, str] = { _py_repr_dict(group_map) }
SOURCE_MAP: Dict[str, str] = { _py_repr_dict(source_map) }
IMPUTE_MAP: Dict[str, str] = { _py_repr_dict(impute_map) }
REQUIRED_KEYS: Set[str] = set({_py_repr_list(required)})
SYNTHETIC_ALLOWED_MAP: Dict[str, bool] = { _py_repr_dict(synth_allowed) }
FEATURE_DIM = {len(keys)}
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate features/feature_contract.py from schema")
    ap.add_argument("--schema", default="schema/liquid_feature_schema.yaml")
    ap.add_argument("--out", default="features/feature_contract.py")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_contract(args.schema), encoding="utf-8")
    print(f"generated:{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
