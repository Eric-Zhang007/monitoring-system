from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional

import numpy as np

from features.feature_contract import DTYPE_MAP, FEATURE_DIM, FEATURE_KEYS, IMPUTE_MAP, REQUIRED_KEYS, SCHEMA_HASH


class FeatureAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class AlignmentResult:
    values: np.ndarray
    mask: np.ndarray


def _coerce_value(name: str, value: object) -> float:
    dtype = str(DTYPE_MAP.get(name, "float32"))
    if dtype in {"float32", "float64"}:
        return float(value)
    if dtype in {"int32", "int64"}:
        return float(int(value))
    if dtype == "bool":
        return 1.0 if bool(value) else 0.0
    raise FeatureAlignmentError(f"unsupported_dtype:{name}:{dtype}")


def _impute_value(name: str, strategy: str, stats: Optional[Mapping[str, float]], ffill: Optional[Mapping[str, float]]) -> float:
    mode = str(strategy or "zero").strip().lower()
    if mode == "zero":
        return 0.0
    if mode == "median":
        if stats and name in stats:
            return float(stats[name])
        return 0.0
    if mode == "ffill":
        if ffill and name in ffill:
            return float(ffill[name])
        return 0.0
    raise FeatureAlignmentError(f"unsupported_impute_strategy:{name}:{strategy}")


def align_row(
    row_dict: Mapping[str, object],
    *,
    stats: Optional[Mapping[str, float]] = None,
    ffill_state: Optional[MutableMapping[str, float]] = None,
    allow_required_missing: bool = False,
) -> AlignmentResult:
    row = dict(row_dict or {})
    extras = sorted(set(row.keys()) - set(FEATURE_KEYS))
    if extras:
        raise FeatureAlignmentError(f"schema_extra_keys:{extras}")

    values = np.zeros((FEATURE_DIM,), dtype=np.float32)
    mask = np.zeros((FEATURE_DIM,), dtype=np.uint8)

    missing_required = []
    for idx, key in enumerate(FEATURE_KEYS):
        raw = row.get(key, None)
        if raw is None:
            missing = True
        elif isinstance(raw, str) and raw.strip() == "":
            missing = True
        else:
            missing = False

        if missing:
            if (key in REQUIRED_KEYS) and (not allow_required_missing):
                missing_required.append(key)
            values[idx] = float(_impute_value(key, IMPUTE_MAP.get(key, "zero"), stats, ffill_state))
            mask[idx] = 1
        else:
            try:
                values[idx] = float(_coerce_value(key, raw))
            except Exception as exc:
                raise FeatureAlignmentError(f"invalid_value:{key}:{raw}") from exc
            mask[idx] = 0
            if ffill_state is not None:
                ffill_state[key] = float(values[idx])

    if missing_required:
        raise FeatureAlignmentError(f"required_missing:{sorted(missing_required)}")

    return AlignmentResult(values=values, mask=mask)


def build_all_missing_row(*, stats: Optional[Mapping[str, float]] = None) -> AlignmentResult:
    return align_row({}, stats=stats, ffill_state=None, allow_required_missing=True)


def validate_schema_hash(schema_hash: str) -> None:
    if str(schema_hash) != str(SCHEMA_HASH):
        raise FeatureAlignmentError(f"schema_hash_mismatch:{schema_hash}:{SCHEMA_HASH}")
