from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from features.feature_contract import FEATURE_DIM, FEATURE_KEYS, SCHEMA_HASH


@dataclass(frozen=True)
class FeatureRow:
    values: np.ndarray
    mask: np.ndarray
    schema_hash: str

    def as_payload(self) -> Dict[str, object]:
        return {
            "values": [float(x) for x in self.values.astype(np.float32).reshape(-1).tolist()],
            "mask": [int(x) for x in self.mask.astype(np.uint8).reshape(-1).tolist()],
            "schema_hash": str(self.schema_hash),
            "feature_dim": int(FEATURE_DIM),
        }


@dataclass(frozen=True)
class SequenceBatch:
    values: np.ndarray
    mask: np.ndarray
    schema_hash: str
    symbol: str
    start_ts: str
    end_ts: str
    bucket_interval: str
    coverage_summary: Dict[str, float]

    def assert_shape(self, lookback: int) -> None:
        if self.values.shape != (int(lookback), int(FEATURE_DIM)):
            raise ValueError(f"invalid_values_shape:{self.values.shape}:expected={(lookback, FEATURE_DIM)}")
        if self.mask.shape != (int(lookback), int(FEATURE_DIM)):
            raise ValueError(f"invalid_mask_shape:{self.mask.shape}:expected={(lookback, FEATURE_DIM)}")
        if str(self.schema_hash) != str(SCHEMA_HASH):
            raise ValueError("schema_hash_mismatch")


def empty_row() -> FeatureRow:
    return FeatureRow(
        values=np.zeros((FEATURE_DIM,), dtype=np.float32),
        mask=np.ones((FEATURE_DIM,), dtype=np.uint8),
        schema_hash=str(SCHEMA_HASH),
    )


def payload_to_row(payload: Dict[str, object]) -> FeatureRow:
    vals = np.array(payload.get("values") or [], dtype=np.float32).reshape(-1)
    msk = np.array(payload.get("mask") or [], dtype=np.uint8).reshape(-1)
    if vals.size != FEATURE_DIM or msk.size != FEATURE_DIM:
        raise ValueError(f"feature_dim_mismatch:values={vals.size}:mask={msk.size}:expected={FEATURE_DIM}")
    if str(payload.get("schema_hash") or "") != str(SCHEMA_HASH):
        raise ValueError("schema_hash_mismatch")
    return FeatureRow(values=vals, mask=msk, schema_hash=str(SCHEMA_HASH))


def values_mask_to_feature_map(values: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    vv = np.array(values, dtype=np.float32).reshape(-1)
    mm = np.array(mask, dtype=np.uint8).reshape(-1)
    if vv.size != FEATURE_DIM or mm.size != FEATURE_DIM:
        raise ValueError("feature_dim_mismatch")
    for i, k in enumerate(FEATURE_KEYS):
        out[k] = {"value": float(vv[i]), "missing": int(mm[i])}
    return out


def feature_map_to_values_mask(rows: Dict[str, Dict[str, float | int]]) -> tuple[np.ndarray, np.ndarray]:
    values: List[float] = []
    mask: List[int] = []
    for k in FEATURE_KEYS:
        item = rows.get(k) if isinstance(rows.get(k), dict) else {}
        values.append(float((item or {}).get("value", 0.0) or 0.0))
        mask.append(int((item or {}).get("missing", 1) or 0))
    return np.array(values, dtype=np.float32), np.array(mask, dtype=np.uint8)
