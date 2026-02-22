from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class WalkForwardPurgedConfig:
    train_days: int = 60
    val_days: int = 7
    test_days: int = 7
    purge_gap_hours: int = 24
    step_days: int = 7


def _to_dt(x: datetime | str) -> datetime:
    if isinstance(x, datetime):
        return x
    s = str(x).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def build_walkforward_purged_splits(
    timestamps: Sequence[datetime | str],
    cfg: WalkForwardPurgedConfig,
) -> List[Dict[str, np.ndarray]]:
    if not timestamps:
        return []

    ts = np.array([_to_dt(x) for x in timestamps], dtype=object)
    start = min(ts)
    stop = max(ts)

    train_days = max(1, int(cfg.train_days))
    val_days = max(1, int(cfg.val_days))
    test_days = max(1, int(cfg.test_days))
    step_days = max(1, int(cfg.step_days))
    purge_gap = timedelta(hours=max(0, int(cfg.purge_gap_hours)))

    splits: List[Dict[str, np.ndarray]] = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=train_days)
        val_start = train_end + purge_gap
        val_end = val_start + timedelta(days=val_days)
        test_start = val_end + purge_gap
        test_end = test_start + timedelta(days=test_days)
        if test_end > stop:
            break

        train_idx = np.where((ts >= train_start) & (ts < train_end))[0].astype(np.int64)
        val_idx = np.where((ts >= val_start) & (ts < val_end))[0].astype(np.int64)
        test_idx = np.where((ts >= test_start) & (ts < test_end))[0].astype(np.int64)

        if train_idx.size > 0 and val_idx.size > 0 and test_idx.size > 0:
            splits.append(
                {
                    "train": train_idx,
                    "val": val_idx,
                    "test": test_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

        cursor = cursor + timedelta(days=step_days)

    return splits
