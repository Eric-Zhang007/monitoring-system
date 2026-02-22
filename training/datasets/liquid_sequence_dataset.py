from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from features.sequence import build_sequence
from training.labels.liquid_labels import compute_label_targets


@dataclass(frozen=True)
class SequenceSample:
    symbol: str
    end_ts: datetime
    x_values: np.ndarray
    x_mask: np.ndarray
    y: Dict[str, float]
    schema_hash: str


class LiquidSequenceDataset(Dataset):
    def __init__(self, samples: Sequence[SequenceSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if torch is None:
            raise RuntimeError("torch_required_for_dataset_tensor_output")
        s = self.samples[idx]
        x_values = torch.tensor(s.x_values, dtype=torch.float32)
        x_mask = torch.tensor(s.x_mask, dtype=torch.float32)
        y = torch.tensor([float(s.y[k]) for k in ("ret_1h_net", "ret_4h_net", "ret_1d_net", "ret_7d_net")], dtype=torch.float32)
        return {
            "symbol": s.symbol,
            "end_ts": s.end_ts.isoformat(),
            "x_values": x_values,
            "x_mask": x_mask,
            "y": y,
            "schema_hash": s.schema_hash,
        }


def _parse_ts(raw: object) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw or "").strip().replace(" ", "T")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_training_samples(
    *,
    db_url: str,
    symbols: List[str],
    start_ts: datetime,
    end_ts: datetime,
    lookback: int,
    max_samples_per_symbol: int = 0,
) -> List[SequenceSample]:
    out: List[SequenceSample] = []
    horizon_steps = {
        "1h": 12,
        "4h": 48,
        "1d": 288,
        "7d": 2016,
    }
    max_h = max(horizon_steps.values())

    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            for sym in symbols:
                cur.execute(
                    """
                    SELECT ts, close::double precision AS close
                    FROM market_bars
                    WHERE symbol=%s
                      AND timeframe='5m'
                      AND ts >= %s
                      AND ts <= %s + INTERVAL '8 day'
                    ORDER BY ts ASC
                    """,
                    (sym, start_ts, end_ts),
                )
                price_rows = [dict(r) for r in cur.fetchall()]
                if len(price_rows) < lookback + max_h + 8:
                    continue
                ts_list = [_parse_ts(r["ts"]) for r in price_rows]
                px_list = [float(r.get("close") or 0.0) for r in price_rows]

                count = 0
                for i in range(lookback, len(ts_list) - max_h):
                    ts = ts_list[i]
                    if ts < start_ts or ts > end_ts:
                        continue
                    if px_list[i] <= 0:
                        continue
                    labels = compute_label_targets(prices=px_list, index=i, horizon_steps=horizon_steps)
                    seq = build_sequence(db_url=db_url, symbol=sym, end_ts=ts, lookback=lookback)
                    if seq.values.shape != (lookback, FEATURE_DIM):
                        continue
                    out.append(
                        SequenceSample(
                            symbol=sym,
                            end_ts=ts,
                            x_values=seq.values,
                            x_mask=seq.mask,
                            y=labels,
                            schema_hash=SCHEMA_HASH,
                        )
                    )
                    count += 1
                    if max_samples_per_symbol > 0 and count >= max_samples_per_symbol:
                        break
    return out
