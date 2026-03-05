from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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

from features.feature_contract import FEATURE_DIM, FEATURE_INDEX, SCHEMA_HASH
from features.sequence import build_sequence
from training.cache.panel_cache import compute_regime_features
from training.labels.liquid_labels import compute_label_targets


HORIZONS = ("1h", "4h", "1d", "7d")


@dataclass(frozen=True)
class SequenceSample:
    symbol: str
    symbol_id: int
    end_ts: datetime
    x_values: np.ndarray
    x_mask: np.ndarray
    regime_features: np.ndarray
    regime_mask: np.ndarray
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
        y_raw = torch.tensor([float(s.y[f"ret_{h}_raw"]) for h in HORIZONS], dtype=torch.float32)
        y_net = torch.tensor([float(s.y[f"ret_{h}_net"]) for h in HORIZONS], dtype=torch.float32)
        cost_bps = torch.tensor([float(s.y[f"cost_{h}_bps"]) for h in HORIZONS], dtype=torch.float32)
        direction = torch.tensor([float(s.y[f"direction_{h}"]) for h in HORIZONS], dtype=torch.float32)
        return {
            "symbol": s.symbol,
            "symbol_id": torch.tensor(int(s.symbol_id), dtype=torch.long),
            "end_ts": s.end_ts.isoformat(),
            "x_values": x_values,
            "x_mask": x_mask,
            "y_raw": y_raw,
            "y": y_net,
            "y_net": y_net,
            "cost_bps": cost_bps,
            "direction": direction,
            "regime_features": torch.tensor(np.asarray(s.regime_features, dtype=np.float32), dtype=torch.float32),
            "regime_mask": torch.tensor(np.asarray(s.regime_mask, dtype=np.float32), dtype=torch.float32),
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


def _liquidity_context(last_values: np.ndarray) -> Dict[str, float]:
    depth_idx = FEATURE_INDEX.get("orderbook_depth_total")
    vol_idx = FEATURE_INDEX.get("vol_12")
    context: Dict[str, float] = {}
    if depth_idx is not None:
        context["orderbook_depth_total"] = float(last_values[int(depth_idx)])
    if vol_idx is not None:
        context["realized_vol"] = float(abs(last_values[int(vol_idx)]))
    return context


def _estimate_turnover_base_from_prices(prices: Sequence[float], idx: int, lookback: int) -> float:
    lo = max(1, int(idx - max(8, lookback) + 1))
    win = np.asarray(prices[lo : idx + 1], dtype=np.float64)
    if win.size < 3:
        return 0.35
    ret = np.diff(win) / np.clip(win[:-1], 1e-12, None)
    # Scale absolute return activity into a turnover baseline range.
    base = float(np.mean(np.abs(ret)) * 24.0)
    return float(np.clip(base, 0.02, 1.5))


def load_training_samples(
    *,
    db_url: str,
    symbols: List[str],
    start_ts: datetime,
    end_ts: datetime,
    lookback: int,
    max_samples_per_symbol: int = 0,
    cost_profile_name: str = "standard",
) -> List[SequenceSample]:
    out: List[SequenceSample] = []
    symbol_to_id = {str(s).upper(): i for i, s in enumerate(sorted({str(x).upper() for x in symbols if str(x).strip()}))}
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

                    seq = build_sequence(db_url=db_url, symbol=sym, end_ts=ts, lookback=lookback)
                    if seq.values.shape != (lookback, FEATURE_DIM):
                        continue
                    regime_features_arr, regime_mask_arr = compute_regime_features(seq.values, seq.mask)
                    regime_features = np.asarray(regime_features_arr[-1], dtype=np.float32)
                    regime_mask = np.asarray(regime_mask_arr[-1], dtype=np.float32)

                    liq_ctx = _liquidity_context(seq.values[-1, :])
                    labels = compute_label_targets(
                        prices=px_list,
                        index=i,
                        horizon_steps=horizon_steps,
                        market_state={"realized_vol": float(abs(liq_ctx.get("realized_vol", 0.0)))},
                        liquidity_features=liq_ctx,
                        account_state={"turnover_estimate": _estimate_turnover_base_from_prices(px_list, i, lookback)},
                        turnover_estimate=None,
                        cost_profile_name=cost_profile_name,
                    )
                    out.append(
                        SequenceSample(
                            symbol=sym,
                            symbol_id=int(symbol_to_id.get(sym, 0)),
                            end_ts=ts,
                            x_values=seq.values,
                            x_mask=seq.mask,
                            regime_features=regime_features,
                            regime_mask=regime_mask,
                            y=labels,
                            schema_hash=SCHEMA_HASH,
                        )
                    )
                    count += 1
                    if max_samples_per_symbol > 0 and count >= max_samples_per_symbol:
                        break
    return out
