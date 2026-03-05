from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cost.cost_profile import compute_cost_map, load_cost_profile


@dataclass(frozen=True)
class PanelSampleRef:
    symbol_id: int
    t_idx: int
    end_ts: int


def _parse_horizon_seconds(h: str) -> int:
    text = str(h or "").strip().lower()
    if text.endswith("m"):
        return max(1, int(text[:-1])) * 60
    if text.endswith("h"):
        return max(1, int(text[:-1])) * 3600
    if text.endswith("d"):
        return max(1, int(text[:-1])) * 86400
    raise ValueError(f"unsupported_horizon:{h}")


def _parse_bar_seconds(raw: str) -> int:
    text = str(raw or "5m").strip().lower()
    if text.endswith("m"):
        return max(1, int(text[:-1])) * 60
    if text.endswith("h"):
        return max(1, int(text[:-1])) * 3600
    if text.endswith("d"):
        return max(1, int(text[:-1])) * 86400
    return max(1, int(text))


class LiquidPanelCacheDataset(Dataset):
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        lookback: int,
        horizons: Sequence[str] = ("1h", "4h", "1d", "7d"),
        cost_profile_name: str = "standard",
        require_cache: bool = True,
    ):
        self.cache_dir = Path(str(cache_dir))
        if require_cache and not self.cache_dir.exists():
            raise RuntimeError(f"training_cache_missing:{self.cache_dir}")
        manifest_file = self.cache_dir / "cache_manifest.json"
        if not manifest_file.exists():
            raise RuntimeError(f"training_cache_manifest_missing:{manifest_file}")
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.lookback = int(max(8, lookback))
        self.horizons = [str(h).lower() for h in horizons]
        self.profile = load_cost_profile(cost_profile_name)
        bar_size = str(self.manifest.get("bar_size") or "5m")
        bar_sec = _parse_bar_seconds(bar_size)
        self.horizon_steps = {h: max(1, int(round(_parse_horizon_seconds(h) / bar_sec))) for h in self.horizons}

        symbols = [str(s).upper() for s in (self.manifest.get("symbols") or [])]
        if not symbols:
            raise RuntimeError("training_cache_symbols_empty")
        self.symbols = symbols
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for s, i in self.symbol_to_id.items()}

        index_file = self.cache_dir / str(self.manifest.get("index_file") or "index.npz")
        if not index_file.exists():
            raise RuntimeError(f"training_cache_index_missing:{index_file}")
        idx_npz = np.load(index_file, mmap_mode="r")
        self._refs = [
            PanelSampleRef(symbol_id=int(sid), t_idx=int(ti), end_ts=int(ts))
            for sid, ti, ts in zip(idx_npz["symbol_id"], idx_npz["t_idx"], idx_npz["end_ts"])
        ]
        if not self._refs:
            raise RuntimeError("training_cache_index_empty")

        self._symbol_cache: Dict[str, Mapping[str, np.ndarray]] = {}
        self.require_multi_tf_context = str(os.getenv("LIQUID_REQUIRE_MULTI_TF_CONTEXT", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_tf_feature_names = [str(x) for x in (self.manifest.get("multi_tf_feature_names") or [])]
        if self.require_multi_tf_context and not self.multi_tf_feature_names:
            raise RuntimeError("training_cache_multi_tf_feature_names_missing")
        for sym in self.symbols:
            p = self.cache_dir / f"{sym}.npz"
            if not p.exists():
                raise RuntimeError(f"training_cache_symbol_file_missing:{sym}:{p}")
            payload = np.load(p, mmap_mode="r")
            if self.require_multi_tf_context and ("multi_tf_context" not in payload or "multi_tf_mask" not in payload):
                raise RuntimeError(f"training_cache_multi_tf_context_missing:{sym}:{p}")
            self._symbol_cache[sym] = payload

    def __len__(self) -> int:
        return len(self._refs)

    def _market_state_from_regime(self, regime_vec: np.ndarray) -> Dict[str, float]:
        return {
            "realized_vol": float(regime_vec[0]),
            "notional_usd": 1.0,
        }

    def _liquidity_from_regime(self, regime_vec: np.ndarray) -> Dict[str, float]:
        depth_val = max(0.0, float(np.expm1(regime_vec[4])))
        spread_val = max(0.0, float(regime_vec[3]))
        liq_score = 1.0 / (1.0 + spread_val * 0.01) if spread_val > 0 else 1.0
        return {
            "orderbook_depth_total": depth_val,
            "liquidity_score": float(np.clip(liq_score, 0.0, 1.0)),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self._refs[idx]
        symbol = self.id_to_symbol.get(int(ref.symbol_id))
        if symbol is None:
            raise RuntimeError(f"symbol_id_missing:{ref.symbol_id}")
        payload = self._symbol_cache[symbol]
        values_all = payload["values"]
        mask_all = payload["mask"]
        close_all = payload["close"]
        regime_all = payload["regime_features"]
        regime_mask_all = payload["regime_mask"]
        mtf_all = payload["multi_tf_context"] if "multi_tf_context" in payload else None
        mtf_mask_all = payload["multi_tf_mask"] if "multi_tf_mask" in payload else None
        t_idx = int(ref.t_idx)
        start = int(t_idx - self.lookback + 1)
        if start < 0:
            raise RuntimeError(f"lookback_window_underflow:{symbol}:{t_idx}:{self.lookback}")
        seq_vals = np.asarray(values_all[start : t_idx + 1], dtype=np.float32)
        seq_mask = np.asarray(mask_all[start : t_idx + 1], dtype=np.float32)
        if seq_vals.shape[0] != self.lookback:
            raise RuntimeError(f"lookback_window_mismatch:{symbol}:{seq_vals.shape[0]}:{self.lookback}")

        p0 = float(close_all[t_idx])
        if p0 <= 0:
            raise RuntimeError(f"invalid_close_price:{symbol}:{t_idx}:{p0}")
        regime_vec = np.asarray(regime_all[t_idx], dtype=np.float32)
        regime_msk = np.asarray(regime_mask_all[t_idx], dtype=np.uint8)
        if mtf_all is None or mtf_mask_all is None:
            if self.require_multi_tf_context:
                raise RuntimeError(f"multi_tf_context_missing_in_sample:{symbol}:{t_idx}")
            mtf_vec = np.zeros((len(self.multi_tf_feature_names),), dtype=np.float32)
            mtf_msk = np.ones((len(self.multi_tf_feature_names),), dtype=np.uint8)
        else:
            mtf_vec = np.asarray(mtf_all[t_idx], dtype=np.float32)
            mtf_msk = np.asarray(mtf_mask_all[t_idx], dtype=np.uint8)
        if self.multi_tf_feature_names:
            expected = len(self.multi_tf_feature_names)
            if mtf_vec.shape[0] != expected or mtf_msk.shape[0] != expected:
                raise RuntimeError(f"multi_tf_context_dim_mismatch:{symbol}:{mtf_vec.shape[0]}:{mtf_msk.shape[0]}:{expected}")
        regime_full = np.concatenate([regime_vec, mtf_vec], axis=0).astype(np.float32)
        regime_full_mask = np.concatenate([regime_msk, mtf_msk], axis=0).astype(np.uint8)

        market_state = self._market_state_from_regime(regime_vec)
        liquidity = self._liquidity_from_regime(regime_vec)
        turnover_est = float(np.clip(np.mean(np.abs(np.diff(seq_vals[:, 0]))), 0.01, 1.5))
        cost_map_bps = compute_cost_map(
            horizons=self.horizons,
            profile=self.profile,
            account_state={"turnover_estimate": turnover_est, "notional_usd": float(market_state.get("notional_usd", 1.0) or 1.0)},
            market_state=market_state,
            liquidity_features=liquidity,
            turnover_estimate=None,
        )

        y_raw: List[float] = []
        y_net: List[float] = []
        cost_bps: List[float] = []
        direction: List[float] = []
        future_vol: List[float] = []
        drawdown_proxy: List[float] = []
        for h in self.horizons:
            step = int(self.horizon_steps[h])
            fut_idx = t_idx + step
            if fut_idx >= close_all.shape[0]:
                raise RuntimeError(f"future_index_oob:{symbol}:{h}:{t_idx}:{fut_idx}:{close_all.shape[0]}")
            p1 = float(close_all[fut_idx])
            raw = (p1 - p0) / max(1e-12, p0)
            c_bps = float(cost_map_bps[h])
            net = raw - c_bps / 10000.0
            win = np.asarray(close_all[t_idx + 1 : fut_idx + 1], dtype=np.float64)
            if win.size > 1:
                ret = np.diff(win) / np.clip(win[:-1], 1e-12, None)
                f_vol = float(np.std(ret))
                dd = float(np.min((win / max(win[0], 1e-12)) - 1.0))
            else:
                f_vol = 0.0
                dd = 0.0
            y_raw.append(float(raw))
            y_net.append(float(net))
            cost_bps.append(c_bps)
            direction.append(1.0 if net >= 0 else 0.0)
            future_vol.append(f_vol)
            drawdown_proxy.append(dd)

        return {
            "symbol": symbol,
            "symbol_id": torch.tensor(int(ref.symbol_id), dtype=torch.long),
            "end_ts": int(ref.end_ts),
            "x_values": torch.tensor(seq_vals, dtype=torch.float32),
            "x_mask": torch.tensor(seq_mask, dtype=torch.float32),
            "values_seq": torch.tensor(seq_vals, dtype=torch.float32),
            "mask_seq": torch.tensor(seq_mask, dtype=torch.float32),
            "y_raw": torch.tensor(y_raw, dtype=torch.float32),
            "y": torch.tensor(y_net, dtype=torch.float32),
            "y_net": torch.tensor(y_net, dtype=torch.float32),
            "cost_bps": torch.tensor(cost_bps, dtype=torch.float32),
            "direction": torch.tensor(direction, dtype=torch.float32),
            "future_vol": torch.tensor(future_vol, dtype=torch.float32),
            "drawdown_proxy": torch.tensor(drawdown_proxy, dtype=torch.float32),
            "regime_features": torch.tensor(regime_full, dtype=torch.float32),
            "regime_mask": torch.tensor(regime_full_mask.astype(np.float32), dtype=torch.float32),
            "multi_timeframe_context": torch.tensor(mtf_vec, dtype=torch.float32),
            "multi_timeframe_mask": torch.tensor(mtf_msk.astype(np.float32), dtype=torch.float32),
            "extra": {
                "regime_features": torch.tensor(regime_full, dtype=torch.float32),
                "regime_mask": torch.tensor(regime_full_mask.astype(np.float32), dtype=torch.float32),
                "multi_timeframe_context": torch.tensor(mtf_vec, dtype=torch.float32),
                "multi_timeframe_mask": torch.tensor(mtf_msk.astype(np.float32), dtype=torch.float32),
            },
        }
