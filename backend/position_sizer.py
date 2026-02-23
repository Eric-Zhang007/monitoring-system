from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from account_state.models import AccountState
from risk_state.models import RiskRegime, RiskState


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


class PositionSizer:
    def __init__(self) -> None:
        self.base_band = _env_float("BASE_BAND", 0.02)
        self.pos_max = _env_float("POS_MAX", 1.0)
        self.risk_budget_k = _env_float("RISK_BUDGET_K", 1.0)
        self.default_horizon_seconds = {
            "1h": 3600,
            "4h": 4 * 3600,
            "1d": 24 * 3600,
            "7d": 7 * 24 * 3600,
        }

    @staticmethod
    def _weighted_score(
        mu_map: Dict[str, float],
        sigma_map: Dict[str, float],
        cost_map: Dict[str, float],
        weights: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        eps = _env_float("SIGNAL_VOL_EPS", 1e-4)
        score_map: Dict[str, float] = {}
        acc = 0.0
        wsum = 0.0
        for h in ("1h", "4h", "1d", "7d"):
            if h not in mu_map:
                continue
            mu = float(mu_map.get(h, 0.0) or 0.0)
            sigma = float(sigma_map.get(h, 0.0) or 0.0)
            cost = float(cost_map.get(h, 0.0) or 0.0)
            score_h = (mu - cost) / max(eps, sigma)
            w = float(weights.get(h, 0.0))
            acc += w * score_h
            wsum += abs(w)
            score_map[h] = float(score_h)
        if wsum <= 1e-9:
            return 0.0, score_map
        return float(acc / wsum), score_map

    @staticmethod
    def _horizon_weights(pred: Dict[str, Any]) -> Dict[str, float]:
        default_weights = {"1h": 0.40, "4h": 0.30, "1d": 0.20, "7d": 0.10}
        raw = pred.get("weights")
        if not isinstance(raw, dict):
            return default_weights
        out = dict(default_weights)
        for h in tuple(out.keys()):
            if h in raw:
                try:
                    out[h] = float(raw[h])
                except Exception:
                    continue
        return out

    @staticmethod
    def _estimate_uncertainty(sigma_map: Dict[str, float], horizons: Iterable[str]) -> float:
        sigmas = [float(sigma_map.get(h, 0.0) or 0.0) for h in horizons]
        if not sigmas:
            return 0.0
        return float(np.mean(np.array(sigmas, dtype=np.float64)))

    def compute_target_position(
        self,
        pred: Dict[str, Any],
        cost_map: Dict[str, float],
        account: AccountState,
        risk_state: RiskState,
        params: Dict[str, Any] | None = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        params = dict(params or {})
        symbol = str(params.get("symbol") or pred.get("symbol") or "").upper()
        horizon = str(params.get("horizon") or "1h").lower()
        mu_map = dict(pred.get("mu") or {})
        sigma_map = dict(pred.get("sigma") or {})
        if not mu_map:
            mu_map = {"1h": float(pred.get("expected_return", 0.0) or 0.0)}
        if not sigma_map:
            sigma_map = {"1h": max(1e-6, float(pred.get("vol_forecast", 0.0) or 0.0))}
        weights = self._horizon_weights(pred)
        score, score_map = self._weighted_score(mu_map, sigma_map, cost_map, weights)

        current = 0.0
        if symbol and symbol in account.positions:
            current = float(account.positions[symbol].qty)

        base_target = float(np.clip(self.risk_budget_k * score, -self.pos_max, self.pos_max))
        pos_scale = float((risk_state.soft_penalty_factors or {}).get("pos_scale", 1.0) or 1.0)
        band_scale = float((risk_state.soft_penalty_factors or {}).get("band_scale", 1.0) or 1.0)
        target = float(base_target * pos_scale)
        band = float(max(1e-6, self.base_band * band_scale))
        delta = float(target - current)

        reduce_only = risk_state.regime == RiskRegime.RED
        if abs(delta) <= band:
            action = "hold"
            qty = 0.0
        elif delta > 0:
            action = "buy"
            qty = abs(delta)
        else:
            action = "sell"
            qty = abs(delta)

        intent = {
            "symbol": symbol,
            "horizon": horizon,
            "horizon_seconds": int(self.default_horizon_seconds.get(horizon, 3600)),
            "current_pos": float(current),
            "target_pos": float(target),
            "delta_pos": float(delta),
            "qty": float(qty),
            "band": float(band),
            "score": float(score),
            "score_h": score_map,
            "action": action,
            "reduce_only": bool(reduce_only),
            "uncertainty": self._estimate_uncertainty(sigma_map, score_map.keys()),
        }
        return float(target), float(band), intent

