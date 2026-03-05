from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import HTTPException
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlops_artifacts.validate import validate_manifest_dir
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from inference.feature_reader import fetch_sequence
from liquid_model_registry import get_active_model as get_registry_active_model
from models.liquid_model import build_liquid_model_from_checkpoint
from training.cache.panel_cache import (
    REGIME_FEATURE_NAMES,
    build_multi_tf_feature_names,
    build_multi_tf_vector_from_payload,
    compute_regime_features,
    normalize_context_timeframes,
)


@dataclass(frozen=True)
class _LoadedModelBundle:
    model: Any
    manifest: Dict[str, Any]
    ckpt: Dict[str, Any]
    lookback: int
    model_dir: Path


class LiquidModelService:
    def __init__(
        self,
        *,
        repo: Any,
        feature_keys: List[str],
        feature_version: str,
        data_version: str,
        default_model_name: str = "liquid_main",
        default_model_version: str = "main",
    ):
        self.repo = repo
        self.feature_keys = list(feature_keys)
        self.feature_version = str(feature_version)
        self.data_version = str(data_version)
        self.default_model_name = str(default_model_name)
        self.default_model_version = str(default_model_version)
        self.model_dir = Path(str(os.getenv("LIQUID_MODEL_DIR", "artifacts/models/liquid_main")))

        self.model, self.manifest, self.ckpt, self.lookback = self._load_model_or_fail()
        primary = _LoadedModelBundle(
            model=self.model,
            manifest=dict(self.manifest),
            ckpt=dict(self.ckpt),
            lookback=int(self.lookback),
            model_dir=self.model_dir,
        )
        self.ensemble_members, self.ensemble_weights = self._load_ensemble_members_or_fail(primary)
        self.ensemble_enabled = len(self.ensemble_members) > 1

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_model_dirs(raw: str) -> List[Path]:
        out: List[Path] = []
        seen = set()
        for part in str(raw or "").split(","):
            text = str(part or "").strip()
            if not text:
                continue
            p = Path(text).expanduser().resolve()
            if str(p) in seen:
                continue
            seen.add(str(p))
            out.append(p)
        return out

    @staticmethod
    def _parse_weights(raw: str, *, count: int) -> np.ndarray:
        if count <= 0:
            raise RuntimeError("ensemble_member_count_invalid")
        text = str(raw or "").strip()
        if not text:
            return np.asarray([1.0 / float(count)] * count, dtype=np.float64)
        parts = [x.strip() for x in text.split(",") if x.strip()]
        if len(parts) != count:
            raise RuntimeError(f"ensemble_weights_count_mismatch:{len(parts)}:{count}")
        vals: List[float] = []
        for p in parts:
            v = float(p)
            if not np.isfinite(v) or v <= 0:
                raise RuntimeError(f"ensemble_weight_invalid:{p}")
            vals.append(v)
        w = np.asarray(vals, dtype=np.float64)
        s = float(np.sum(w))
        if s <= 0:
            raise RuntimeError("ensemble_weights_sum_nonpositive")
        return w / s

    @staticmethod
    def _horizon_list(ckpt: Dict[str, Any]) -> List[str]:
        return [str(h).strip().lower() for h in list(ckpt.get("horizons") or ["1h", "4h", "1d", "7d"])]

    @staticmethod
    def _quantile_list(ckpt: Dict[str, Any]) -> List[float]:
        return [float(q) for q in list(ckpt.get("quantiles") or [0.1, 0.5, 0.9])]

    def _load_model_from_dir_or_fail(self, model_dir: Path) -> _LoadedModelBundle:
        manifest = validate_manifest_dir(model_dir, expected_schema_hash=SCHEMA_HASH)
        weights = model_dir / manifest["files"]["weights"]
        ckpt = torch.load(weights, map_location="cpu")
        if str(ckpt.get("schema_hash") or "") != SCHEMA_HASH:
            raise RuntimeError(f"schema_hash_mismatch_weights:{model_dir}")
        lookback = int(ckpt.get("lookback") or 0)
        if lookback <= 0:
            raise RuntimeError(f"invalid_lookback:{model_dir}")
        fdim = int(ckpt.get("feature_dim") or 0)
        if fdim != FEATURE_DIM:
            raise RuntimeError(f"feature_dim_mismatch_weights:{fdim}:{FEATURE_DIM}:{model_dir}")

        model = build_liquid_model_from_checkpoint(ckpt)
        model.eval()
        return _LoadedModelBundle(
            model=model,
            manifest=dict(manifest),
            ckpt=dict(ckpt),
            lookback=lookback,
            model_dir=model_dir,
        )

    def _load_model_or_fail(self):
        loaded = self._load_model_from_dir_or_fail(self.model_dir)
        return loaded.model, loaded.manifest, loaded.ckpt, loaded.lookback

    def _load_ensemble_members_or_fail(self, primary: _LoadedModelBundle) -> Tuple[List[_LoadedModelBundle], np.ndarray]:
        ensemble_dirs_raw = str(os.getenv("LIQUID_ENSEMBLE_MODEL_DIRS", "")).strip()
        if not ensemble_dirs_raw:
            return [primary], np.asarray([1.0], dtype=np.float64)

        model_dirs = self._normalize_model_dirs(ensemble_dirs_raw)
        if not model_dirs:
            raise RuntimeError("ensemble_model_dirs_empty")

        members: List[_LoadedModelBundle] = []
        for d in model_dirs:
            if d == primary.model_dir.resolve():
                members.append(primary)
            else:
                members.append(self._load_model_from_dir_or_fail(d))

        base_lookback = int(primary.lookback)
        base_horizons = self._horizon_list(primary.ckpt)
        base_quantiles = self._quantile_list(primary.ckpt)
        for m in members:
            if int(m.lookback) != base_lookback:
                raise RuntimeError(f"ensemble_lookback_mismatch:{m.lookback}:{base_lookback}:{m.model_dir}")
            if self._horizon_list(m.ckpt) != base_horizons:
                raise RuntimeError(f"ensemble_horizons_mismatch:{m.model_dir}")
            if self._quantile_list(m.ckpt) != base_quantiles:
                raise RuntimeError(f"ensemble_quantiles_mismatch:{m.model_dir}")

        weights = self._parse_weights(str(os.getenv("LIQUID_ENSEMBLE_WEIGHTS", "")), count=len(members))
        return members, weights

    def _active_liquid_model(self, *, symbol: str = "", horizon: str = "1h") -> Tuple[str, str]:
        reg_hit = get_registry_active_model(symbol=symbol, horizon=horizon)
        if reg_hit:
            return reg_hit
        return self.default_model_name, self.default_model_version

    def latest_feature_payload(self, symbol: str, *, max_lookback_days: int = 30) -> Dict[str, float]:
        target = str(symbol or "").strip().upper()
        if not target:
            return {}
        with self.repo._connect() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT values, mask, schema_hash
                    FROM feature_matrix_main
                    WHERE symbol = %s
                      AND as_of_ts >= NOW() - make_interval(days => %s)
                    ORDER BY as_of_ts DESC
                    LIMIT 1
                    """,
                    (target, max(1, int(max_lookback_days))),
                )
                row = cur.fetchone()
        if not row:
            return {}
        payload = dict(row)
        if str(payload.get("schema_hash") or "") != SCHEMA_HASH:
            raise RuntimeError("schema_hash_mismatch_feature_matrix")
        vals = payload.get("values") if isinstance(payload.get("values"), list) else []
        if len(vals) != FEATURE_DIM:
            raise RuntimeError("feature_dim_mismatch_feature_matrix")
        return {k: float(vals[i]) for i, k in enumerate(self.feature_keys)}

    def has_required_artifacts(self, *, target: str, model_name: str) -> bool:
        _ = target
        _ = model_name
        try:
            for m in self.ensemble_members:
                validate_manifest_dir(m.model_dir, expected_schema_hash=SCHEMA_HASH)
            return True
        except Exception:
            return False

    @staticmethod
    def _calibrated_confidence_for_ckpt(
        ckpt: Dict[str, Any],
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        direction_logit: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        cal = ckpt.get("calibration") if isinstance(ckpt.get("calibration"), dict) else {}
        sigma_scale = float(cal.get("sigma_scale", 1.0) or 1.0)
        direction_temperature = float(cal.get("direction_temperature", 1.0) or 1.0)

        sigma_cal = np.clip(sigma * sigma_scale, 1e-6, None)
        if direction_logit is not None:
            z = direction_logit / max(1e-6, direction_temperature)
        else:
            z = mu / sigma_cal
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        return np.clip(p, 0.01, 0.99), sigma_cal, {
            "sigma_scale": sigma_scale,
            "direction_temperature": direction_temperature,
        }

    def _build_regime_for_member(
        self,
        *,
        member: _LoadedModelBundle,
        base_regime_vec: np.ndarray,
        base_regime_mask: np.ndarray,
        symbol: str,
        as_of: datetime,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        regime_vec = np.asarray(base_regime_vec, dtype=np.float32)
        regime_msk = np.asarray(base_regime_mask, dtype=np.uint8)
        regime_feature_map = {str(k): float(regime_vec[i]) for i, k in enumerate(REGIME_FEATURE_NAMES)}

        expected_regime_dim = int(member.ckpt.get("regime_dim") or regime_vec.shape[0])
        if expected_regime_dim > regime_vec.shape[0]:
            context_timeframes = normalize_context_timeframes(
                member.ckpt.get("context_timeframes") or os.getenv("ANALYST_CONTEXT_TIMEFRAMES", "5m,15m,1h,4h,1d")
            )
            ctx_payload = dict(
                self.repo.load_multi_timeframe_context(
                    symbol=symbol,
                    as_of=as_of,
                    timeframes=context_timeframes,
                    primary_timeframe=str(os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m")).strip().lower() or "5m",
                )
                or {}
            )
            if str(ctx_payload.get("source") or "") != "market_context_multi_tf":
                raise RuntimeError(f"multi_tf_context_source_invalid:{ctx_payload.get('source')}")
            mtf_vec, mtf_msk = build_multi_tf_vector_from_payload(
                ctx_payload,
                timeframes=context_timeframes,
                require_complete=True,
            )
            mtf_names = build_multi_tf_feature_names(context_timeframes)
            for i, key in enumerate(mtf_names):
                regime_feature_map[str(key)] = float(mtf_vec[i])
            regime_vec = np.concatenate([regime_vec, mtf_vec], axis=0).astype(np.float32)
            regime_msk = np.concatenate([regime_msk, mtf_msk], axis=0).astype(np.uint8)

        if regime_vec.shape[0] != expected_regime_dim:
            raise RuntimeError(f"regime_dim_mismatch_inference:{regime_vec.shape[0]}:{expected_regime_dim}")
        return regime_vec, regime_msk, regime_feature_map

    def _predict_one_member(
        self,
        *,
        member: _LoadedModelBundle,
        x_values: np.ndarray,
        x_mask: np.ndarray,
        symbol: str,
        as_of: datetime,
        base_regime_vec: np.ndarray,
        base_regime_mask: np.ndarray,
    ) -> Dict[str, Any]:
        xv = torch.tensor(np.asarray(x_values, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        xm = torch.tensor(np.asarray(x_mask, dtype=np.float32), dtype=torch.float32).unsqueeze(0)

        symbol_to_id = member.ckpt.get("symbol_to_id") if isinstance(member.ckpt.get("symbol_to_id"), dict) else {}
        if symbol_to_id:
            if symbol not in symbol_to_id:
                raise RuntimeError(f"symbol_not_in_universe_snapshot:{symbol}:{member.model_dir}")
            sid = int(symbol_to_id[symbol])
        else:
            sid = 0

        regime_vec, regime_msk, regime_feature_map = self._build_regime_for_member(
            member=member,
            base_regime_vec=base_regime_vec,
            base_regime_mask=base_regime_mask,
            symbol=symbol,
            as_of=as_of,
        )
        regime_last = torch.tensor(regime_vec, dtype=torch.float32).unsqueeze(0)
        regime_mask_last = torch.tensor(regime_msk.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        sid_t = torch.tensor([sid], dtype=torch.long)

        with torch.no_grad():
            pred = member.model(
                xv,
                xm,
                symbol_id=sid_t,
                regime_features=regime_last,
                regime_mask=regime_mask_last,
            )

        mu = pred.mu.squeeze(0).cpu().numpy().astype(np.float64)
        sigma = torch.exp(pred.log_sigma).squeeze(0).cpu().numpy().astype(np.float64)
        q = pred.q.squeeze(0).cpu().numpy().astype(np.float64) if pred.q is not None else None
        dlogit = pred.direction_logit.squeeze(0).cpu().numpy().astype(np.float64) if pred.direction_logit is not None else None
        conf, sigma_cal, cal = self._calibrated_confidence_for_ckpt(member.ckpt, mu=mu, sigma=sigma, direction_logit=dlogit)

        expert_vec = None
        if pred.expert_weights is not None:
            raw = pred.expert_weights.squeeze(0).cpu().numpy().astype(np.float64)
            if raw.ndim > 1:
                raw = np.mean(raw, axis=0)
            s = float(np.sum(raw))
            if s > 0:
                raw = raw / s
            expert_vec = raw

        regime_probs_vec = None
        if pred.regime_probs is not None:
            raw = pred.regime_probs.squeeze(0).cpu().numpy().astype(np.float64)
            if raw.ndim > 1:
                raw = np.mean(raw, axis=0)
            s = float(np.sum(raw))
            if s > 0:
                raw = raw / s
            regime_probs_vec = raw

        horizons = self._horizon_list(member.ckpt)
        gate_val = 0.0
        if isinstance(pred.aux, dict) and isinstance(pred.aux.get("gate"), torch.Tensor):
            gate_val = float(np.mean(pred.aux["gate"].detach().cpu().numpy()))

        return {
            "horizons": horizons,
            "mu": mu,
            "sigma": sigma_cal,
            "q": q,
            "direction_logit": dlogit,
            "confidence": conf,
            "expert_vec": expert_vec,
            "regime_probs_vec": regime_probs_vec,
            "regime_features": regime_feature_map,
            "gate": gate_val,
            "model_id": str(member.manifest.get("model_id") or self.default_model_name),
            "model_dir": str(member.model_dir),
            "backbone": str(member.ckpt.get("backbone_name") or "unknown"),
            "calibration": cal,
        }

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))

    def _aggregate_member_predictions(
        self,
        *,
        members_pred: Sequence[Dict[str, Any]],
        horizon: str,
        coverage_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not members_pred:
            raise RuntimeError("ensemble_predictions_empty")

        horizons = list(members_pred[0]["horizons"])
        h_count = len(horizons)
        for row in members_pred:
            if list(row["horizons"]) != horizons:
                raise RuntimeError("ensemble_horizon_alignment_failed")

        w = np.asarray(self.ensemble_weights, dtype=np.float64)
        if w.shape[0] != len(members_pred):
            raise RuntimeError("ensemble_weights_member_count_mismatch")

        mu_stack = np.stack([np.asarray(r["mu"], dtype=np.float64) for r in members_pred], axis=0)
        sigma_stack = np.stack([np.asarray(r["sigma"], dtype=np.float64) for r in members_pred], axis=0)
        conf_stack = np.stack([np.asarray(r["confidence"], dtype=np.float64) for r in members_pred], axis=0)
        mu = np.tensordot(w, mu_stack, axes=(0, 0)).astype(np.float64)
        conf = np.clip(np.tensordot(w, conf_stack, axes=(0, 0)).astype(np.float64), 0.01, 0.99)
        second = np.tensordot(w, sigma_stack * sigma_stack + mu_stack * mu_stack, axes=(0, 0)).astype(np.float64)
        sigma = np.sqrt(np.clip(second - mu * mu, 1e-12, None))
        dlogit = self._logit(conf)

        q_avail = [r["q"] for r in members_pred if isinstance(r.get("q"), np.ndarray)]
        q = None
        if len(q_avail) == len(members_pred):
            q_stack = np.stack([np.asarray(r["q"], dtype=np.float64) for r in members_pred], axis=0)
            q = np.tensordot(w, q_stack, axes=(0, 0)).astype(np.float64)
            q = np.sort(q, axis=-1)

        expert_vec = None
        if all(isinstance(r.get("expert_vec"), np.ndarray) for r in members_pred):
            e_stack = np.stack([np.asarray(r["expert_vec"], dtype=np.float64) for r in members_pred], axis=0)
            expert_vec = np.tensordot(w, e_stack, axes=(0, 0)).astype(np.float64)
            s = float(np.sum(expert_vec))
            if s > 0:
                expert_vec = expert_vec / s

        regime_probs = None
        if all(isinstance(r.get("regime_probs_vec"), np.ndarray) for r in members_pred):
            r_stack = np.stack([np.asarray(r["regime_probs_vec"], dtype=np.float64) for r in members_pred], axis=0)
            regime_probs = np.tensordot(w, r_stack, axes=(0, 0)).astype(np.float64)
            s = float(np.sum(regime_probs))
            if s > 0:
                regime_probs = regime_probs / s

        if q is not None and q.ndim >= 2 and q.shape[-1] >= 2:
            vol_map = {h: float(max(1e-6, (q[i, -1] - q[i, 0]) / 2.56)) for i, h in enumerate(horizons)}
        else:
            vol_map = {h: float(max(1e-6, sigma[i])) for i, h in enumerate(horizons)}

        pred_map = {h: float(mu[i]) for i, h in enumerate(horizons)}
        conf_map = {h: float(conf[i]) for i, h in enumerate(horizons)}
        dlogit_map = {h: float(dlogit[i]) for i, h in enumerate(horizons)}

        quantiles_map = {}
        if q is not None and q.ndim >= 2:
            for i, h in enumerate(horizons):
                p10 = float(q[i, 0])
                p50 = float(q[i, q.shape[-1] // 2])
                p90 = float(q[i, -1])
                quantiles_map[h] = {"p10": p10, "p50": p50, "p90": p90}

        expert_map = {}
        if isinstance(expert_vec, np.ndarray):
            names = ["trend", "mean_reversion", "liquidation_risk", "neutral"]
            for h in horizons:
                expert_map[h] = {names[i]: float(expert_vec[i]) for i in range(min(len(names), expert_vec.shape[0]))}

        regime_map = {}
        if isinstance(regime_probs, np.ndarray):
            names = ["trend", "crowding", "liquidation"]
            for h in horizons:
                regime_map[h] = {names[i]: float(regime_probs[i]) for i in range(min(len(names), regime_probs.shape[0]))}

        regime_features_map = dict(members_pred[0].get("regime_features") or {})
        regime_features_h = {h: dict(regime_features_map) for h in horizons}

        h = str(horizon or "1h").strip().lower()
        if h not in pred_map:
            h = "1h" if "1h" in pred_map else sorted(pred_map.keys())[0]

        member_meta = [
            {
                "model_id": str(r.get("model_id") or ""),
                "model_dir": str(r.get("model_dir") or ""),
                "backbone": str(r.get("backbone") or ""),
                "weight": float(w[i]),
                "calibration": dict(r.get("calibration") or {}),
            }
            for i, r in enumerate(members_pred)
        ]

        gate = float(np.sum(w * np.asarray([float(r.get("gate") or 0.0) for r in members_pred], dtype=np.float64)))
        stack = {
            "model_id": str(members_pred[0].get("model_id") or self.default_model_name),
            "schema_hash": SCHEMA_HASH,
            "backbone": "ensemble" if self.ensemble_enabled else str(members_pred[0].get("backbone") or "unknown"),
            "calibration": {
                "mode": "ensemble_weighted",
                "members": member_meta,
            },
            "coverage_summary": dict(coverage_summary or {}),
            "gate": gate,
            "per_horizon_uncertainty": {hh: float(sigma[i]) for i, hh in enumerate(horizons)},
            "ensemble": {
                "enabled": bool(self.ensemble_enabled),
                "member_count": int(len(members_pred)),
                "weights": [float(x) for x in w.tolist()],
                "members": member_meta,
            },
        }

        return {
            "expected_return": float(pred_map[h]),
            "signal_confidence": float(conf_map[h]),
            "vol_forecast": float(vol_map[h]),
            "expected_return_horizons": pred_map,
            "signal_confidence_horizons": conf_map,
            "vol_forecast_horizons": vol_map,
            "direction_logit_horizons": dlogit_map,
            "quantiles_horizons": quantiles_map,
            "expert_weights_horizons": expert_map,
            "regime_probs_horizons": regime_map,
            "regime_features_horizons": regime_features_h,
            "stack": stack,
            "model_name": str(members_pred[0].get("model_id") or self.default_model_name),
            "model_version": str(self.default_model_version),
            "degraded": False,
            "degraded_reasons": [],
            "horizons": list(horizons),
            "member_count": int(len(members_pred)),
        }

    def _predict_sequence(self, *, symbol: str, horizon: str, as_of: datetime) -> Dict[str, Any]:
        seq = fetch_sequence(
            db_url=str(self.repo.db_url),
            symbol=symbol,
            end_ts=as_of,
            lookback=self.lookback,
        )
        if str(seq["schema_hash"]) != SCHEMA_HASH:
            raise RuntimeError("schema_hash_mismatch_sequence")

        values_np = np.asarray(seq["values"], dtype=np.float32)
        mask_np = np.asarray(seq["mask"], dtype=np.uint8)
        base_regime_np, base_regime_mask_np = compute_regime_features(values_np, mask_np)
        base_regime_vec = np.asarray(base_regime_np[-1], dtype=np.float32)
        base_regime_mask = np.asarray(base_regime_mask_np[-1], dtype=np.uint8)

        members_pred: List[Dict[str, Any]] = []
        for member in self.ensemble_members:
            members_pred.append(
                self._predict_one_member(
                    member=member,
                    x_values=values_np,
                    x_mask=mask_np,
                    symbol=symbol,
                    as_of=as_of,
                    base_regime_vec=base_regime_vec,
                    base_regime_mask=base_regime_mask,
                )
            )

        return self._aggregate_member_predictions(
            members_pred=members_pred,
            horizon=horizon,
            coverage_summary=dict(seq.get("coverage_summary") or {}),
        )

    def predict_from_feature_payload(
        self,
        *,
        target: str,
        payload: Dict[str, Any],
        horizon: str = "1d",
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        require_artifact: bool = True,
    ) -> Dict[str, Any]:
        _ = payload
        _ = model_name
        _ = model_version
        _ = require_artifact
        return self._predict_sequence(symbol=str(target or "").upper(), horizon=horizon, as_of=self._utcnow())

    def predict_with_context(self, *, symbol: str, horizon: str) -> Dict[str, Any]:
        target = str(symbol or "").strip().upper()
        if not target:
            raise HTTPException(status_code=400, detail="symbol is required")
        price_row = self.repo.latest_price_snapshot(target)
        if not price_row:
            raise HTTPException(status_code=404, detail=f"no price snapshot for {target}")

        try:
            pred = self._predict_sequence(symbol=target, horizon=horizon, as_of=self._utcnow())
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        payload = self.latest_feature_payload(target, max_lookback_days=30)
        context = self.repo.recent_event_context(target, limit=8)
        explanation = {
            "top_event_contributors": [
                {
                    "event_id": e.get("id"),
                    "event_type": e.get("event_type"),
                    "title": e.get("title"),
                    "weight": round(0.1 + float(e.get("confidence_score") or 0.5) * 0.7, 3),
                }
                for e in context[:5]
            ],
            "top_feature_contributors": [
                {"feature": "ret_12", "value": round(float(payload.get("ret_12", 0.0)), 6), "contribution": round(float(payload.get("ret_12", 0.0)), 6)},
                {"feature": "vol_12", "value": round(float(payload.get("vol_12", 0.0)), 6), "contribution": round(-abs(float(payload.get("vol_12", 0.0))), 6)},
                {"feature": "orderbook_imbalance", "value": round(float(payload.get("orderbook_imbalance", 0.0)), 6), "contribution": round(float(payload.get("orderbook_imbalance", 0.0)) * 0.5, 6)},
            ],
            "evidence_links": [str(e.get("source_url") or "") for e in context if str(e.get("source_url") or "").strip()][:5],
            "model_version": f"{pred['model_name']}:{pred['model_version']}",
            "feature_version": self.feature_version,
        }
        outputs = {
            "expected_return": round(float(pred["expected_return"]), 6),
            "vol_forecast": round(float(pred["vol_forecast"]), 6),
            "signal_confidence": round(float(pred["signal_confidence"]), 4),
            "expected_return_horizons": {str(k): round(float(v), 6) for k, v in dict(pred.get("expected_return_horizons") or {}).items()},
            "signal_confidence_horizons": {str(k): round(float(v), 4) for k, v in dict(pred.get("signal_confidence_horizons") or {}).items()},
            "vol_forecast_horizons": {str(k): round(float(v), 6) for k, v in dict(pred.get("vol_forecast_horizons") or {}).items()},
            "direction_logit_horizons": {str(k): round(float(v), 6) for k, v in dict(pred.get("direction_logit_horizons") or {}).items()},
            "quantiles_horizons": dict(pred.get("quantiles_horizons") or {}),
            "current_price": float(price_row.get("price") or 0.0),
            "horizon": str(horizon),
            "as_of": self._utcnow().isoformat(),
            "model_name": pred["model_name"],
            "model_version": pred["model_version"],
            "score_source": "model",
            "stack": pred["stack"],
            "degraded": False,
            "degraded_reasons": [],
        }
        return {
            "target": target,
            "score": float(round(float(pred["expected_return"]), 6)),
            "confidence": float(round(float(pred["signal_confidence"]), 4)),
            "outputs": outputs,
            "explanation": explanation,
            "model_name": str(pred["model_name"]),
            "model_version": str(pred["model_version"]),
        }
