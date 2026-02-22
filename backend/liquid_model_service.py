from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException
import torch

from artifacts.validate import validate_manifest_dir
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from inference.feature_reader import fetch_sequence
from liquid_model_registry import get_active_model as get_registry_active_model
from models.liquid_model import build_liquid_model_from_checkpoint


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

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _load_model_or_fail(self):
        manifest = validate_manifest_dir(self.model_dir, expected_schema_hash=SCHEMA_HASH)
        weights = self.model_dir / manifest["files"]["weights"]
        ckpt = torch.load(weights, map_location="cpu")
        if str(ckpt.get("schema_hash") or "") != SCHEMA_HASH:
            raise RuntimeError("schema_hash_mismatch_weights")
        lookback = int(ckpt.get("lookback") or 0)
        if lookback <= 0:
            raise RuntimeError("invalid_lookback")
        fdim = int(ckpt.get("feature_dim") or 0)
        if fdim != FEATURE_DIM:
            raise RuntimeError(f"feature_dim_mismatch_weights:{fdim}:{FEATURE_DIM}")

        model = build_liquid_model_from_checkpoint(ckpt)
        model.eval()
        return model, manifest, ckpt, lookback

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
            validate_manifest_dir(self.model_dir, expected_schema_hash=SCHEMA_HASH)
            return True
        except Exception:
            return False

    def _calibrated_confidence(self, mu: np.ndarray, sigma: np.ndarray, direction_logit: Optional[np.ndarray]) -> np.ndarray:
        cal = self.ckpt.get("calibration") if isinstance(self.ckpt.get("calibration"), dict) else {}
        sigma_scale = float(cal.get("sigma_scale", 1.0) or 1.0)
        direction_temperature = float(cal.get("direction_temperature", 1.0) or 1.0)

        sigma_cal = np.clip(sigma * sigma_scale, 1e-6, None)
        if direction_logit is not None:
            z = direction_logit / max(1e-6, direction_temperature)
        else:
            z = mu / sigma_cal
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        return np.clip(p, 0.01, 0.99)

    def _predict_sequence(self, *, symbol: str, horizon: str, as_of: datetime) -> Dict[str, Any]:
        seq = fetch_sequence(
            db_url=str(self.repo.db_url),
            symbol=symbol,
            end_ts=as_of,
            lookback=self.lookback,
        )
        if str(seq["schema_hash"]) != SCHEMA_HASH:
            raise RuntimeError("schema_hash_mismatch_sequence")

        xv = torch.tensor(np.array(seq["values"], dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        xm = torch.tensor(np.array(seq["mask"], dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(xv, xm)

        mu = pred.mu.squeeze(0).cpu().numpy().astype(np.float64)
        sigma = torch.exp(pred.log_sigma).squeeze(0).cpu().numpy().astype(np.float64)
        q = pred.q.squeeze(0).cpu().numpy().astype(np.float64) if pred.q is not None else None
        dlogit = pred.direction_logit.squeeze(0).cpu().numpy().astype(np.float64) if pred.direction_logit is not None else None

        conf = self._calibrated_confidence(mu=mu, sigma=sigma, direction_logit=dlogit)
        cal = self.ckpt.get("calibration") if isinstance(self.ckpt.get("calibration"), dict) else {}
        sigma_scale = float(cal.get("sigma_scale", 1.0) or 1.0)
        sigma_map = sigma * sigma_scale

        horizons = [str(h) for h in list(self.ckpt.get("horizons") or ["1h", "4h", "1d", "7d"])]
        pred_map = {h: float(mu[i]) for i, h in enumerate(horizons)}
        conf_map = {h: float(conf[i]) for i, h in enumerate(horizons)}
        if q is not None and q.shape[2] >= 3:
            vol_map = {h: float(max(1e-6, (q[i, -1] - q[i, 0]) / 2.56)) for i, h in enumerate(horizons)}
        else:
            vol_map = {h: float(max(1e-6, sigma_map[i])) for i, h in enumerate(horizons)}

        quantiles_map = None
        if q is not None:
            quantiles_map = {
                h: {
                    "p10": float(q[i, 0]),
                    "p50": float(q[i, 1] if q.shape[2] > 1 else q[i, 0]),
                    "p90": float(q[i, -1]),
                }
                for i, h in enumerate(horizons)
            }

        h = str(horizon or "1h").strip().lower()
        if h not in pred_map:
            h = "1h" if "1h" in pred_map else sorted(pred_map.keys())[0]

        return {
            "expected_return": float(pred_map[h]),
            "signal_confidence": float(conf_map[h]),
            "vol_forecast": float(vol_map[h]),
            "expected_return_horizons": pred_map,
            "signal_confidence_horizons": conf_map,
            "vol_forecast_horizons": vol_map,
            "direction_logit_horizons": {hh: float(dlogit[i]) for i, hh in enumerate(horizons)} if dlogit is not None else {},
            "quantiles_horizons": quantiles_map or {},
            "stack": {
                "model_id": str(self.manifest.get("model_id") or "liquid_main"),
                "schema_hash": SCHEMA_HASH,
                "backbone": str(self.ckpt.get("backbone_name") or "patchtst"),
                "calibration": cal,
                "coverage_summary": dict(seq.get("coverage_summary") or {}),
                "gate": float(np.mean(pred.aux["gate"].detach().cpu().numpy())) if isinstance(pred.aux, dict) and isinstance(pred.aux.get("gate"), torch.Tensor) else 0.0,
            },
            "model_name": str(self.manifest.get("model_id") or self.default_model_name),
            "model_version": str(self.default_model_version),
            "degraded": False,
            "degraded_reasons": [],
        }

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
