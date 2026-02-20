from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

_ROOT = Path(__file__).resolve().parents[1]
_INFER_DIR = _ROOT / "inference"
if str(_INFER_DIR) not in sys.path:
    sys.path.append(str(_INFER_DIR))

try:
    from model_router import ModelRouter  # type: ignore
except Exception:
    ModelRouter = None  # type: ignore[assignment]


class LiquidModelService:
    def __init__(
        self,
        *,
        repo: Any,
        feature_keys: List[str],
        feature_version: str,
        data_version: str,
        default_model_name: str = "liquid_ttm_ensemble",
        default_model_version: str = "v2.1",
    ):
        self.repo = repo
        self.feature_keys = list(feature_keys)
        self.feature_version = str(feature_version)
        self.data_version = str(data_version)
        self.default_model_name = str(default_model_name)
        self.default_model_version = str(default_model_version)
        self._model_router: Optional[Any] = None
        self.require_neural_artifact = str(os.getenv("LIQUID_REQUIRE_NN_ARTIFACT", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    @staticmethod
    def _default_model_dir() -> Path:
        env_path = str(os.getenv("MODEL_DIR", "")).strip()
        if env_path:
            return Path(env_path)
        local_models = Path(__file__).resolve().parent / "models"
        if local_models.exists():
            return local_models
        return Path("/opt/monitoring-system/models")

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _get_model_router(self) -> Any:
        if self._model_router is not None:
            return self._model_router
        if ModelRouter is None:
            raise RuntimeError("model_router_unavailable")
        self._model_router = ModelRouter()
        return self._model_router

    def _active_liquid_model(self) -> Tuple[str, str]:
        try:
            row = self.repo.get_active_model_state("liquid")
        except Exception:
            row = None
        if isinstance(row, dict):
            name = str(row.get("active_model_name") or "").strip()
            ver = str(row.get("active_model_version") or "").strip()
            if name and ver:
                return name, ver
        return self.default_model_name, self.default_model_version

    def latest_feature_payload(self, symbol: str, *, max_lookback_days: int = 30) -> Dict[str, float]:
        target = str(symbol or "").strip().upper()
        if not target:
            return {}
        rows = self.repo.load_feature_history(
            target=target,
            track="liquid",
            lookback_days=max(1, int(max_lookback_days)),
            data_version=self.data_version,
            limit=20000,
        )
        if not rows:
            rows = self.repo.load_feature_history(
                target=target,
                track="liquid",
                lookback_days=max(1, int(max_lookback_days)),
                data_version=None,
                limit=20000,
            )
        for row in reversed(rows):
            payload = row.get("feature_payload") if isinstance(row, dict) else None
            if isinstance(payload, dict) and payload:
                return {k: float(payload.get(k, 0.0) or 0.0) for k in self.feature_keys}
        return {}

    def _feature_vector_from_payload(self, payload: Dict[str, Any]) -> np.ndarray:
        return np.array([float(payload.get(k, 0.0) or 0.0) for k in self.feature_keys], dtype=np.float32)

    def _has_neural_artifact(self, *, target: str, model_router: Optional[Any] = None) -> bool:
        sym = str(target or "").strip().upper()
        if not sym:
            return False
        router = model_router or self._get_model_router()
        nn_path = os.path.join(str(self._default_model_dir()), f"liquid_{sym.lower()}_tsmixer_v2.pt")
        nn = router._load_torch_model(nn_path, "liquid", len(self.feature_keys))  # type: ignore[attr-defined]
        return nn is not None

    def has_required_artifacts(self, *, target: str, model_name: str) -> bool:
        sym = str(target or "").strip().upper()
        if not sym:
            return False
        model_router = self._get_model_router()
        tab = model_router._load_liquid_tabular_model(sym)  # type: ignore[attr-defined]
        if tab is None:
            return False
        if model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
            if self._has_neural_artifact(target=sym, model_router=model_router):
                return True
            return not bool(self.require_neural_artifact)
        return True

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
        sym = str(target or "").strip().upper()
        if not sym:
            raise RuntimeError("target_required")
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"feature_payload_unavailable:{sym}")
        use_model_name = str(model_name or "").strip()
        use_model_version = str(model_version or "").strip()
        if not use_model_name or not use_model_version:
            use_model_name, use_model_version = self._active_liquid_model()
        if require_artifact and not self.has_required_artifacts(target=sym, model_name=use_model_name):
            if use_model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
                if not self.has_required_artifacts(target=sym, model_name="liquid_baseline"):
                    raise RuntimeError(f"model_artifact_missing:tabular:{sym}")
                raise RuntimeError(f"model_artifact_missing:neural:{sym}")
            raise RuntimeError(f"model_artifact_missing:tabular:{sym}")
        model_router = self._get_model_router()
        vec = self._feature_vector_from_payload(payload)
        out = model_router.predict_liquid(sym, vec, model_name=use_model_name)
        scale = {"1h": 0.3, "1d": 1.0, "7d": 2.1}.get(str(horizon).strip().lower(), 1.0)
        expected_return = float(out.get("expected_return") or 0.0) * float(scale)
        signal_confidence = float(out.get("signal_confidence") or 0.0)
        vol_forecast = float(out.get("vol_forecast") or 0.0)
        stack = out.get("stack") if isinstance(out.get("stack"), dict) else {}
        degraded_reasons: List[str] = []
        if use_model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
            neural_present = self._has_neural_artifact(target=sym, model_router=model_router)
            stack = {**stack, "neural_artifact_present": bool(neural_present)}
            if (not neural_present) and (not self.require_neural_artifact):
                degraded_reasons.append("neural_artifact_missing_tabular_fallback")
        degraded = bool(degraded_reasons)
        return {
            "expected_return": float(expected_return),
            "signal_confidence": float(signal_confidence),
            "vol_forecast": float(vol_forecast),
            "stack": stack,
            "model_name": use_model_name,
            "model_version": use_model_version,
            "degraded": degraded,
            "degraded_reasons": degraded_reasons,
        }

    def predict_with_context(self, *, symbol: str, horizon: str) -> Dict[str, Any]:
        target = str(symbol or "").strip().upper()
        if not target:
            raise HTTPException(status_code=400, detail="symbol is required")
        price_row = self.repo.latest_price_snapshot(target)
        if not price_row:
            raise HTTPException(status_code=404, detail=f"no price snapshot for {target}")
        payload = self.latest_feature_payload(target, max_lookback_days=30)
        if not payload:
            raise HTTPException(status_code=424, detail=f"feature_payload_unavailable:{target}")
        try:
            pred = self.predict_from_feature_payload(target=target, payload=payload, horizon=horizon, require_artifact=True)
        except RuntimeError as exc:
            detail = str(exc)
            if detail.startswith("model_artifact_missing:"):
                raise HTTPException(status_code=503, detail=detail) from exc
            raise HTTPException(status_code=500, detail=detail) from exc
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
                {
                    "feature": "orderbook_imbalance",
                    "value": round(float(payload.get("orderbook_imbalance", 0.0)), 6),
                    "contribution": round(float(payload.get("orderbook_imbalance", 0.0)) * 0.5, 6),
                },
            ],
            "evidence_links": [str(e.get("source_url") or "") for e in context if str(e.get("source_url") or "").strip()][:5],
            "model_version": f"{pred['model_name']}:{pred['model_version']}",
            "feature_version": self.feature_version,
        }
        outputs = {
            "expected_return": round(float(pred["expected_return"]), 6),
            "vol_forecast": round(float(pred["vol_forecast"]), 6),
            "signal_confidence": round(float(pred["signal_confidence"]), 4),
            "current_price": float(price_row.get("price") or 0.0),
            "horizon": str(horizon),
            "as_of": self._utcnow().isoformat(),
            "model_name": pred["model_name"],
            "model_version": pred["model_version"],
            "score_source": "model",
            "stack": pred["stack"],
            "degraded": bool(pred.get("degraded", False)),
            "degraded_reasons": list(pred.get("degraded_reasons") or []),
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
