from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import HTTPException
import torch

from mlops_artifacts.validate import validate_manifest_dir
from vc.feature_spec import VC_FEATURE_KEYS, VC_SCHEMA_HASH, vector_from_context


VC_HORIZONS = ("6m", "12m", "24m")


class _VCModel(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
        )
        self.round_logits = torch.nn.Linear(32, len(VC_HORIZONS))
        self.exit_logit = torch.nn.Linear(32, 1)
        self.moic_mu = torch.nn.Linear(32, 1)
        self.moic_log_sigma = torch.nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "round_logits": self.round_logits(h),
            "exit_logit": self.exit_logit(h).squeeze(-1),
            "moic_mu": self.moic_mu(h).squeeze(-1),
            "moic_log_sigma": self.moic_log_sigma(h).squeeze(-1).clamp(min=-4.0, max=2.0),
        }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class VCModelService:
    def __init__(self, *, repo: Any):
        self.repo = repo
        self.model_dir = Path(str(os.getenv("VC_MODEL_DIR", "artifacts/models/vc_main")))
        self.model, self.manifest, self.ckpt = self._load_model_or_fail()

    def _load_model_or_fail(self):
        manifest = validate_manifest_dir(self.model_dir, expected_schema_hash=VC_SCHEMA_HASH)
        weights = self.model_dir / manifest["files"]["weights"]
        ckpt = torch.load(weights, map_location="cpu")
        if str(ckpt.get("schema_hash") or "") != VC_SCHEMA_HASH:
            raise RuntimeError("vc_schema_hash_mismatch")
        in_dim = int(ckpt.get("in_dim") or 0)
        if in_dim <= 0:
            raise RuntimeError("vc_invalid_in_dim")
        model = _VCModel(in_dim=in_dim)
        state = ckpt.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("vc_state_dict_missing")
        model.load_state_dict(state)
        model.eval()
        return model, manifest, ckpt

    def _build_features(self, company_name: str) -> np.ndarray:
        events = self.repo.recent_event_context(company_name, limit=24)
        return vector_from_context(list(events or []))

    def predict_with_context(self, *, company_name: str, horizon_months: int) -> Dict[str, Any]:
        target = str(company_name or "").strip()
        if not target:
            raise HTTPException(status_code=400, detail="company_name_required")
        feats = self._build_features(target)
        with torch.no_grad():
            out = self.model(torch.tensor(feats, dtype=torch.float32).unsqueeze(0))
        round_logits = out["round_logits"].squeeze(0).cpu().numpy().astype(np.float64)
        exit_logit = float(out["exit_logit"].squeeze(0).cpu().item())
        moic_mu = float(out["moic_mu"].squeeze(0).cpu().item())
        moic_sigma = float(np.exp(float(out["moic_log_sigma"].squeeze(0).cpu().item())))

        round_temp = self.ckpt.get("round_temperature") if isinstance(self.ckpt.get("round_temperature"), dict) else {}
        temps = np.array([float(round_temp.get(h, 1.0) or 1.0) for h in VC_HORIZONS], dtype=np.float64)
        probs_round = _sigmoid(round_logits / np.clip(temps, 1e-6, None))
        probs_round = np.clip(probs_round, 0.01, 0.99)
        p_map = {VC_HORIZONS[i]: float(probs_round[i]) for i in range(len(VC_HORIZONS))}

        exit_temp = float(self.ckpt.get("exit_temperature", 1.0) or 1.0)
        p_exit_24m = float(np.clip(_sigmoid(np.array([exit_logit / max(1e-6, exit_temp)], dtype=np.float64))[0], 0.01, 0.99))

        sigma_scale = float(self.ckpt.get("moic_sigma_scale", 1.0) or 1.0)
        sigma_cal = max(1e-6, moic_sigma * sigma_scale)
        q10 = moic_mu + sigma_cal * (-1.2815516)
        q50 = moic_mu
        q90 = moic_mu + sigma_cal * (1.2815516)
        moic_dist = {
            "p10": round(float(max(0.1, q10)), 3),
            "p50": round(float(max(0.1, q50)), 3),
            "p90": round(float(max(0.1, q90)), 3),
        }

        outputs = {
            "p_next_round": p_map,
            "p_exit_24m": p_exit_24m,
            "expected_moic_distribution": moic_dist,
            "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_id": str(self.manifest.get("model_id") or "vc_main"),
            "schema_hash": VC_SCHEMA_HASH,
            "round_logits": {VC_HORIZONS[i]: float(round_logits[i]) for i in range(len(VC_HORIZONS))},
            "exit_logit": float(exit_logit),
            "calibration": {
                "round_temperature": {VC_HORIZONS[i]: float(temps[i]) for i in range(len(VC_HORIZONS))},
                "exit_temperature": exit_temp,
                "moic_sigma_scale": sigma_scale,
            },
            "feature_keys": list(self.ckpt.get("feature_keys") or VC_FEATURE_KEYS),
        }
        top_events = list(self.repo.recent_event_context(target, limit=5) or [])
        explanation = {
            "top_event_contributors": top_events,
            "top_feature_contributors": [
                {"feature": "event_type_score", "value": float(feats[0]), "contribution": float(feats[0]) * 0.25},
                {"feature": "source_tier_norm", "value": float(feats[1]), "contribution": float(feats[1]) * 0.20},
                {"feature": "confidence_score", "value": float(feats[2]), "contribution": float(feats[2]) * 0.20},
                {"feature": "event_importance", "value": float(feats[3]), "contribution": float(feats[3]) * 0.20},
                {"feature": "novelty_score", "value": float(feats[4]), "contribution": float(feats[4]) * 0.15},
            ],
            "evidence_links": [str(e.get("source_url") or "") for e in top_events if str(e.get("source_url") or "").strip()],
            "model_version": str(self.manifest.get("model_id") or "vc_main"),
            "feature_version": "vc_schema_main",
        }

        hm = int(horizon_months)
        horizon_key = "12m"
        if hm <= 6:
            horizon_key = "6m"
        elif hm >= 24:
            horizon_key = "24m"

        return {
            "target": target,
            "score": float(round(p_map[horizon_key], 4)),
            "confidence": float(round(p_map[horizon_key], 4)),
            "outputs": outputs,
            "explanation": explanation,
        }
