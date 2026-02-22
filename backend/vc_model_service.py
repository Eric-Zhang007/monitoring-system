from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import HTTPException
import torch

from artifacts.validate import validate_manifest_dir
from features.feature_contract import SCHEMA_HASH
from vc.feature_spec import VC_FEATURE_KEYS, vector_from_context


class _VCModel(torch.nn.Module):
    def __init__(self, in_dim: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VCModelService:
    def __init__(self, *, repo: Any):
        self.repo = repo
        self.model_dir = Path(str(os.getenv("VC_MODEL_DIR", "artifacts/models/vc_main")))
        self.model, self.manifest, self.ckpt = self._load_model_or_fail()

    def _load_model_or_fail(self):
        manifest = validate_manifest_dir(self.model_dir, expected_schema_hash=SCHEMA_HASH)
        weights = self.model_dir / manifest["files"]["weights"]
        ckpt = torch.load(weights, map_location="cpu")
        if str(ckpt.get("schema_hash") or "") != SCHEMA_HASH:
            raise RuntimeError("vc_schema_hash_mismatch")
        in_dim = int(ckpt.get("in_dim") or 0)
        if in_dim <= 0:
            raise RuntimeError("vc_invalid_in_dim")
        model = _VCModel(in_dim=in_dim)
        model.load_state_dict(ckpt["state_dict"])
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
            logits = self.model(torch.tensor(feats, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            logit_pos = float(logits[1].cpu().item())
            temp = float(self.ckpt.get("temperature", 1.0) or 1.0)
            prob_pos = float(torch.sigmoid(torch.tensor(logit_pos / max(1e-6, temp))).item())

        p_round = float(np.clip(prob_pos, 0.01, 0.99))
        # Keep horizon view deterministic but model-driven from same calibrated probability.
        p_map = {
            "6m": float(np.clip(p_round * 1.05, 0.01, 0.99)),
            "12m": p_round,
            "24m": float(np.clip(p_round * 0.92, 0.01, 0.99)),
        }

        outputs = {
            "p_next_round": p_map,
            "p_exit_24m": float(np.clip(1.0 - p_round * 0.7, 0.01, 0.95)),
            "expected_moic_distribution": {
                "p10": round(0.6 + p_round * 0.9, 2),
                "p50": round(1.0 + p_round * 1.8, 2),
                "p90": round(1.4 + p_round * 3.2, 2),
            },
            "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_id": str(self.manifest.get("model_id") or "vc_main"),
            "schema_hash": SCHEMA_HASH,
            "direction_logit": logit_pos,
            "calibration": {"temperature": temp},
            "feature_keys": list(self.ckpt.get("feature_keys") or VC_FEATURE_KEYS),
        }
        explanation = {
            "top_event_contributors": self.repo.recent_event_context(target, limit=5),
            "top_feature_contributors": [
                {"feature": "event_type_score", "value": float(feats[0]), "contribution": float(feats[0]) * 0.3},
                {"feature": "source_tier_norm", "value": float(feats[1]), "contribution": float(feats[1]) * 0.2},
                {"feature": "confidence_score", "value": float(feats[2]), "contribution": float(feats[2]) * 0.2},
                {"feature": "event_importance", "value": float(feats[3]), "contribution": float(feats[3]) * 0.2},
                {"feature": "novelty_score", "value": float(feats[4]), "contribution": float(feats[4]) * 0.1},
            ],
            "evidence_links": [str(e.get("source_url") or "") for e in self.repo.recent_event_context(target, limit=5) if str(e.get("source_url") or "").strip()],
            "model_version": str(self.manifest.get("model_id") or "vc_main"),
            "feature_version": "schema_main",
        }

        horizon_key = "12m"
        hm = int(horizon_months)
        if hm <= 6:
            horizon_key = "6m"
        elif hm >= 24:
            horizon_key = "24m"

        return {
            "target": target,
            "score": float(round(p_map[horizon_key], 4)),
            "confidence": float(round(p_round, 4)),
            "outputs": outputs,
            "explanation": explanation,
        }
