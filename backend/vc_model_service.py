from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import HTTPException

from artifacts.validate import validate_manifest_dir
from features.feature_contract import SCHEMA_HASH

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


if torch is not None:
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
else:  # pragma: no cover
    class _VCModel:  # type: ignore[no-redef]
        pass


class VCModelService:
    def __init__(self, *, repo: Any):
        self.repo = repo
        self.model_dir = Path(str(os.getenv("VC_MODEL_DIR", "artifacts/models/vc_main")))
        self.model, self.manifest = self._load_model_or_fail()

    def _load_model_or_fail(self):
        if torch is None:
            raise RuntimeError("torch_required_for_vc_inference")
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
        return model, manifest

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -40.0, 40.0))
        return 1.0 / (1.0 + float(np.exp(-x)))

    def _build_features(self, company_name: str) -> np.ndarray:
        events = self.repo.recent_event_context(company_name, limit=24)
        funding = sum(1 for e in events if str(e.get("event_type") or "") == "funding")
        product = sum(1 for e in events if str(e.get("event_type") or "") == "product")
        regulatory = sum(1 for e in events if str(e.get("event_type") or "") == "regulatory")
        conf = float(np.mean([float(e.get("confidence_score") or 0.0) for e in events])) if events else 0.0
        density = float(len(events) / 24.0)
        return np.array([funding, product, regulatory, conf, density], dtype=np.float32)

    def predict_with_context(self, *, company_name: str, horizon_months: int) -> Dict[str, Any]:
        target = str(company_name or "").strip()
        if not target:
            raise HTTPException(status_code=400, detail="company_name_required")
        feats = self._build_features(target)
        with torch.no_grad():
            logit = self.model(torch.tensor(feats, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            p = torch.softmax(logit, dim=-1).cpu().numpy()
        p_round = float(p[1])
        h_adj = {6: 0.06, 12: 0.0, 24: -0.08}.get(int(horizon_months), 0.0)
        p_h = float(np.clip(p_round + h_adj, 0.01, 0.99))
        p_exit = float(np.clip(1.0 - p_round * 0.7, 0.01, 0.95))

        outputs = {
            "p_next_round": {
                "6m": float(np.clip(p_round + 0.05, 0.01, 0.99)),
                "12m": float(np.clip(p_round, 0.01, 0.99)),
                "24m": float(np.clip(p_round - 0.08, 0.01, 0.99)),
            },
            "p_exit_24m": p_exit,
            "expected_moic_distribution": {
                "p10": round(0.6 + p_h * 0.9, 2),
                "p50": round(1.0 + p_h * 1.8, 2),
                "p90": round(1.4 + p_h * 3.2, 2),
            },
            "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_id": str(self.manifest.get("model_id") or "vc_main"),
            "schema_hash": SCHEMA_HASH,
        }
        explanation = {
            "top_event_contributors": self.repo.recent_event_context(target, limit=5),
            "top_feature_contributors": [
                {"feature": "funding_count", "value": float(feats[0]), "contribution": float(feats[0]) * 0.3},
                {"feature": "product_count", "value": float(feats[1]), "contribution": float(feats[1]) * 0.2},
                {"feature": "regulatory_count", "value": float(feats[2]), "contribution": -float(feats[2]) * 0.2},
                {"feature": "mean_confidence", "value": float(feats[3]), "contribution": float(feats[3]) * 0.2},
            ],
            "evidence_links": [str(e.get("source_url") or "") for e in self.repo.recent_event_context(target, limit=5) if str(e.get("source_url") or "").strip()],
            "model_version": str(self.manifest.get("model_id") or "vc_main"),
            "feature_version": "schema_main",
        }
        return {
            "target": target,
            "score": float(round(p_h, 4)),
            "confidence": float(round(max(0.01, min(0.99, 0.55 + abs(p_h - 0.5))), 4)),
            "outputs": outputs,
            "explanation": explanation,
        }
