from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch
from torch import nn

MODEL_DIR = "/app/models"


class TinyVCModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TinyLiquidModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ModelRouter:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.torch_cache: Dict[str, Dict[str, object]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _align_features(features: np.ndarray, in_dim: int) -> np.ndarray:
        x = np.array(features, dtype=np.float32).reshape(-1)
        if x.shape[0] == in_dim:
            return x
        if x.shape[0] > in_dim:
            return x[:in_dim]
        pad = np.zeros((in_dim - x.shape[0],), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)

    @staticmethod
    def _normalize_features(features: np.ndarray, normalization: object) -> np.ndarray:
        if not isinstance(normalization, dict):
            return features
        x_mean = np.array(normalization.get("x_mean", []), dtype=np.float32).reshape(-1)
        x_std = np.array(normalization.get("x_std", []), dtype=np.float32).reshape(-1)
        if x_mean.size == 0 or x_std.size == 0:
            return features
        x_mean = ModelRouter._align_features(x_mean, features.shape[0])
        x_std = np.clip(ModelRouter._align_features(x_std, features.shape[0]), 1e-6, None)
        return (features - x_mean) / x_std

    def _load_json_model(self, name: str) -> Dict:
        if name in self.cache:
            return self.cache[name]
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.cache[name] = model
        return model

    def _load_torch_model(self, path: str, model_type: str, in_dim: int):
        key = f"{path}:{model_type}"
        if key in self.torch_cache:
            return self.torch_cache[key]
        if not os.path.exists(path):
            return None
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("state_dict", {})
        in_dim = int(ckpt.get("in_dim", in_dim))
        if model_type == "vc":
            model = TinyVCModel(in_dim)
        else:
            model = TinyLiquidModel(in_dim)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        payload: Dict[str, object] = {
            "model": model,
            "in_dim": in_dim,
            "normalization": ckpt.get("normalization"),
        }
        self.torch_cache[key] = payload
        return payload

    def predict_vc(self, features: np.ndarray, model_name: str = "vc_survival_baseline") -> Dict:
        if model_name in {"vc_survival_ttm", "vc_survival_model"}:
            pt_bundle = self._load_torch_model(os.path.join(MODEL_DIR, "vc_survival_ttm_v2.pt"), "vc", features.shape[0])
            if pt_bundle is not None:
                with torch.no_grad():
                    in_dim = int(pt_bundle.get("in_dim", features.shape[0]))
                    aligned = self._align_features(features, in_dim)
                    fx = torch.tensor(aligned, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logit = float(pt_bundle["model"](fx).item())
                    prob = float(1.0 / (1.0 + np.exp(-logit)))
            else:
                prob = 0.5
        else:
            prob = 0.5
        model = self._load_json_model("vc_survival_baseline_v2.json")
        if model:
            w = np.array(model.get("weights", []), dtype=np.float32)
            aligned = self._align_features(features, int(w.shape[0]))
            base_prob = float(1.0 / (1.0 + np.exp(-(aligned @ w))))
            if model_name in {"vc_survival_ttm", "vc_survival_model"}:
                prob = 0.7 * prob + 0.3 * base_prob
            else:
                prob = base_prob
        return {
            "p_next_round_6m": max(0.01, min(0.99, prob + 0.06)),
            "p_next_round_12m": max(0.01, min(0.99, prob)),
            "p_exit_24m": max(0.01, min(0.95, prob * 0.5 + 0.1)),
            "expected_moic_distribution": {
                "p10": round(0.7 + prob * 0.7, 2),
                "p50": round(1.0 + prob * 1.8, 2),
                "p90": round(1.3 + prob * 3.9, 2),
            },
        }

    def predict_liquid(self, symbol: str, features: np.ndarray, model_name: str = "liquid_baseline") -> Dict:
        expected_return = 0.0
        if model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
            pt_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_ttm_v2.pt")
            pt_bundle = self._load_torch_model(pt_path, "liquid", features.shape[0])
            if pt_bundle is not None:
                with torch.no_grad():
                    in_dim = int(pt_bundle.get("in_dim", features.shape[0]))
                    aligned = self._align_features(features, in_dim)
                    aligned = self._normalize_features(aligned, pt_bundle.get("normalization"))
                    fx = torch.tensor(aligned, dtype=torch.float32, device=self.device).unsqueeze(0)
                    expected_return = float(pt_bundle["model"](fx).item())
        model = self._load_json_model(f"liquid_{symbol.lower()}_baseline_v2.json")
        if model:
            w = np.array(model.get("weights", []), dtype=np.float32)
            aligned = self._align_features(features, int(w.shape[0]))
            x_mean = np.array(model.get("x_mean", []), dtype=np.float32)
            x_std = np.array(model.get("x_std", []), dtype=np.float32)
            if x_mean.size == w.shape[0] and x_std.size == w.shape[0]:
                aligned = (aligned - x_mean) / np.clip(x_std, 1e-6, None)
            baseline_return = float(aligned @ w)
            if model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
                expected_return = 0.7 * expected_return + 0.3 * baseline_return
            elif expected_return == 0.0:
                expected_return = baseline_return

        vol_forecast = float(max(0.01, 0.02 + abs(expected_return) * 4))
        confidence = float(max(0.35, min(0.95, 0.75 - vol_forecast * 4)))

        return {
            "expected_return": expected_return,
            "vol_forecast": vol_forecast,
            "signal_confidence": confidence,
        }
