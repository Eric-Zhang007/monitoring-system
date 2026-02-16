from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    HAS_LGB = False

def _default_model_dir() -> str:
    env_path = str(os.getenv("MODEL_DIR", "")).strip()
    if env_path:
        return env_path
    local_models = Path(__file__).resolve().parents[1] / "backend" / "models"
    if local_models.exists():
        return str(local_models)
    return "/app/models"


MODEL_DIR = _default_model_dir()


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


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, n_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_tokens),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_channels),
        )
        self.ln1 = nn.LayerNorm(n_channels)
        self.ln2 = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y
        z = self.ln2(x)
        z = self.channel_mlp(z)
        return x + z


class TSMixerLiquidModel(nn.Module):
    def __init__(self, n_tokens: int, n_channels: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock(n_tokens=n_tokens, n_channels=n_channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(n_channels),
            nn.Flatten(),
            nn.Linear(n_tokens * n_channels, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)


class ModelRouter:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.torch_cache: Dict[str, Dict[str, Any]] = {}
        self.tabular_cache: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expected_feature_schema = os.getenv("FEATURE_PAYLOAD_SCHEMA_VERSION", "v2.3")
        self.compatible_feature_schemas = self._resolve_compatible_schemas(self.expected_feature_schema)
        self.expected_data_version = os.getenv("DATA_VERSION", "v1")

    @staticmethod
    def _resolve_compatible_schemas(expected: str) -> set[str]:
        cur = str(expected or "").strip() or "v2.3"
        out = {cur, "v2.2"}
        return {x for x in out if x}

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

    @staticmethod
    def _to_sequence(features: np.ndarray) -> np.ndarray:
        x = np.array(features, dtype=np.float32).reshape(-1)
        if x.shape[0] < 15:
            return x[None, None, :]
        seq = np.stack(
            [
                np.stack([x[0], x[4], x[10]], axis=0),
                np.stack([x[1], x[5], x[9]], axis=0),
                np.stack([x[2], x[6], x[11]], axis=0),
                np.stack([x[3], x[7], x[12]], axis=0),
                np.stack([x[13], x[14], x[8]], axis=0),
            ],
            axis=0,
        )
        return seq[None, :, :].astype(np.float32)

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
        model_name = str(ckpt.get("type") or "")

        if model_type == "vc":
            resolved_in_dim = int(ckpt.get("in_dim", in_dim))
            model = TinyVCModel(resolved_in_dim)
            model.load_state_dict(state)
            payload: Dict[str, Any] = {
                "model": model,
                "in_dim": resolved_in_dim,
                "normalization": ckpt.get("normalization"),
            }
        elif model_name == "tsmixer_liquid":
            ckpt_schema = str(ckpt.get("feature_payload_schema_version") or "").strip()
            ckpt_data_version = str(ckpt.get("data_version") or "").strip()
            if ckpt_schema and ckpt_schema not in self.compatible_feature_schemas:
                return None
            if ckpt_data_version and ckpt_data_version != self.expected_data_version:
                return None
            if not str(ckpt.get("train_report_hash") or "").strip():
                return None
            if not isinstance(ckpt.get("normalization"), dict):
                return None
            n_tokens = int(ckpt.get("n_tokens", 5))
            n_channels = int(ckpt.get("n_channels", 3))
            model = TSMixerLiquidModel(n_tokens=n_tokens, n_channels=n_channels, n_blocks=2)
            model.load_state_dict(state)
            payload = {
                "model": model,
                "normalization": ckpt.get("normalization"),
                "ensemble_alpha": float(ckpt.get("ensemble_alpha", 0.7) or 0.7),
                "n_tokens": n_tokens,
                "n_channels": n_channels,
                "feature_payload_schema_version": ckpt_schema,
                "data_version": ckpt_data_version,
                "train_report_hash": str(ckpt.get("train_report_hash")),
            }
        else:
            return None

        model.to(self.device)
        model.eval()
        self.torch_cache[key] = payload
        return payload

    def _load_liquid_tabular_model(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = symbol.lower()
        if key in self.tabular_cache:
            return self.tabular_cache[key]

        model = self._load_json_model(f"liquid_{symbol.lower()}_lgbm_baseline_v2.json")
        if not model:
            return None
        required = ("model_name", "model_version", "track", "type", "feature_version", "data_version")
        if any(not str(model.get(k) or "").strip() for k in required):
            return None
        if str(model.get("track") or "").strip().lower() != "liquid":
            return None
        if str(model.get("feature_version") or "").strip() != os.getenv("FEATURE_VERSION", "feature-store-v2.1"):
            return None
        if str(model.get("data_version") or "").strip() != self.expected_data_version:
            return None
        schema_version = str(model.get("feature_payload_schema_version") or "").strip()
        if schema_version and schema_version not in self.compatible_feature_schemas:
            return None
        loaded: Dict[str, Any] = {
            "name": str(model.get("model") or "unknown"),
            "x_mean": np.array(model.get("x_mean", []), dtype=np.float32),
            "x_std": np.array(model.get("x_std", []), dtype=np.float32),
            "feature_dim": int(model.get("feature_dim", 15) or 15),
        }
        if model.get("model") == "lightgbm" and HAS_LGB and isinstance(model.get("booster_model"), str):
            try:
                booster = lgb.Booster(model_str=str(model["booster_model"]))
                loaded["booster"] = booster
            except Exception:
                pass
        if isinstance(model.get("weights"), list):
            loaded["weights"] = np.array(model.get("weights", []), dtype=np.float32)

        self.tabular_cache[key] = loaded
        return loaded

    def _predict_liquid_tabular(self, symbol: str, features: np.ndarray) -> float:
        bundle = self._load_liquid_tabular_model(symbol)
        if not bundle:
            return 0.0

        feature_dim = int(bundle.get("feature_dim", features.shape[0]))
        aligned = self._align_features(features, feature_dim)
        x_mean = np.array(bundle.get("x_mean", []), dtype=np.float32).reshape(-1)
        x_std = np.array(bundle.get("x_std", []), dtype=np.float32).reshape(-1)
        if x_mean.size > 0 and x_std.size > 0:
            x_mean = self._align_features(x_mean, feature_dim)
            x_std = np.clip(self._align_features(x_std, feature_dim), 1e-6, None)
            aligned = (aligned - x_mean) / x_std

        booster = bundle.get("booster")
        if booster is not None:
            pred = booster.predict(aligned.reshape(1, -1))
            return float(np.array(pred, dtype=np.float32).reshape(-1)[0])

        weights = bundle.get("weights")
        if isinstance(weights, np.ndarray) and weights.size > 0:
            ww = self._align_features(weights, feature_dim)
            return float(aligned @ ww)

        return 0.0

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
        expected_return_nn = 0.0
        expected_return_tabular = self._predict_liquid_tabular(symbol, features)
        ensemble_alpha = 0.0

        if model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
            pt_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_tsmixer_v2.pt")
            pt_bundle = self._load_torch_model(pt_path, "liquid", features.shape[0])
            if pt_bundle is not None:
                with torch.no_grad():
                    fx = np.array(features, dtype=np.float32).reshape(-1)
                    fx = self._normalize_features(fx, pt_bundle.get("normalization"))
                    seq = self._to_sequence(fx)
                    xt = torch.tensor(seq, dtype=torch.float32, device=self.device)
                    expected_return_nn = float(pt_bundle["model"](xt).item())
                    ensemble_alpha = float(pt_bundle.get("ensemble_alpha", 0.7) or 0.7)

        if model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
            if abs(expected_return_nn) <= 1e-12 and abs(expected_return_tabular) <= 1e-12:
                expected_return = 0.0
            elif abs(expected_return_nn) <= 1e-12:
                expected_return = expected_return_tabular
            elif abs(expected_return_tabular) <= 1e-12:
                expected_return = expected_return_nn
            else:
                alpha = max(0.0, min(1.0, ensemble_alpha))
                expected_return = alpha * expected_return_nn + (1.0 - alpha) * expected_return_tabular
        else:
            expected_return = expected_return_tabular

        vol_forecast = float(max(0.01, 0.02 + abs(expected_return) * 4))
        confidence = float(max(0.35, min(0.95, 0.75 - vol_forecast * 4)))

        return {
            "expected_return": float(expected_return),
            "vol_forecast": vol_forecast,
            "signal_confidence": confidence,
            "stack": {
                "nn": float(expected_return_nn),
                "tabular": float(expected_return_tabular),
                "alpha": float(ensemble_alpha),
            },
        }
