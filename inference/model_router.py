from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    HAS_TORCH = False

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
    return "/opt/monitoring-system/models"


MODEL_DIR = _default_model_dir()


if HAS_TORCH:
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
else:
    class TinyVCModel:  # type: ignore[no-redef]
        pass


    class MixerBlock:  # type: ignore[no-redef]
        pass


    class TSMixerLiquidModel:  # type: ignore[no-redef]
        pass


class ModelRouter:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.torch_cache: Dict[str, Dict[str, Any]] = {}
        self.tabular_cache: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
        self.expected_feature_schema = os.getenv("FEATURE_PAYLOAD_SCHEMA_VERSION", "main")
        self.compatible_feature_schemas = self._resolve_compatible_schemas(self.expected_feature_schema)
        self.expected_data_version = os.getenv("DATA_VERSION", "v1")

    @staticmethod
    def _resolve_compatible_schemas(expected: str) -> set[str]:
        cur = str(expected or "").strip() or "main"
        out = {cur, "main", "v2.3", "v2.2", "legacy"}
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
    def _to_sequence(features: np.ndarray, n_tokens: int | None = None, n_channels: int | None = None) -> np.ndarray:
        x = np.array(features, dtype=np.float32).reshape(-1)
        if n_tokens is not None and n_channels is not None:
            tok = max(1, int(n_tokens))
            ch = max(1, int(n_channels))
        else:
            ch = max(1, int(os.getenv("LIQUID_SEQUENCE_CHANNELS", "5")))
            tok = int(max(1, int(np.ceil(float(x.shape[0]) / float(ch)))))
        target_dim = tok * ch
        if x.shape[0] < target_dim:
            pad = np.zeros((target_dim - x.shape[0],), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        elif x.shape[0] > target_dim:
            x = x[:target_dim]
        return x.reshape(1, tok, ch).astype(np.float32)

    @staticmethod
    def _sigmoid_scalar(x: float) -> float:
        z = float(np.clip(float(x), -40.0, 40.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    @staticmethod
    def _sanitize_indices(indices: object, feature_dim: int) -> np.ndarray:
        if isinstance(indices, np.ndarray):
            raw = indices.reshape(-1).tolist()
        elif isinstance(indices, list):
            raw = indices
        else:
            return np.zeros((0,), dtype=np.int64)
        out: List[int] = []
        for idx in raw:
            try:
                iv = int(idx)
            except Exception:
                continue
            if 0 <= iv < int(feature_dim):
                out.append(iv)
        if not out:
            return np.zeros((0,), dtype=np.int64)
        return np.array(sorted(set(out)), dtype=np.int64)

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
        if not HAS_TORCH:
            return None
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
        fusion_mode = str(model.get("fusion_mode") or "single_ridge").strip().lower()
        if fusion_mode not in {"single_ridge", "residual_gate"}:
            fusion_mode = "single_ridge"
        loaded["fusion_mode"] = fusion_mode
        if model.get("model") == "lightgbm" and HAS_LGB and isinstance(model.get("booster_model"), str):
            try:
                booster = lgb.Booster(model_str=str(model["booster_model"]))
                loaded["booster"] = booster
            except Exception:
                pass
        if isinstance(model.get("weights"), list):
            loaded["weights"] = np.array(model.get("weights", []), dtype=np.float32)
        loaded["default_horizon"] = str(model.get("default_horizon") or "1h").strip().lower() or "1h"
        if isinstance(model.get("horizons"), list):
            loaded["horizons"] = [str(x).strip().lower() for x in model.get("horizons", []) if str(x).strip()]
        raw_heads = model.get("horizon_heads")
        if isinstance(raw_heads, dict):
            parsed_heads: Dict[str, Dict[str, Any]] = {}
            for h, item in raw_heads.items():
                if not isinstance(item, dict):
                    continue
                h_key = str(h).strip().lower()
                if not h_key:
                    continue
                head: Dict[str, Any] = {}
                if isinstance(item.get("weights"), list):
                    head["weights"] = np.array(item.get("weights", []), dtype=np.float32)
                if isinstance(item.get("vol_weights"), list):
                    head["vol_weights"] = np.array(item.get("vol_weights", []), dtype=np.float32)
                head["residual_std"] = float(item.get("residual_std", 0.0) or 0.0)
                calib = item.get("calibration") if isinstance(item.get("calibration"), dict) else {}
                head["calibration"] = {
                    "method": str(calib.get("method") or "sigmoid_abs_z"),
                    "a": float(calib.get("a", 0.85) or 0.85),
                    "b": float(calib.get("b", -0.1) or -0.1),
                    "bins": list(calib.get("bins") or []),
                }
                parsed_heads[h_key] = head
            if parsed_heads:
                loaded["horizon_heads"] = parsed_heads
        raw_models = model.get("horizon_models")
        if isinstance(raw_models, dict):
            parsed_models: Dict[str, Dict[str, Any]] = {}
            for h, item in raw_models.items():
                if not isinstance(item, dict):
                    continue
                h_key = str(h).strip().lower()
                if not h_key:
                    continue
                sub: Dict[str, Any] = {}
                if isinstance(item.get("weights"), list):
                    sub["weights"] = np.array(item.get("weights", []), dtype=np.float32)
                if isinstance(item.get("vol_weights"), list):
                    sub["vol_weights"] = np.array(item.get("vol_weights", []), dtype=np.float32)
                sub["residual_std"] = float(item.get("residual_std", 0.0) or 0.0)
                sub["model_type"] = str(item.get("model_type") or "ridge_submodel")
                calib = item.get("calibration") if isinstance(item.get("calibration"), dict) else {}
                sub["calibration"] = {
                    "method": str(calib.get("method") or "sigmoid_abs_z"),
                    "a": float(calib.get("a", 0.85) or 0.85),
                    "b": float(calib.get("b", -0.1) or -0.1),
                    "bins": list(calib.get("bins") or []),
                }
                parsed_models[h_key] = sub
            if parsed_models:
                loaded["horizon_models"] = parsed_models
        if isinstance(model.get("meta_aggregator"), dict):
            loaded["meta_aggregator"] = dict(model.get("meta_aggregator") or {})
        if isinstance(model.get("cost_config"), dict):
            loaded["cost_config"] = dict(model.get("cost_config") or {})
        if fusion_mode == "residual_gate":
            if isinstance(model.get("base_feature_indices"), list):
                loaded["base_feature_indices"] = np.array(model.get("base_feature_indices", []), dtype=np.int64)
            if isinstance(model.get("text_feature_indices"), list):
                loaded["text_feature_indices"] = np.array(model.get("text_feature_indices", []), dtype=np.int64)
            if isinstance(model.get("base_weights"), list):
                loaded["base_weights"] = np.array(model.get("base_weights", []), dtype=np.float32)
            if isinstance(model.get("text_weights"), list):
                loaded["text_weights"] = np.array(model.get("text_weights", []), dtype=np.float32)
            loaded["gate_mu"] = float(model.get("gate_mu", 0.0) or 0.0)
            loaded["gate_sigma"] = float(model.get("gate_sigma", 0.0) or 0.0)
            loaded["gate_multiplier"] = float(model.get("gate_multiplier", 1.0) or 1.0)

        self.tabular_cache[key] = loaded
        return loaded

    def _predict_liquid_tabular(self, symbol: str, features: np.ndarray) -> float:
        pred, _ = self._predict_liquid_tabular_with_meta(symbol, features)
        return float(pred)

    def _predict_liquid_tabular_with_meta(self, symbol: str, features: np.ndarray) -> Tuple[float, Dict[str, float | str]]:
        bundle = self._load_liquid_tabular_model(symbol)
        if not bundle:
            return 0.0, {"mode": "missing"}

        feature_dim = int(bundle.get("feature_dim", features.shape[0]))
        aligned = self._align_features(features, feature_dim)
        x_mean = np.array(bundle.get("x_mean", []), dtype=np.float32).reshape(-1)
        x_std = np.array(bundle.get("x_std", []), dtype=np.float32).reshape(-1)
        if x_mean.size > 0 and x_std.size > 0:
            x_mean = self._align_features(x_mean, feature_dim)
            x_std = np.clip(self._align_features(x_std, feature_dim), 1e-6, None)
            aligned = (aligned - x_mean) / x_std

        horizon_heads = bundle.get("horizon_heads")
        horizon_models = bundle.get("horizon_models")
        horizon_source = None
        if isinstance(horizon_heads, dict) and horizon_heads:
            horizon_source = horizon_heads
        elif isinstance(horizon_models, dict) and horizon_models:
            horizon_source = horizon_models
        if isinstance(horizon_source, dict) and horizon_source:
            pred_map: Dict[str, float] = {}
            vol_map: Dict[str, float] = {}
            conf_map: Dict[str, float] = {}
            for horizon, head in horizon_source.items():
                if not isinstance(head, dict):
                    continue
                ww = head.get("weights")
                if not isinstance(ww, np.ndarray) or ww.size <= 0:
                    continue
                w = self._align_features(ww, feature_dim)
                pred_h = float(aligned @ w)
                wv_raw = head.get("vol_weights")
                if isinstance(wv_raw, np.ndarray) and wv_raw.size > 0:
                    wv = self._align_features(wv_raw, feature_dim)
                    vol_h = float(max(1e-6, abs(float(aligned @ wv))))
                else:
                    vol_h = float(max(1e-6, abs(pred_h) * 0.5 + float(head.get("residual_std", 0.0) or 0.0)))
                calib = head.get("calibration") if isinstance(head.get("calibration"), dict) else {}
                a = float(calib.get("a", 0.85) or 0.85)
                b = float(calib.get("b", -0.1) or -0.1)
                z = pred_h / max(vol_h, 1e-6)
                conf_h = self._sigmoid_scalar(a * abs(z) + b)
                pred_map[str(horizon).lower()] = float(pred_h)
                vol_map[str(horizon).lower()] = float(vol_h)
                conf_map[str(horizon).lower()] = float(max(0.01, min(0.99, conf_h)))
            if pred_map:
                default_h = str(bundle.get("default_horizon") or "1h").strip().lower() or "1h"
                if default_h not in pred_map:
                    default_h = "1h" if "1h" in pred_map else sorted(pred_map.keys())[0]
                mode = "multi_horizon_heads" if horizon_source is horizon_heads else "multi_model_meta"
                meta_agg = dict(bundle.get("meta_aggregator") or {})
                return float(pred_map.get(default_h, 0.0)), {
                    "mode": mode,
                    "default_horizon": default_h,
                    "expected_return_horizons": pred_map,
                    "vol_forecast_horizons": vol_map,
                    "signal_confidence_horizons": conf_map,
                    "meta_aggregator": meta_agg,
                    "cost_config": dict(bundle.get("cost_config") or {}),
                }

        fusion_mode = str(bundle.get("fusion_mode") or "single_ridge").strip().lower()
        if fusion_mode == "residual_gate":
            base_indices = self._sanitize_indices(bundle.get("base_feature_indices"), feature_dim)
            text_indices = self._sanitize_indices(bundle.get("text_feature_indices"), feature_dim)
            if base_indices.size == 0:
                base_indices = np.arange(feature_dim, dtype=np.int64)
            base_slice = aligned[base_indices]
            base_weights = bundle.get("base_weights")
            if isinstance(base_weights, np.ndarray) and base_weights.size > 0:
                ww_base = self._align_features(base_weights, base_slice.shape[0])
                base_pred = float(base_slice @ ww_base)
            else:
                base_pred = 0.0

            text_slice = aligned[text_indices] if text_indices.size > 0 else np.zeros((0,), dtype=np.float32)
            text_weights = bundle.get("text_weights")
            if isinstance(text_weights, np.ndarray) and text_weights.size > 0 and text_slice.size > 0:
                ww_text = self._align_features(text_weights, text_slice.shape[0])
                delta_pred = float(text_slice @ ww_text)
                text_activity = float(np.mean(np.abs(text_slice)))
                gate_mu = float(bundle.get("gate_mu", 0.0) or 0.0)
                gate_sigma = max(float(bundle.get("gate_sigma", 0.0) or 0.0), 1e-6)
                gate = self._sigmoid_scalar((text_activity - gate_mu) / gate_sigma)
            else:
                delta_pred = 0.0
                text_activity = 0.0
                gate = 0.0
            gate_multiplier = float(np.clip(float(bundle.get("gate_multiplier", 1.0) or 1.0), 0.0, 2.0))
            pred = float(base_pred + gate_multiplier * gate * delta_pred)
            return pred, {
                "mode": "residual_gate",
                "base": float(base_pred),
                "delta": float(delta_pred),
                "gate": float(gate),
                "gate_multiplier": float(gate_multiplier),
                "text_activity": float(text_activity),
            }

        booster = bundle.get("booster")
        if booster is not None:
            pred = booster.predict(aligned.reshape(1, -1))
            return float(np.array(pred, dtype=np.float32).reshape(-1)[0]), {"mode": "lightgbm"}

        weights = bundle.get("weights")
        if isinstance(weights, np.ndarray) and weights.size > 0:
            ww = self._align_features(weights, feature_dim)
            return float(aligned @ ww), {"mode": "single_ridge"}

        return 0.0, {"mode": "single_ridge"}

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
        expected_return_tabular, tab_meta = self._predict_liquid_tabular_with_meta(symbol, features)
        ensemble_alpha = 0.0

        if model_name in {"liquid_ttm_ensemble", "liquid_ttm"}:
            pt_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_tsmixer_v2.pt")
            pt_bundle = self._load_torch_model(pt_path, "liquid", features.shape[0])
            if pt_bundle is not None:
                with torch.no_grad():
                    fx = np.array(features, dtype=np.float32).reshape(-1)
                    fx = self._normalize_features(fx, pt_bundle.get("normalization"))
                    seq = self._to_sequence(
                        fx,
                        n_tokens=int(pt_bundle.get("n_tokens", 1) or 1),
                        n_channels=int(pt_bundle.get("n_channels", 1) or 1),
                    )
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
        expected_return_h = tab_meta.get("expected_return_horizons") if isinstance(tab_meta.get("expected_return_horizons"), dict) else {}
        vol_forecast_h = tab_meta.get("vol_forecast_horizons") if isinstance(tab_meta.get("vol_forecast_horizons"), dict) else {}
        signal_conf_h = tab_meta.get("signal_confidence_horizons") if isinstance(tab_meta.get("signal_confidence_horizons"), dict) else {}
        if not expected_return_h:
            expected_return_h = {"1h": float(expected_return)}
        if not vol_forecast_h:
            vol_forecast_h = {"1h": float(vol_forecast)}
        if not signal_conf_h:
            signal_conf_h = {"1h": float(confidence)}

        return {
            "expected_return": float(expected_return),
            "vol_forecast": vol_forecast,
            "signal_confidence": confidence,
            "expected_return_horizons": {str(k): float(v) for k, v in expected_return_h.items()},
            "vol_forecast_horizons": {str(k): float(v) for k, v in vol_forecast_h.items()},
            "signal_confidence_horizons": {str(k): float(v) for k, v in signal_conf_h.items()},
            "stack": {
                "nn": float(expected_return_nn),
                "tabular": float(expected_return_tabular),
                "alpha": float(ensemble_alpha),
                "tabular_mode": str(tab_meta.get("mode") or "single_ridge"),
                "tabular_base": float(tab_meta.get("base") or 0.0),
                "tabular_delta": float(tab_meta.get("delta") or 0.0),
                "tabular_gate": float(tab_meta.get("gate") or 0.0),
                "tabular_gate_multiplier": float(tab_meta.get("gate_multiplier") or 0.0),
                "default_horizon": str(tab_meta.get("default_horizon") or "1h"),
                "cost_config": dict(tab_meta.get("cost_config") or {}),
            },
        }
