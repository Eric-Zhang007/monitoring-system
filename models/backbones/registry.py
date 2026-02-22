from __future__ import annotations

from typing import Any, Dict

from models.backbones.base import BackboneBase
from models.backbones.itransformer import ITransformerBackbone
from models.backbones.patchtst import PatchTSTBackbone
from models.backbones.tft import TFTBackbone


BACKBONE_WHITELIST = ("patchtst", "itransformer", "tft")


def build_backbone(name: str, **cfg: Dict[str, Any]) -> BackboneBase:
    key = str(name or "").strip().lower()
    params = dict(cfg)
    if key == "patchtst":
        return PatchTSTBackbone(**params)
    if key == "itransformer":
        params.pop("patch_len", None)
        return ITransformerBackbone(**params)
    if key == "tft":
        params.pop("patch_len", None)
        params.pop("n_layers", None)
        return TFTBackbone(**params)
    raise ValueError(f"unsupported_backbone:{key}:allowed={list(BACKBONE_WHITELIST)}")
