from __future__ import annotations


class ModelRouter:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs
        raise RuntimeError("legacy_model_router_disabled_strict_only")
