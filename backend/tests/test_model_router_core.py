from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "inference"))

from model_router import ModelRouter  # noqa: E402


def test_model_router_disabled_in_strict_mode():
    with pytest.raises(RuntimeError, match="legacy_model_router_disabled_strict_only"):
        ModelRouter()
