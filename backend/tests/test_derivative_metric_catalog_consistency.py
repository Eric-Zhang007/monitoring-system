from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "training"))
sys.path.append(str(ROOT / "inference"))

from feature_pipeline import DERIVATIVE_METRIC_NAMES as TRAIN_DERIVATIVE_METRICS  # noqa: E402
from liquid_feature_contract import DERIVATIVE_METRIC_NAMES as CONTRACT_DERIVATIVE_METRICS  # noqa: E402


def _load_script_module(name: str, rel_path: str):
    script_path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derivative_metric_catalog_is_shared_across_training_and_scripts():
    audit_mod = _load_script_module("audit_required_data_readiness", "scripts/audit_required_data_readiness.py")
    ingest_mod = _load_script_module("ingest_binance_derivatives_signals", "scripts/ingest_binance_derivatives_signals.py")
    assert list(TRAIN_DERIVATIVE_METRICS) == list(CONTRACT_DERIVATIVE_METRICS)
    assert list(audit_mod.CONTRACT_DERIVATIVE_METRICS) == list(CONTRACT_DERIVATIVE_METRICS)
    assert list(ingest_mod.CONTRACT_DERIVATIVE_METRICS) == list(CONTRACT_DERIVATIVE_METRICS)
