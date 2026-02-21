from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _registry_path() -> Path:
    raw = os.getenv("LIQUID_HORIZON_REGISTRY_FILE", "artifacts/models/liquid_horizon_registry.json")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    return p


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_registry() -> Dict[str, Any]:
    p = _registry_path()
    if not p.exists():
        return {"status": "missing", "updated_at": _now_iso(), "symbols": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "invalid", "updated_at": _now_iso(), "symbols": {}}
    if not isinstance(obj, dict):
        return {"status": "invalid", "updated_at": _now_iso(), "symbols": {}}
    obj.setdefault("updated_at", _now_iso())
    obj.setdefault("symbols", {})
    obj.setdefault("status", "ok")
    return obj


def save_registry(payload: Dict[str, Any]) -> Path:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload or {})
    out["updated_at"] = _now_iso()
    out.setdefault("status", "ok")
    out.setdefault("symbols", {})
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def get_active_model(symbol: str, horizon: str) -> Optional[Tuple[str, str]]:
    reg = load_registry()
    symbols = reg.get("symbols") if isinstance(reg.get("symbols"), dict) else {}
    sym = str(symbol or "").upper().strip()
    hh = str(horizon or "").lower().strip()
    block = symbols.get(sym) if isinstance(symbols.get(sym), dict) else {}
    item = block.get(hh) if isinstance(block.get(hh), dict) else {}
    active = item.get("active") if isinstance(item.get("active"), dict) else {}
    name = str(active.get("model_name") or "").strip()
    ver = str(active.get("model_version") or "").strip()
    if name and ver:
        return name, ver
    return None


def get_candidate_model(symbol: str, horizon: str) -> Optional[Tuple[str, str]]:
    reg = load_registry()
    symbols = reg.get("symbols") if isinstance(reg.get("symbols"), dict) else {}
    sym = str(symbol or "").upper().strip()
    hh = str(horizon or "").lower().strip()
    block = symbols.get(sym) if isinstance(symbols.get(sym), dict) else {}
    item = block.get(hh) if isinstance(block.get(hh), dict) else {}
    cand = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    name = str(cand.get("model_name") or "").strip()
    ver = str(cand.get("model_version") or "").strip()
    if name and ver:
        return name, ver
    return None


def upsert_active(symbol: str, horizon: str, model_name: str, model_version: str, actor: str = "manual") -> Dict[str, Any]:
    reg = load_registry()
    symbols = reg.setdefault("symbols", {})
    sym = str(symbol or "").upper().strip()
    hh = str(horizon or "").lower().strip()
    s_block = symbols.setdefault(sym, {})
    h_block = s_block.setdefault(hh, {})
    prev = h_block.get("active") if isinstance(h_block.get("active"), dict) else {}
    h_block["previous"] = dict(prev)
    h_block["active"] = {
        "model_name": str(model_name),
        "model_version": str(model_version),
        "updated_at": _now_iso(),
        "updated_by": str(actor or "manual"),
    }
    reg["status"] = "ok"
    save_registry(reg)
    return h_block


def upsert_candidate(symbol: str, horizon: str, model_name: str, model_version: str, actor: str = "manual") -> Dict[str, Any]:
    reg = load_registry()
    symbols = reg.setdefault("symbols", {})
    sym = str(symbol or "").upper().strip()
    hh = str(horizon or "").lower().strip()
    s_block = symbols.setdefault(sym, {})
    h_block = s_block.setdefault(hh, {})
    h_block["candidate"] = {
        "model_name": str(model_name),
        "model_version": str(model_version),
        "updated_at": _now_iso(),
        "updated_by": str(actor or "manual"),
    }
    reg["status"] = "ok"
    save_registry(reg)
    return h_block


def promote_candidate(symbol: str, horizon: str, actor: str = "manual") -> Dict[str, Any]:
    reg = load_registry()
    symbols = reg.setdefault("symbols", {})
    sym = str(symbol or "").upper().strip()
    hh = str(horizon or "").lower().strip()
    s_block = symbols.setdefault(sym, {})
    h_block = s_block.setdefault(hh, {})
    cand = h_block.get("candidate") if isinstance(h_block.get("candidate"), dict) else {}
    if not cand:
        return {"promoted": False, "reason": "no_candidate_model", "entry": h_block}
    active = h_block.get("active") if isinstance(h_block.get("active"), dict) else {}
    h_block["previous"] = dict(active)
    h_block["active"] = {
        "model_name": str(cand.get("model_name") or ""),
        "model_version": str(cand.get("model_version") or ""),
        "updated_at": _now_iso(),
        "updated_by": str(actor or "manual"),
        "source": "candidate_promote",
    }
    reg["status"] = "ok"
    save_registry(reg)
    return {"promoted": True, "entry": h_block}


def rollback_active(symbol: str, horizon: str, actor: str = "manual") -> Dict[str, Any]:
    reg = load_registry()
    symbols = reg.setdefault("symbols", {})
    sym = str(symbol or "").upper().strip()
    hh = str(horizon or "").lower().strip()
    s_block = symbols.setdefault(sym, {})
    h_block = s_block.setdefault(hh, {})
    prev = h_block.get("previous") if isinstance(h_block.get("previous"), dict) else {}
    active = h_block.get("active") if isinstance(h_block.get("active"), dict) else {}
    if not prev:
        return {"rolled_back": False, "reason": "no_previous_model", "active": active, "previous": prev}
    h_block["active"] = {
        "model_name": str(prev.get("model_name") or ""),
        "model_version": str(prev.get("model_version") or ""),
        "updated_at": _now_iso(),
        "updated_by": str(actor or "manual"),
    }
    h_block["previous"] = dict(active) if active else {}
    reg["status"] = "ok"
    save_registry(reg)
    return {"rolled_back": True, "active": h_block.get("active"), "previous": h_block.get("previous")}
