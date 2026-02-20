#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


def _to_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _to_bool(raw: object) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def build_ablation_summary(eval_report: dict) -> dict:
    if not isinstance(eval_report, dict):
        return {"available": False, "count": 0}
    rows = eval_report.get("ablation_results")
    if not isinstance(rows, list):
        return {"available": False, "count": 0}
    parsed = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").lower() != "ok":
            continue
        name = str(item.get("ablation") or "").strip().lower()
        if not name:
            continue
        parsed.append(
            {
                "ablation": name,
                "mse": _to_float(item.get("mse"), 0.0),
                "mae": _to_float(item.get("mae"), 0.0),
                "hit_rate": _to_float(item.get("hit_rate"), 0.0),
                "rows": int(item.get("rows", 0) or 0),
            }
        )
    if not parsed:
        return {"available": False, "count": 0}
    by_name = {str(x["ablation"]): x for x in parsed}
    full = by_name.get("full")
    no_text = by_name.get("no_text")
    no_macro = by_name.get("no_macro")
    event_window = by_name.get("event_window")
    out = {
        "available": True,
        "count": int(len(parsed)),
        "primary": str(eval_report.get("primary_ablation") or "full"),
        "rows": {x["ablation"]: int(x["rows"]) for x in parsed},
    }
    if full and no_text:
        out["delta_mse_no_text_vs_full"] = float(no_text["mse"] - full["mse"])
        out["delta_hit_no_text_vs_full"] = float(no_text["hit_rate"] - full["hit_rate"])
    if full and no_macro:
        out["delta_mse_no_macro_vs_full"] = float(no_macro["mse"] - full["mse"])
        out["delta_hit_no_macro_vs_full"] = float(no_macro["hit_rate"] - full["hit_rate"])
    if full and event_window:
        out["delta_mse_event_window_vs_full"] = float(event_window["mse"] - full["mse"])
        out["delta_hit_event_window_vs_full"] = float(event_window["hit_rate"] - full["hit_rate"])
    return out


def evaluate_candidate_gate(
    row: dict,
    *,
    min_oos_hit_rate: float,
    max_oos_mse: float,
    min_backbone_ready: int,
    required_backbones: List[str],
    max_delta_mse_no_text_vs_full: float,
    max_delta_mse_no_macro_vs_full: float,
) -> Tuple[bool, List[str], dict]:
    reasons: List[str] = []
    metrics = {
        "oos_hit_rate": _to_float(row.get("oos_hit_rate"), 0.0),
        "oos_mse": _to_float(row.get("oos_mse"), 0.0),
        "ready_backbone_count": int(len(list(row.get("backbone_ready_list") or []))),
        "required_backbones": list(required_backbones),
    }
    if metrics["oos_hit_rate"] < float(min_oos_hit_rate):
        reasons.append("oos_hit_rate_below_threshold")
    if metrics["oos_mse"] > float(max_oos_mse):
        reasons.append("oos_mse_above_threshold")
    if metrics["ready_backbone_count"] < int(min_backbone_ready):
        reasons.append("ready_backbone_count_below_threshold")

    ready_set = {str(x).strip().lower() for x in list(row.get("backbone_ready_list") or []) if str(x).strip()}
    missing = [x for x in required_backbones if x not in ready_set]
    if missing:
        metrics["missing_required_backbones"] = missing
        reasons.append("required_backbones_not_ready")

    ab = row.get("ablation_summary") if isinstance(row.get("ablation_summary"), dict) else {}
    if bool(ab.get("available")):
        d_text = ab.get("delta_mse_no_text_vs_full")
        d_macro = ab.get("delta_mse_no_macro_vs_full")
        if d_text is not None:
            metrics["delta_mse_no_text_vs_full"] = _to_float(d_text, 0.0)
            if metrics["delta_mse_no_text_vs_full"] > float(max_delta_mse_no_text_vs_full):
                reasons.append("delta_mse_no_text_vs_full_above_threshold")
        if d_macro is not None:
            metrics["delta_mse_no_macro_vs_full"] = _to_float(d_macro, 0.0)
            if metrics["delta_mse_no_macro_vs_full"] > float(max_delta_mse_no_macro_vs_full):
                reasons.append("delta_mse_no_macro_vs_full_above_threshold")

    passed = len(reasons) == 0
    return passed, reasons, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Register multimodal candidate model")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--candidate", default="artifacts/models/multimodal_candidate.json")
    parser.add_argument("--eval", default="artifacts/models/multimodal_eval.json")
    parser.add_argument("--backbone-report", default="artifacts/experiments/backbone_suite_latest.json")
    parser.add_argument("--registry", default="artifacts/models/candidate_registry.jsonl")
    parser.add_argument("--enforce-gates", default=os.getenv("CANDIDATE_ENFORCE_GATES", "0"))
    parser.add_argument("--min-oos-hit-rate", type=float, default=float(os.getenv("CANDIDATE_MIN_OOS_HIT_RATE", "0.5")))
    parser.add_argument("--max-oos-mse", type=float, default=float(os.getenv("CANDIDATE_MAX_OOS_MSE", "1.0")))
    parser.add_argument("--min-backbone-ready", type=int, default=int(os.getenv("CANDIDATE_MIN_BACKBONE_READY", "0")))
    parser.add_argument("--required-backbones", default=os.getenv("CANDIDATE_REQUIRED_BACKBONES", ""))
    parser.add_argument(
        "--max-delta-mse-no-text-vs-full",
        type=float,
        default=float(os.getenv("CANDIDATE_MAX_DELTA_MSE_NO_TEXT_VS_FULL", "1.0")),
    )
    parser.add_argument(
        "--max-delta-mse-no-macro-vs-full",
        type=float,
        default=float(os.getenv("CANDIDATE_MAX_DELTA_MSE_NO_MACRO_VS_FULL", "1.0")),
    )
    args = parser.parse_args()

    c_path = Path(args.candidate)
    e_path = Path(args.eval)
    if not c_path.exists():
        raise SystemExit("candidate_model_missing")
    if not e_path.exists():
        raise SystemExit("eval_report_missing")

    candidate = json.loads(c_path.read_text(encoding="utf-8"))
    eval_report = json.loads(e_path.read_text(encoding="utf-8"))
    backbone_report = {}
    bb_path = Path(args.backbone_report)
    if bb_path.exists():
        try:
            backbone_report = json.loads(bb_path.read_text(encoding="utf-8"))
        except Exception:
            backbone_report = {}

    ready_backbones = []
    if isinstance(backbone_report, dict):
        rb = backbone_report.get("ready_backbones")
        if isinstance(rb, list):
            ready_backbones = [str(x) for x in rb if str(x).strip()]
    torch_available = bool(backbone_report.get("torch_available", False)) if isinstance(backbone_report, dict) else False
    ablation_summary = build_ablation_summary(eval_report)

    reg_row = {
        "registered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "candidate",
        "model_name": candidate.get("model_name", "multimodal_ridge"),
        "model_version": candidate.get("model_version", "main"),
        "feature_payload_schema_version": candidate.get("feature_payload_schema_version", "main"),
        "feature_dim": int(candidate.get("feature_dim", 0) or 0),
        "val_mse": float(candidate.get("val_mse", 0.0) or 0.0),
        "val_mae": float(candidate.get("val_mae", 0.0) or 0.0),
        "oos_mse": float(eval_report.get("mse", 0.0) or 0.0),
        "oos_mae": float(eval_report.get("mae", 0.0) or 0.0),
        "oos_hit_rate": float(eval_report.get("hit_rate", 0.0) or 0.0),
        "candidate_path": str(c_path),
        "eval_path": str(e_path),
        "fusion_mode": str(candidate.get("fusion_mode") or "single_ridge"),
        "text_dropout_prob": float(candidate.get("text_dropout_prob", 0.0) or 0.0),
        "backbone_report_path": str(bb_path) if bb_path.exists() else "",
        "backbone_ready_list": ready_backbones,
        "backbone_torch_available": bool(torch_available),
        "ablation_summary": ablation_summary,
    }
    required_backbones = [x.strip().lower() for x in str(args.required_backbones).split(",") if x.strip()]
    gate_passed, gate_reasons, gate_metrics = evaluate_candidate_gate(
        reg_row,
        min_oos_hit_rate=float(args.min_oos_hit_rate),
        max_oos_mse=float(args.max_oos_mse),
        min_backbone_ready=int(args.min_backbone_ready),
        required_backbones=required_backbones,
        max_delta_mse_no_text_vs_full=float(args.max_delta_mse_no_text_vs_full),
        max_delta_mse_no_macro_vs_full=float(args.max_delta_mse_no_macro_vs_full),
    )
    enforce_gates = _to_bool(args.enforce_gates)
    reg_row["gate"] = {
        "enforced": bool(enforce_gates),
        "passed": bool(gate_passed),
        "reasons": gate_reasons,
        "metrics": gate_metrics,
        "config": {
            "min_oos_hit_rate": float(args.min_oos_hit_rate),
            "max_oos_mse": float(args.max_oos_mse),
            "min_backbone_ready": int(args.min_backbone_ready),
            "required_backbones": required_backbones,
            "max_delta_mse_no_text_vs_full": float(args.max_delta_mse_no_text_vs_full),
            "max_delta_mse_no_macro_vs_full": float(args.max_delta_mse_no_macro_vs_full),
        },
    }
    if enforce_gates and (not gate_passed):
        reg_row["status"] = "rejected_gate"

    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(reg_row, ensure_ascii=False) + "\n")

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_models (
                    id BIGSERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                INSERT INTO candidate_models(model_name, model_version, status, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (
                    str(reg_row.get("model_name")),
                    str(reg_row.get("model_version")),
                    str(reg_row.get("status") or "candidate"),
                    json.dumps(reg_row),
                ),
            )
        conn.commit()
    out = {
        "status": "ok" if str(reg_row.get("status")) == "candidate" else "rejected_gate",
        "registry": str(registry_path),
        "model": reg_row.get("model_name"),
        "gate_passed": bool(gate_passed),
        "gate_reasons": gate_reasons,
    }
    print(json.dumps(out, ensure_ascii=False))
    if enforce_gates and (not gate_passed):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
