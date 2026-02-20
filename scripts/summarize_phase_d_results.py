#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _backbone_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = report.get("results")
    if not isinstance(items, list):
        return rows
    for it in items:
        if not isinstance(it, dict):
            continue
        wf = it.get("walk_forward") if isinstance(it.get("walk_forward"), dict) else {}
        basic = wf.get("basic") if isinstance(wf.get("basic"), dict) else {}
        rows.append(
            {
                "backbone": str(it.get("backbone") or ""),
                "ready": bool(it.get("ready", False)),
                "status": str(wf.get("status") or ""),
                "reason": str(wf.get("reason") or ""),
                "folds": int(wf.get("folds", 0) or 0),
                "mse": float(basic.get("mse", 0.0) or 0.0),
                "mae": float(basic.get("mae", 0.0) or 0.0),
                "hit_rate": float(basic.get("hit_rate", 0.0) or 0.0),
            }
        )
    return rows


def _ablation_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = report.get("ablation_results")
    if not isinstance(items, list):
        return rows
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append(
            {
                "ablation": str(it.get("ablation") or ""),
                "status": str(it.get("status") or ""),
                "rows": int(it.get("rows", 0) or 0),
                "folds": int(it.get("folds", 0) or 0),
                "mse": float(it.get("mse", 0.0) or 0.0),
                "mae": float(it.get("mae", 0.0) or 0.0),
                "hit_rate": float(it.get("hit_rate", 0.0) or 0.0),
            }
        )
    return rows


def _render_markdown_table(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows_\n"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def build_summary(
    *,
    candidate_report: Dict[str, Any],
    backbone_report: Dict[str, Any],
    eval_report: Dict[str, Any],
) -> Dict[str, Any]:
    backbone_rows = _backbone_rows(backbone_report)
    ablation_rows = _ablation_rows(eval_report)
    ready_backbones = [r["backbone"] for r in backbone_rows if bool(r.get("ready"))]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidate": {
            "model_name": candidate_report.get("model_name"),
            "fusion_mode": candidate_report.get("fusion_mode"),
            "val_mse": candidate_report.get("val_mse"),
            "val_mae": candidate_report.get("val_mae"),
        },
        "backbone": {
            "torch_available": bool(backbone_report.get("torch_available", False)),
            "ready_backbones": ready_backbones,
            "rows": backbone_rows,
        },
        "ablation": {
            "primary_ablation": eval_report.get("primary_ablation"),
            "rows": ablation_rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase-D multimodal results")
    parser.add_argument("--candidate", default="artifacts/models/multimodal_candidate.json")
    parser.add_argument("--backbone", default="artifacts/experiments/backbone_suite_latest.json")
    parser.add_argument("--eval", default="artifacts/models/multimodal_eval.json")
    parser.add_argument("--out-json", default="artifacts/experiments/phase_d_summary_latest.json")
    parser.add_argument("--out-md", default="artifacts/experiments/phase_d_summary_latest.md")
    args = parser.parse_args()

    candidate = _load_json(Path(args.candidate))
    backbone = _load_json(Path(args.backbone))
    eval_report = _load_json(Path(args.eval))
    out = build_summary(candidate_report=candidate, backbone_report=backbone, eval_report=eval_report)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# Phase-D Summary")
    md.append("")
    md.append(f"- generated_at: `{out.get('generated_at')}`")
    md.append(f"- model_name: `{(out.get('candidate') or {}).get('model_name')}`")
    md.append(f"- fusion_mode: `{(out.get('candidate') or {}).get('fusion_mode')}`")
    md.append(f"- primary_ablation: `{(out.get('ablation') or {}).get('primary_ablation')}`")
    md.append("")
    md.append("## Backbone Comparison")
    md.append("")
    md.append(
        _render_markdown_table(
            ["backbone", "ready", "status", "reason", "folds", "mse", "mae", "hit_rate"],
            list((out.get("backbone") or {}).get("rows") or []),
        )
    )
    md.append("## Ablation Comparison")
    md.append("")
    md.append(
        _render_markdown_table(
            ["ablation", "status", "rows", "folds", "mse", "mae", "hit_rate"],
            list((out.get("ablation") or {}).get("rows") or []),
        )
    )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_json": str(out_json),
                "out_md": str(out_md),
                "ready_backbones": (out.get("backbone") or {}).get("ready_backbones", []),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
