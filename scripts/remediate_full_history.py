#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

FIXED_ORDER = [
    "market",
    "aux_modalities",
    "social_posts",
    "social_comments",
]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_capture(cmd: Sequence[str]) -> Dict[str, Any]:
    p = subprocess.run(list(cmd), capture_output=True, text=True)
    out: Dict[str, Any] = {
        "cmd": list(cmd),
        "returncode": int(p.returncode),
        "stdout_tail": (p.stdout or "")[-4000:],
        "stderr_tail": (p.stderr or "")[-2000:],
    }
    lines = [x for x in (p.stdout or "").splitlines() if x.strip()]
    if lines:
        try:
            out["stdout_json"] = json.loads(lines[-1])
        except Exception:
            pass
    return out


def _run_json(cmd: Sequence[str]) -> Dict[str, Any]:
    p = subprocess.run(list(cmd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command_failed rc={p.returncode} cmd={' '.join(cmd)} stderr={(p.stderr or '').strip()}")
    lines = [x for x in (p.stdout or "").splitlines() if x.strip()]
    if not lines:
        raise RuntimeError("empty_json_output")
    return json.loads(lines[-1])


def _collect_tasks(audit: Dict[str, Any]) -> Dict[str, Any]:
    per_symbol = audit.get("per_symbol") if isinstance(audit.get("per_symbol"), dict) else {}
    modality_cov = audit.get("modality_coverage") if isinstance(audit.get("modality_coverage"), dict) else {}
    ratio = audit.get("comment_post_ratio") if isinstance(audit.get("comment_post_ratio"), dict) else {}

    market_tasks: List[Dict[str, Any]] = []
    for symbol, block in per_symbol.items():
        gaps = block.get("gap_buckets") if isinstance(block, dict) else []
        if isinstance(gaps, list) and gaps:
            market_tasks.append({"symbol": symbol, "gap_ranges": gaps})

    aux_tasks: List[Dict[str, Any]] = []
    for modality in ("orderbook_l2", "funding_rates", "onchain_signals"):
        cov = modality_cov.get(modality) if isinstance(modality_cov.get(modality), dict) else {}
        if float(cov.get("coverage_ratio") or 0.0) < 1.0:
            aux_tasks.append({"modality": modality, "coverage": cov})

    social_posts_needed = float((modality_cov.get("social_posts") or {}).get("coverage_ratio") or 0.0) < 1.0
    social_comments_needed = float((modality_cov.get("social_comments") or {}).get("coverage_ratio") or 0.0) < 1.0

    bucket_ratio = ratio.get("bucket_ratio") if isinstance(ratio.get("bucket_ratio"), dict) else {}
    ratio_fail = (float(ratio.get("full_window_ratio") or 0.0) < 10.0) or (not bool(bucket_ratio.get("bucket_pass")))
    social_comments_needed = bool(social_comments_needed or ratio_fail)
    social_posts_needed = bool(social_posts_needed or ratio_fail)

    return {
        "market": market_tasks,
        "aux_modalities": aux_tasks,
        "social_posts": [{"required": True}] if social_posts_needed else [],
        "social_comments": [{"required": True}] if social_comments_needed else [],
    }


def _format_cmd(template: str, **kwargs: str) -> List[str]:
    text = str(template or "").strip()
    if not text:
        return []
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", str(v))
    return shlex.split(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remediate full-history completeness with fixed modality order")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--symbols", default="BTC,ETH,SOL")
    parser.add_argument("--max-rounds", type=int, default=int(os.getenv("REMEDIATION_MAX_ROUNDS", "12")))
    parser.add_argument("--market-repair-cmd", default=os.getenv("MARKET_REPAIR_CMD", ""))
    parser.add_argument("--aux-repair-cmd", default=os.getenv("AUX_REPAIR_CMD", ""))
    parser.add_argument("--comment-target-ratio", type=float, default=float(os.getenv("COMMENT_TARGET_RATIO", "10")))
    parser.add_argument("--comment-backfill-max-retry", type=int, default=int(os.getenv("COMMENT_BACKFILL_MAX_RETRY", "4")))
    parser.add_argument("--comment-backfill-page-limit", type=int, default=int(os.getenv("COMMENT_BACKFILL_PAGE_LIMIT", "60")))
    args = parser.parse_args()

    rounds: List[Dict[str, Any]] = []

    for idx in range(1, max(1, int(args.max_rounds)) + 1):
        audit_cmd = [
            "python3",
            "scripts/audit_full_history_completeness.py",
            "--database-url",
            args.database_url,
            "--start",
            args.start,
            "--timeframe",
            args.timeframe,
            "--symbols",
            args.symbols,
        ]
        if str(args.end).strip():
            audit_cmd.extend(["--end", str(args.end).strip()])
        audit = _run_json(audit_cmd)
        summary = dict(audit.get("summary") or {})

        round_obj: Dict[str, Any] = {
            "round": idx,
            "timestamp": _iso_now(),
            "summary_before": summary,
            "task_plan": {},
            "actions": [],
        }

        if bool(summary.get("history_window_complete")) and bool(summary.get("comment_ratio_ge_10x")):
            round_obj["status"] = "already_green"
            rounds.append(round_obj)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "rounds": rounds,
                        "summary": summary,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        task_plan = _collect_tasks(audit)
        round_obj["task_plan"] = task_plan

        for stage in FIXED_ORDER:
            stage_tasks = task_plan.get(stage) if isinstance(task_plan.get(stage), list) else []
            if not stage_tasks:
                continue

            if stage == "market":
                if str(args.market_repair_cmd).strip():
                    for task in stage_tasks:
                        sym = str(task.get("symbol") or "")
                        cmd = _format_cmd(
                            args.market_repair_cmd,
                            symbol=sym,
                            start=args.start,
                            end=args.end,
                            timeframe=args.timeframe,
                        )
                        if cmd:
                            round_obj["actions"].append({"stage": stage, "symbol": sym, **_run_capture(cmd)})
                else:
                    round_obj["actions"].append({"stage": stage, "status": "skipped", "reason": "market_repair_cmd_not_set"})

            elif stage == "aux_modalities":
                if str(args.aux_repair_cmd).strip():
                    for task in stage_tasks:
                        modality = str(task.get("modality") or "")
                        cmd = _format_cmd(
                            args.aux_repair_cmd,
                            modality=modality,
                            start=args.start,
                            end=args.end,
                            timeframe=args.timeframe,
                        )
                        if cmd:
                            round_obj["actions"].append({"stage": stage, "modality": modality, **_run_capture(cmd)})
                else:
                    round_obj["actions"].append({"stage": stage, "status": "skipped", "reason": "aux_repair_cmd_not_set"})

            elif stage == "social_posts":
                cmd = [
                    "python3",
                    "scripts/backfill_social_history.py",
                    "--start",
                    args.start,
                    "--timeframe",
                    args.timeframe,
                    "--symbols",
                    args.symbols,
                    "--pipeline",
                    "posts",
                    "--comment-target-ratio",
                    str(float(args.comment_target_ratio)),
                ]
                if str(args.end).strip():
                    cmd.extend(["--end", str(args.end).strip()])
                round_obj["actions"].append({"stage": stage, **_run_capture(cmd)})

            elif stage == "social_comments":
                cmd = [
                    "python3",
                    "scripts/backfill_social_history.py",
                    "--start",
                    args.start,
                    "--timeframe",
                    args.timeframe,
                    "--symbols",
                    args.symbols,
                    "--pipeline",
                    "comments",
                    "--comment-backfill-mode",
                    "--comment-target-ratio",
                    str(float(args.comment_target_ratio)),
                    "--comment-backfill-max-retry",
                    str(max(1, int(args.comment_backfill_max_retry))),
                    "--comment-backfill-page-limit",
                    str(max(1, int(args.comment_backfill_page_limit))),
                ]
                if str(args.end).strip():
                    cmd.extend(["--end", str(args.end).strip()])
                round_obj["actions"].append({"stage": stage, **_run_capture(cmd)})

        audit_after = _run_json(audit_cmd)
        summary_after = dict(audit_after.get("summary") or {})
        round_obj["summary_after"] = summary_after
        round_obj["status"] = "green" if (bool(summary_after.get("history_window_complete")) and bool(summary_after.get("comment_ratio_ge_10x"))) else "needs_more"
        rounds.append(round_obj)

        if round_obj["status"] == "green":
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "rounds": rounds,
                        "summary": summary_after,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

    final_summary = rounds[-1].get("summary_after") if rounds else {}
    print(
        json.dumps(
            {
                "status": "failed",
                "reason": "max_rounds_exhausted",
                "rounds": rounds,
                "summary": final_summary,
            },
            ensure_ascii=False,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
