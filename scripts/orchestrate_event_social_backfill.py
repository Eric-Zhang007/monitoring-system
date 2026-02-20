#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _parse_dt_utc(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_datetime")
    norm = text.replace(" ", "T")
    if norm.endswith("Z"):
        norm = norm[:-1] + "+00:00"
    dt_obj = datetime.fromisoformat(norm)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def _to_iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_windows(start: datetime, end: datetime, chunk_days: int) -> Iterable[Tuple[datetime, datetime]]:
    step = max(1, int(chunk_days))
    cur = start
    while cur < end:
        nxt = min(end, cur + timedelta(days=step))
        yield cur, nxt
        cur = nxt


def _chunk_id(idx: int, start: datetime, end: datetime) -> str:
    return f"{idx:05d}_{start.strftime('%Y%m%dT%H%M%SZ')}-{end.strftime('%Y%m%dT%H%M%SZ')}"


def _save_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(tmp), str(p))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("invalid_checkpoint_not_object")
    return obj


def _detect_language(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return "other"
    if re.search(r"[\u4e00-\u9fff]", body):
        return "zh"
    if re.search(r"[A-Za-z]", body):
        return "en"
    return "other"


def _parse_event_dt(raw: object, fallback: datetime) -> datetime:
    try:
        return _parse_dt_utc(str(raw or ""))
    except Exception:
        return fallback


def _normalize_event_time_alignment(event: Dict[str, Any], fallback_now: datetime) -> Tuple[Dict[str, Any], bool]:
    occurred_at = _parse_event_dt(event.get("occurred_at"), fallback_now)
    published_at = _parse_event_dt(event.get("published_at"), occurred_at)
    if published_at < occurred_at:
        published_at = occurred_at
    available_at = _parse_event_dt(event.get("available_at"), published_at)
    if available_at < published_at:
        available_at = published_at
    effective_at = _parse_event_dt(event.get("effective_at"), available_at)
    if effective_at < available_at:
        effective_at = available_at

    event["occurred_at"] = _to_iso_z(occurred_at)
    event["published_at"] = _to_iso_z(published_at)
    event["available_at"] = _to_iso_z(available_at)
    event["effective_at"] = _to_iso_z(effective_at)

    source_latency_ms = int(max(0.0, (available_at - occurred_at).total_seconds() * 1000.0))
    event["source_latency_ms"] = int(max(source_latency_ms, int(event.get("source_latency_ms") or 0)))
    event["latency_ms"] = int(max(source_latency_ms, int(event.get("latency_ms") or 0)))
    monotonic = bool(occurred_at <= published_at <= available_at <= effective_at)
    return event, monotonic


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _enrich_chunk_file(
    *,
    path: str,
    stream: str,
    chunk_start: datetime,
    chunk_end: datetime,
    pipeline_tag: str,
    run_id: str,
    language_targets: Sequence[str],
    google_locales: Sequence[str],
) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {"input_rows": 0, "output_rows": 0, "dropped_outside_window": 0, "dropped_language": 0, "alignment_violations": 0}
    out: List[Dict[str, Any]] = []
    dropped_outside_window = 0
    dropped_language = 0
    alignment_violations = 0
    lang_targets = {str(x).strip().lower() for x in language_targets if str(x).strip()}
    fallback_now = datetime.now(timezone.utc)
    for ev in rows:
        if not isinstance(ev, dict):
            continue
        ev, monotonic = _normalize_event_time_alignment(ev, fallback_now=fallback_now)
        if not monotonic:
            alignment_violations += 1
        occurred_at = _parse_event_dt(ev.get("occurred_at"), fallback_now)
        if occurred_at < chunk_start or occurred_at > chunk_end:
            dropped_outside_window += 1
            continue
        payload = dict(ev.get("payload") or {})
        text_blob = f"{ev.get('title') or ''}\n{payload.get('summary') or ''}"
        language = str(payload.get("language") or "").strip().lower() or _detect_language(text_blob)
        payload["language"] = language
        if lang_targets and language not in lang_targets:
            dropped_language += 1
            continue
        provenance = dict(payload.get("provenance") or {})
        provenance.update(
            {
                "pipeline_tag": str(pipeline_tag),
                "orchestrator_run_id": str(run_id),
                "stream": str(stream),
                "window_start": _to_iso_z(chunk_start),
                "window_end": _to_iso_z(chunk_end),
                "language_targets": sorted(list(lang_targets)),
                "google_locales": list(google_locales),
                "orchestrator_script": "scripts/orchestrate_event_social_backfill.py",
                "enriched_at": _to_iso_z(datetime.now(timezone.utc)),
            }
        )
        time_alignment = dict(payload.get("time_alignment") or {})
        time_alignment.update(
            {
                "alignment_mode": "strict_asof_v1",
                "occurred_at": str(ev.get("occurred_at")),
                "published_at": str(ev.get("published_at")),
                "available_at": str(ev.get("available_at")),
                "effective_at": str(ev.get("effective_at")),
                "monotonic_non_decreasing": bool(monotonic),
            }
        )
        payload["provenance"] = provenance
        payload["time_alignment"] = time_alignment
        ev["payload"] = payload
        out.append(ev)

    out.sort(key=lambda x: str(x.get("occurred_at") or ""))
    _write_jsonl(path, out)
    return {
        "input_rows": len(rows),
        "output_rows": len(out),
        "dropped_outside_window": int(dropped_outside_window),
        "dropped_language": int(dropped_language),
        "alignment_violations": int(alignment_violations),
    }


def _event_key(ev: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(ev.get("title") or "").strip().lower(),
            str(ev.get("source_url") or "").strip().lower(),
            str(ev.get("occurred_at") or "").strip(),
            str(((ev.get("payload") or {}) if isinstance(ev.get("payload"), dict) else {}).get("provider") or "").strip().lower(),
        ]
    )


def _merge_stream_chunks(chunk_files: Sequence[str], out_path: str) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    for path in chunk_files:
        all_rows.extend(_read_jsonl(path))
    raw_n = len(all_rows)
    dedup: Dict[str, Dict[str, Any]] = {}
    for ev in all_rows:
        if not isinstance(ev, dict):
            continue
        dedup[_event_key(ev)] = ev
    rows = sorted(dedup.values(), key=lambda x: str(x.get("occurred_at") or ""))
    _write_jsonl(out_path, rows)
    return {"chunks": len(chunk_files), "rows_raw": raw_n, "rows_final": len(rows), "out_jsonl": out_path}


def _run_command(cmd: List[str], retries: int, backoff_sec: float) -> Dict[str, Any]:
    attempts = max(1, int(retries))
    last_error = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode == 0:
            return {
                "status": "ok",
                "attempt": attempt,
                "returncode": 0,
                "stdout_tail": (proc.stdout or "").strip()[-1200:],
                "stderr_tail": (proc.stderr or "").strip()[-800:],
            }
        last_error = (proc.stderr or proc.stdout or "").strip()[-1200:]
        if attempt < attempts:
            sleep_s = min(90.0, float(backoff_sec) * (2 ** (attempt - 1)))
            time.sleep(max(0.1, sleep_s))
    return {"status": "failed", "returncode": 1, "error": last_error[:1200]}


def _cmd_text(cmd: Sequence[str]) -> str:
    return " ".join(str(x) for x in cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Task H orchestration: long-horizon 2018-now event/social backfill with checkpoints")
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--chunk-days", type=int, default=max(1, int(os.getenv("BACKFILL_CHUNK_DAYS", "30"))))
    ap.add_argument("--max-chunks", type=int, default=0, help="optional staged cap")
    ap.add_argument("--checkpoint-file", default="artifacts/backfill_2018_now/checkpoints/task_h_2018_now.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-command-retries", type=int, default=max(1, int(os.getenv("BACKFILL_CMD_MAX_RETRIES", "2"))))
    ap.add_argument("--retry-backoff-sec", type=float, default=float(os.getenv("BACKFILL_CMD_RETRY_BACKOFF_SEC", "1.0")))
    ap.add_argument("--pipeline-tag", default=os.getenv("BACKFILL_PIPELINE_TAG", "task_h_2018_now"))
    ap.add_argument("--run-id", default=os.getenv("BACKFILL_RUN_ID", ""))
    ap.add_argument("--language-targets", default=os.getenv("BACKFILL_LANGUAGE_TARGETS", "en,zh"))
    ap.add_argument("--google-locales", default=os.getenv("BACKFILL_GOOGLE_LOCALES", "US:en,CN:zh-Hans"))
    ap.add_argument("--social-sources", default=os.getenv("BACKFILL_SOCIAL_SOURCES", "twitter,reddit,youtube,telegram"))
    ap.add_argument("--skip-events", action="store_true")
    ap.add_argument("--skip-social", action="store_true")
    ap.add_argument("--event-disable-google", action="store_true", default=str(os.getenv("BACKFILL_EVENT_DISABLE_GOOGLE", "0")).strip().lower() in {"1", "true", "yes", "on"})
    ap.add_argument("--event-disable-gdelt", action="store_true", default=str(os.getenv("BACKFILL_EVENT_DISABLE_GDELT", "0")).strip().lower() in {"1", "true", "yes", "on"})
    ap.add_argument("--event-disable-official-rss", action="store_true", default=str(os.getenv("BACKFILL_EVENT_DISABLE_OFFICIAL_RSS", "0")).strip().lower() in {"1", "true", "yes", "on"})
    ap.add_argument("--event-disable-source-balance", action="store_true", default=str(os.getenv("BACKFILL_EVENT_DISABLE_SOURCE_BALANCE", "0")).strip().lower() in {"1", "true", "yes", "on"})
    ap.add_argument("--event-gdelt-max-records", type=int, default=max(10, int(os.getenv("BACKFILL_EVENT_GDELT_MAX_RECORDS", "50"))))
    ap.add_argument("--event-day-step", type=int, default=max(1, int(os.getenv("BACKFILL_EVENT_DAY_STEP", "7"))))
    ap.add_argument("--event-script", default="scripts/build_multisource_events_2025.py")
    ap.add_argument("--social-script", default="scripts/backfill_social_history.py")
    ap.add_argument("--chunk-dir", default="artifacts/backfill_2018_now/chunks")
    ap.add_argument("--out-events-jsonl", default="artifacts/backfill_2018_now/events_2018_now.jsonl")
    ap.add_argument("--out-social-jsonl", default="artifacts/backfill_2018_now/social_2018_now.jsonl")
    args = ap.parse_args()
    python_bin = str(os.getenv("PYTHON_BIN", "python3")).strip() or "python3"

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range_end_must_be_gt_start")

    run_id = str(args.run_id).strip() or f"{int(time.time())}"
    language_targets = [x.strip().lower() for x in str(args.language_targets).split(",") if x.strip()]
    if not language_targets:
        language_targets = ["en", "zh"]
    google_locales = [x.strip() for x in str(args.google_locales).split(",") if x.strip()]
    if not google_locales:
        google_locales = ["US:en", "CN:zh-Hans"]

    windows = list(_iter_windows(start_dt, end_dt, chunk_days=max(1, int(args.chunk_days))))
    checkpoint_file = str(args.checkpoint_file).strip()
    checkpoint: Dict[str, Any]
    run_spec = {
        "start": _to_iso_z(start_dt),
        "end": _to_iso_z(end_dt),
        "chunk_days": int(args.chunk_days),
        "language_targets": list(language_targets),
        "google_locales": list(google_locales),
    }
    if bool(args.resume) and checkpoint_file and os.path.exists(checkpoint_file):
        checkpoint = _load_json(checkpoint_file)
        ck_spec = dict(checkpoint.get("run_spec") or {})
        if ck_spec != run_spec:
            raise RuntimeError(
                json.dumps(
                    {
                        "checkpoint_mismatch": True,
                        "checkpoint_file": checkpoint_file,
                        "checkpoint_run_spec": ck_spec,
                        "current_run_spec": run_spec,
                    },
                    ensure_ascii=False,
                )
            )
        if not isinstance(checkpoint.get("completed_chunks"), dict):
            checkpoint["completed_chunks"] = {}
    else:
        checkpoint = {
            "version": 1,
            "created_at": _to_iso_z(datetime.now(timezone.utc)),
            "updated_at": _to_iso_z(datetime.now(timezone.utc)),
            "pipeline_tag": str(args.pipeline_tag),
            "run_id": run_id,
            "run_spec": run_spec,
            "completed_chunks": {},
        }
        if checkpoint_file and not bool(args.dry_run):
            _save_json(checkpoint_file, checkpoint)
    completed = set((checkpoint.get("completed_chunks") or {}).keys())

    chunk_dir = Path(str(args.chunk_dir))
    chunk_dir.mkdir(parents=True, exist_ok=True)

    dry_plan: List[Dict[str, Any]] = []
    chunk_results: List[Dict[str, Any]] = []
    chunks_attempted = 0
    chunks_skipped = 0

    for idx, (w_start, w_end) in enumerate(windows):
        if int(args.max_chunks) > 0 and chunks_attempted >= int(args.max_chunks):
            break
        cid = _chunk_id(idx, w_start, w_end)
        if cid in completed:
            chunks_skipped += 1
            continue
        chunks_attempted += 1

        event_chunk = str(chunk_dir / f"{cid}.events.jsonl")
        social_chunk = str(chunk_dir / f"{cid}.social.jsonl")
        event_cmd = [
            python_bin,
            str(args.event_script),
            "--start",
            _to_iso_z(w_start),
            "--end",
            _to_iso_z(w_end),
            "--day-step",
            str(max(1, int(args.event_day_step))),
            "--google-locales",
            ",".join(google_locales),
            "--out-jsonl",
            event_chunk,
        ]
        if bool(args.event_disable_google):
            event_cmd.append("--disable-google")
        if bool(args.event_disable_gdelt):
            event_cmd.append("--disable-gdelt")
        if bool(args.event_disable_official_rss):
            event_cmd.append("--disable-official-rss")
        if bool(args.event_disable_source_balance):
            event_cmd.append("--disable-source-balance")
        event_cmd.extend(["--gdelt-max-records", str(max(10, int(args.event_gdelt_max_records)))])
        social_cmd = [
            python_bin,
            str(args.social_script),
            "--start",
            _to_iso_z(w_start),
            "--end",
            _to_iso_z(w_end),
            "--language-targets",
            ",".join(language_targets),
            "--sources",
            str(args.social_sources),
            "--pipeline-tag",
            str(args.pipeline_tag),
            "--provenance-run-id",
            str(run_id),
            "--out-jsonl",
            social_chunk,
        ]
        if bool(args.dry_run):
            dry_plan.append(
                {
                    "chunk": cid,
                    "start": _to_iso_z(w_start),
                    "end": _to_iso_z(w_end),
                    "event_cmd": _cmd_text(event_cmd),
                    "social_cmd": _cmd_text(social_cmd),
                    "skip_events": bool(args.skip_events),
                    "skip_social": bool(args.skip_social),
                }
            )
            continue

        result: Dict[str, Any] = {
            "chunk": cid,
            "start": _to_iso_z(w_start),
            "end": _to_iso_z(w_end),
            "event_chunk": event_chunk,
            "social_chunk": social_chunk,
        }
        if not bool(args.skip_events):
            event_run = _run_command(
                event_cmd,
                retries=int(args.max_command_retries),
                backoff_sec=float(args.retry_backoff_sec),
            )
            result["event_command"] = _cmd_text(event_cmd)
            result["event_run"] = event_run
            if event_run.get("status") != "ok":
                raise RuntimeError(json.dumps({"failed_chunk": cid, "stream": "events", "details": event_run}, ensure_ascii=False))
            result["event_enrichment"] = _enrich_chunk_file(
                path=event_chunk,
                stream="events",
                chunk_start=w_start,
                chunk_end=w_end,
                pipeline_tag=str(args.pipeline_tag),
                run_id=run_id,
                language_targets=language_targets,
                google_locales=google_locales,
            )
        if not bool(args.skip_social):
            social_run = _run_command(
                social_cmd,
                retries=int(args.max_command_retries),
                backoff_sec=float(args.retry_backoff_sec),
            )
            result["social_command"] = _cmd_text(social_cmd)
            result["social_run"] = social_run
            if social_run.get("status") != "ok":
                raise RuntimeError(json.dumps({"failed_chunk": cid, "stream": "social", "details": social_run}, ensure_ascii=False))
            result["social_enrichment"] = _enrich_chunk_file(
                path=social_chunk,
                stream="social",
                chunk_start=w_start,
                chunk_end=w_end,
                pipeline_tag=str(args.pipeline_tag),
                run_id=run_id,
                language_targets=language_targets,
                google_locales=google_locales,
            )

        result["completed_at"] = _to_iso_z(datetime.now(timezone.utc))
        chunk_results.append(result)
        checkpoint.setdefault("completed_chunks", {})[cid] = result
        checkpoint["updated_at"] = _to_iso_z(datetime.now(timezone.utc))
        if checkpoint_file:
            _save_json(checkpoint_file, checkpoint)

    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "pipeline_tag": str(args.pipeline_tag),
                    "run_id": run_id,
                    "start": _to_iso_z(start_dt),
                    "end": _to_iso_z(end_dt),
                    "chunk_days": int(args.chunk_days),
                    "chunks_total": len(windows),
                    "chunks_planned": len(dry_plan),
                    "chunks_skipped_resume": chunks_skipped,
                    "checkpoint_file": checkpoint_file or None,
                    "plan": dry_plan,
                },
                ensure_ascii=False,
            )
        )
        return 0

    completed_chunks = checkpoint.get("completed_chunks") or {}
    event_chunks = [
        str((completed_chunks[cid] or {}).get("event_chunk") or "")
        for cid in sorted(completed_chunks.keys())
        if str((completed_chunks[cid] or {}).get("event_chunk") or "").strip()
    ]
    social_chunks = [
        str((completed_chunks[cid] or {}).get("social_chunk") or "")
        for cid in sorted(completed_chunks.keys())
        if str((completed_chunks[cid] or {}).get("social_chunk") or "").strip()
    ]

    merged_events = {"status": "skipped"}
    merged_social = {"status": "skipped"}
    if not bool(args.skip_events):
        merged_events = _merge_stream_chunks(event_chunks, out_path=str(args.out_events_jsonl))
    if not bool(args.skip_social):
        merged_social = _merge_stream_chunks(social_chunks, out_path=str(args.out_social_jsonl))

    print(
        json.dumps(
            {
                "status": "ok",
                "pipeline_tag": str(args.pipeline_tag),
                "run_id": run_id,
                "start": _to_iso_z(start_dt),
                "end": _to_iso_z(end_dt),
                "chunk_days": int(args.chunk_days),
                "chunks_total": len(windows),
                "chunks_attempted": chunks_attempted,
                "chunks_skipped_resume": chunks_skipped,
                "checkpoint_file": checkpoint_file or None,
                "events_merged": merged_events,
                "social_merged": merged_social,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
