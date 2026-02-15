#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import request


def _http_json(url: str, method: str = "GET", payload: Dict[str, Any] | None = None, timeout: int = 30) -> Dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)


def export_samples(api_base: str, limit: int, min_quality_score: float, out_dir: Path) -> Path:
    data = _http_json(
        f"{api_base.rstrip('/')}/api/v2/data-quality/sample",
        method="POST",
        payload={"limit": limit, "min_quality_score": min_quality_score},
    )
    items = data.get("items", [])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"data_quality_samples_{ts}.json"
    out_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    source_cnt = Counter((x.get("source_name") or "unknown") for x in items)
    print(f"exported={len(items)} file={out_file}")
    print("by_source=", dict(source_cnt))
    print("review_format=[{\"audit_id\":123,\"reviewer\":\"alice\",\"verdict\":\"correct|incorrect|uncertain\",\"note\":\"...\"}]")
    return out_file


def apply_reviews(api_base: str, review_file: Path) -> None:
    records: List[Dict[str, Any]] = json.loads(review_file.read_text(encoding="utf-8"))
    ok = 0
    failed = 0
    for rec in records:
        try:
            _http_json(
                f"{api_base.rstrip('/')}/api/v2/data-quality/audit",
                method="POST",
                payload={
                    "audit_id": int(rec["audit_id"]),
                    "reviewer": str(rec["reviewer"]),
                    "verdict": str(rec["verdict"]),
                    "note": str(rec.get("note") or ""),
                },
            )
            ok += 1
        except Exception:
            failed += 1
    print(f"applied_reviews_ok={ok} failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly data quality audit helper for V2 API")
    parser.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--review-file", default="", help="JSON file with manual review results to apply")
    args = parser.parse_args()

    export_samples(args.api_base, args.limit, args.min_quality_score, Path(args.out_dir))
    if args.review_file:
        apply_reviews(args.api_base, Path(args.review_file))


if __name__ == "__main__":
    main()
