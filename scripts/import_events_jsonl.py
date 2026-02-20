#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def _load_events(jsonl_path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _batch(items: List[Dict], size: int) -> List[List[Dict]]:
    n = max(1, int(size))
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Import event JSONL into canonical events/entities/event_links tables")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    jsonl = str(args.jsonl).strip()
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)

    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from schemas_v2 import Event  # type: ignore
    from v2_repository import V2Repository  # type: ignore

    rows = _load_events(jsonl)
    repo = V2Repository(db_url=args.database_url)
    accepted = 0
    inserted = 0
    deduped = 0
    event_ids: List[int] = []
    for chunk in _batch(rows, size=int(args.batch_size)):
        evs = []
        for raw in chunk:
            try:
                evs.append(Event(**raw))
            except Exception:
                continue
        if not evs:
            continue
        a, i, d, ids = repo.ingest_events(evs)
        accepted += int(a)
        inserted += int(i)
        deduped += int(d)
        event_ids.extend([int(x) for x in ids])

    print(
        json.dumps(
            {
                "status": "ok",
                "jsonl": jsonl,
                "rows_loaded": len(rows),
                "accepted": accepted,
                "inserted": inserted,
                "deduplicated": deduped,
                "event_ids_count": len(event_ids),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
