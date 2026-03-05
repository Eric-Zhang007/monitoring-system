#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "INGEST_PROGRESS.md"

TOP10 = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]
TOP10_CKPT = ROOT / "artifacts/checkpoints/all_timeframes/ingest_perp_1m_5559965e699675e6.json"
TOP11_CKPT = ROOT / "artifacts/checkpoints/top11_50_crypto/ingest_perp_3m_ebe0b4ba1563f9a9.json"
TOP11_ENV = ROOT / "artifacts/universe/top11_50_crypto_exchange.env"

TOP10_AGENT_ID = os.getenv("TOP10_INGEST_AGENT_ID", "019cad7d-6997-7e42-b5cf-1ac1a333197a")
TOP11_AGENT_ID = os.getenv("TOP11_INGEST_AGENT_ID", "019cad7d-69e1-7753-bd4c-05e06fb97a23")


def run(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def get_db_url() -> str:
    out = run("cd %s && rg -n '^DATABASE_URL=' .env | head -n1" % str(ROOT))
    if "DATABASE_URL=" not in out:
        raise RuntimeError("DATABASE_URL missing in .env")
    return out.split("DATABASE_URL=", 1)[1]


def ckpt_info(path: Path, total: int) -> dict:
    if not path.exists():
        return {"exists": False, "completed_chunks": 0, "total_chunks": total, "pct": 0.0, "updated_at": None, "last_chunk": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = data.get("completed_chunks", {})
    if not isinstance(completed, dict):
        completed = {}
    keys = sorted(completed.keys())
    done = len(keys)
    return {
        "exists": True,
        "completed_chunks": done,
        "total_chunks": total,
        "pct": round(100.0 * done / total, 2),
        "updated_at": data.get("updated_at"),
        "last_chunk": keys[-1] if keys else None,
    }


def db_rows(db_url: str, symbols: list[str]) -> list[str]:
    quoted = ",".join("'%s'" % s for s in symbols)
    q = (
        "SELECT timeframe||E'\\t'||COUNT(DISTINCT symbol)||E'\\t'||MIN(ts)||E'\\t'||MAX(ts)||E'\\t'||COUNT(*) "
        "FROM market_bars WHERE symbol IN (%s) GROUP BY timeframe ORDER BY timeframe;" % quoted
    )
    out = run('psql "%s" -At -F $\'\\t\' -c "%s"' % (db_url, q))
    return out.splitlines() if out else []


def top11_symbols() -> list[str]:
    if not TOP11_ENV.exists():
        return []
    for line in TOP11_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("SYMBOL_MAP="):
            raw = line.split("=", 1)[1]
            return [x.split(":", 1)[1] for x in raw.split(",") if ":" in x]
    return []


def process_list() -> str:
    try:
        out = run("ps -eo pid,etimes,cmd | rg -i 'ingest_bitget_market_bars.py --database-url' | rg -v rg || true")
    except Exception:
        out = ""
    return out or "<no running ingest process detected>"


def main() -> None:
    db_url = get_db_url()
    t10 = ckpt_info(TOP10_CKPT, 199)
    t11 = ckpt_info(TOP11_CKPT, 150)
    rows_t10 = db_rows(db_url, TOP10)
    rows_t11 = db_rows(db_url, top11_symbols()) if top11_symbols() else []

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines: list[str] = []
    lines.append("# Ingestion Progress (Single Source of Truth)")
    lines.append("")
    lines.append(f"Last updated: **{now}**")
    lines.append("")
    lines.append("## Stable Runners")
    lines.append(f"- top10_1m awaiter agent id: `{TOP10_AGENT_ID}`")
    lines.append(f"- top11_50_3m awaiter agent id: `{TOP11_AGENT_ID}`")
    lines.append("- Ground truth process list (must be non-empty while running):")
    lines.append("```text")
    lines.append(process_list())
    lines.append("```")
    lines.append("")
    lines.append("## Checkpoint Progress")
    lines.append(f"- top10 1m checkpoint: `{TOP10_CKPT.relative_to(ROOT)}`")
    lines.append(f"  - completed: **{t10['completed_chunks']} / {t10['total_chunks']}** ({t10['pct']}%)")
    lines.append(f"  - updated_at: `{t10['updated_at']}`")
    lines.append(f"  - last_chunk: `{t10['last_chunk']}`")
    lines.append(f"- top11~50 3m checkpoint: `{TOP11_CKPT.relative_to(ROOT)}`")
    lines.append(f"  - completed: **{t11['completed_chunks']} / {t11['total_chunks']}** ({t11['pct']}%)")
    lines.append(f"  - updated_at: `{t11['updated_at']}`")
    lines.append(f"  - last_chunk: `{t11['last_chunk']}`")
    lines.append("")
    lines.append("## DB Coverage Snapshot (actual rows)")
    lines.append("Top10 (BTC/ETH/SOL/BNB/XRP/ADA/DOGE/TRX/AVAX/LINK):")
    lines.append("```text")
    lines.extend(rows_t10 or ["<no rows>"])
    lines.append("```")
    lines.append("Top11~50 (current selection env):")
    lines.append("```text")
    lines.extend(rows_t11 or ["<no rows>"])
    lines.append("```")
    lines.append("")
    lines.append("## Current Facts (no guessing)")
    lines.append("- Top10 currently has data in `1m/5m/1h` in DB snapshot above.")
    lines.append("- Top11~50 currently has `3m` in DB snapshot above.")
    lines.append("- Full top50 all-timeframes is NOT complete yet.")

    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(DOC))


if __name__ == "__main__":
    main()
