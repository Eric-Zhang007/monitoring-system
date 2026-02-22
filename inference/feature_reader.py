from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from features.sequence import build_sequence


def fetch_sequence(
    *,
    db_url: str,
    symbol: str,
    end_ts: datetime | str,
    lookback: int,
) -> Dict[str, Any]:
    seq = build_sequence(db_url=db_url, symbol=symbol, end_ts=end_ts, lookback=lookback)
    return {
        "values": seq.values,
        "mask": seq.mask,
        "schema_hash": seq.schema_hash,
        "symbol": seq.symbol,
        "start_ts": seq.start_ts,
        "end_ts": seq.end_ts,
        "bucket_interval": seq.bucket_interval,
        "coverage_summary": seq.coverage_summary,
    }
