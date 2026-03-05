from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor


STABLE_TOKEN_MARKERS = (
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "USDP",
    "FDUSD",
    "DAI",
    "UST",
)

LEVERAGED_TOKEN_MARKERS = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")


@dataclass(frozen=True)
class UniverseBuildConfig:
    as_of: datetime
    top_n: int
    lookback_days: int
    timeframe: str
    min_notional_usd: float
    exclude_stable: bool
    exclude_leveraged: bool
    hysteresis_keep_rank: int
    track: str = "liquid"
    rank_by: str = "volume_usd_30d"
    source: str = "db"


def _parse_ts(raw: str | datetime) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw or "").strip().replace(" ", "T")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _is_filtered_symbol(symbol: str, *, exclude_stable: bool, exclude_leveraged: bool) -> bool:
    sym = str(symbol or "").upper()
    if exclude_stable and any(marker in sym for marker in STABLE_TOKEN_MARKERS):
        return True
    if exclude_leveraged and any(sym.endswith(marker) for marker in LEVERAGED_TOKEN_MARKERS):
        return True
    return False


def _ranking_hash(payload: Dict[str, object]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _load_previous_symbols(snapshot_file: Path) -> List[str]:
    if not snapshot_file.exists():
        return []
    try:
        payload = json.loads(snapshot_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    return _normalize_symbols(payload.get("symbols") or [])


def _rank_symbols_by_liquidity(
    *,
    database_url: str,
    as_of: datetime,
    lookback_days: int,
    timeframe: str,
    min_notional_usd: float,
    exclude_stable: bool,
    exclude_leveraged: bool,
) -> List[Dict[str, object]]:
    start_ts = as_of - timedelta(days=max(1, int(lookback_days)))
    rows: List[Dict[str, object]] = []
    with psycopg2.connect(database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH bars AS (
                    SELECT
                        symbol,
                        ts,
                        ABS(COALESCE(close, 0.0) * COALESCE(volume, 0.0))::double precision AS volume_usd
                    FROM market_bars
                    WHERE timeframe = %s
                      AND ts >= %s
                      AND ts <= %s
                )
                SELECT
                    symbol,
                    SUM(volume_usd)::double precision AS volume_usd_30d,
                    SUM(volume_usd)::double precision AS notional_usd_30d,
                    COUNT(*)::bigint AS bars,
                    COUNT(DISTINCT DATE_TRUNC('day', ts))::bigint AS active_days
                FROM bars
                GROUP BY symbol
                HAVING SUM(volume_usd) >= %s
                ORDER BY volume_usd_30d DESC
                """,
                (str(timeframe), start_ts, as_of, float(min_notional_usd)),
            )
            rows = [dict(r) for r in cur.fetchall()]
    out: List[Dict[str, object]] = []
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if not sym:
            continue
        if _is_filtered_symbol(sym, exclude_stable=exclude_stable, exclude_leveraged=exclude_leveraged):
            continue
        volume_usd_30d = float(r.get("volume_usd_30d") or 0.0)
        notional_usd_30d = float(r.get("notional_usd_30d") or volume_usd_30d)
        active_days = max(1, int(r.get("active_days") or 0))
        adv_usd_30d = float(volume_usd_30d / active_days)
        out.append(
            {
                "symbol": sym,
                "volume_usd_30d": volume_usd_30d,
                "notional_usd_30d": notional_usd_30d,
                "adv_usd_30d": adv_usd_30d,
                "bars": int(r.get("bars") or 0),
                "active_days": active_days,
            }
        )
    out.sort(key=lambda x: float(x["volume_usd_30d"]), reverse=True)
    return out


def _rank_score(row: Dict[str, object], rank_by: str) -> float:
    rb = str(rank_by or "volume_usd_30d").strip().lower()
    if rb not in {"volume_usd_30d", "adv_usd_30d", "notional_usd_30d"}:
        raise RuntimeError(f"unsupported_rank_by:{rank_by}")
    return float(row.get(rb) or 0.0)


def build_top_universe_snapshot(
    *,
    database_url: str,
    cfg: UniverseBuildConfig,
    previous_snapshot_file: Path | None = None,
) -> Dict[str, object]:
    source = str(cfg.source or "db").strip().lower()
    if source != "db":
        raise RuntimeError(f"unsupported_universe_source:{source}")
    ranked = _rank_symbols_by_liquidity(
        database_url=database_url,
        as_of=cfg.as_of,
        lookback_days=cfg.lookback_days,
        timeframe=cfg.timeframe,
        min_notional_usd=cfg.min_notional_usd,
        exclude_stable=cfg.exclude_stable,
        exclude_leveraged=cfg.exclude_leveraged,
    )
    rank_by = str(cfg.rank_by or "volume_usd_30d").strip().lower()
    ranked = sorted(ranked, key=lambda r: _rank_score(r, rank_by), reverse=True)
    if len(ranked) < int(cfg.top_n):
        raise RuntimeError(f"insufficient_ranked_symbols:{len(ranked)}:{cfg.top_n}")

    rank_map = {str(r["symbol"]): i + 1 for i, r in enumerate(ranked)}
    base_selection = [str(r["symbol"]) for r in ranked[: int(cfg.top_n)]]

    prev_symbols: List[str] = []
    if previous_snapshot_file is not None:
        prev_symbols = _load_previous_symbols(previous_snapshot_file)
    selected: List[str] = []
    if prev_symbols:
        # Keep previous members that remain inside hysteresis rank first to reduce churn.
        keep = [s for s in prev_symbols if int(rank_map.get(s, 10**9)) <= int(cfg.hysteresis_keep_rank)]
        selected.extend(keep)
    for sym in base_selection:
        if sym not in selected:
            selected.append(sym)
    if len(selected) < int(cfg.top_n):
        for row in ranked:
            sym = str(row.get("symbol") or "").upper()
            if not sym or sym in selected:
                continue
            selected.append(sym)
            if len(selected) >= int(cfg.top_n):
                break
    selected = selected[: int(cfg.top_n)]

    symbols = _normalize_symbols(selected)
    symbol_rows = []
    for sym in symbols:
        rank = int(rank_map.get(sym, 10**9))
        src = next((r for r in ranked if str(r["symbol"]) == sym), None)
        symbol_rows.append(
            {
                "symbol": sym,
                "rank": rank,
                "rank_score": _rank_score(src or {}, rank_by),
                "rank_by": rank_by,
                "volume_usd_30d": float((src or {}).get("volume_usd_30d") or 0.0),
                "adv_usd_30d": float((src or {}).get("adv_usd_30d") or 0.0),
                "notional_usd_30d": float((src or {}).get("notional_usd_30d") or 0.0),
                "bars": int((src or {}).get("bars") or 0),
                "active_days": int((src or {}).get("active_days") or 0),
                "kept_by_hysteresis": bool(sym in prev_symbols and sym not in base_selection),
            }
        )

    hash_payload = {
        "track": str(cfg.track),
        "as_of": cfg.as_of.isoformat(),
        "top_n": int(cfg.top_n),
        "lookback_days": int(cfg.lookback_days),
        "timeframe": str(cfg.timeframe),
        "min_notional_usd": float(cfg.min_notional_usd),
        "exclude_stable": bool(cfg.exclude_stable),
        "exclude_leveraged": bool(cfg.exclude_leveraged),
        "hysteresis_keep_rank": int(cfg.hysteresis_keep_rank),
        "rank_by": rank_by,
        "source": source,
        "symbols": symbols,
    }
    snap_hash = _ranking_hash(hash_payload)

    return {
        "track": str(cfg.track),
        "as_of": cfg.as_of.isoformat(),
        "symbols": symbols,
        "snapshot_hash": snap_hash,
        "source": "ranked_notional_hysteresis",
        "criteria": {
            "top_n": int(cfg.top_n),
            "lookback_days": int(cfg.lookback_days),
            "timeframe": str(cfg.timeframe),
            "min_notional_usd": float(cfg.min_notional_usd),
            "exclude_stable": bool(cfg.exclude_stable),
            "exclude_leveraged": bool(cfg.exclude_leveraged),
            "hysteresis_keep_rank": int(cfg.hysteresis_keep_rank),
            "rank_by": rank_by,
            "source": source,
        },
        "symbol_rows": symbol_rows,
        "previous_symbols": prev_symbols,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_universe_snapshot(snapshot: Dict[str, object], output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_file
