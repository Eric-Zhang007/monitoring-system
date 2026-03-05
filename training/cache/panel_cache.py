from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from features.feature_contract import FEATURE_DIM, FEATURE_INDEX, SCHEMA_HASH


REGIME_FEATURE_NAMES = (
    "realized_vol",
    "vol_of_vol",
    "return_skew_proxy",
    "spread_proxy",
    "depth_proxy",
    "imbalance_proxy",
    "funding_rate",
    "funding_zscore",
    "basis_proxy",
    "open_interest",
    "oi_change",
    "liquidation_proxy",
    "trend_strength",
    "text_coverage",
    "text_disagreement",
    "event_density",
)


DEFAULT_CONTEXT_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
MULTI_TF_CONTEXT_FIELDS = ("missing", "ret_1", "volume", "lag_sec")


@dataclass(frozen=True)
class CacheBuildConfig:
    universe_snapshot_file: Path
    start_ts: datetime
    end_ts: datetime
    bar_size: str
    lookback_len: int
    horizons: Sequence[str]
    feature_contract_hash: str
    output_dir: Path
    database_url: str
    incremental: bool = False
    incremental_warmup_steps: int = 288
    context_timeframes: Sequence[str] = DEFAULT_CONTEXT_TIMEFRAMES
    require_multi_tf_context: bool = True


def _parse_ts(raw: Any) -> datetime:
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


def _parse_bar_seconds(raw: str) -> int:
    text = str(raw or "5m").strip().lower()
    if text.endswith("m"):
        return max(1, int(text[:-1])) * 60
    if text.endswith("h"):
        return max(1, int(text[:-1])) * 3600
    if text.endswith("d"):
        return max(1, int(text[:-1])) * 86400
    return max(1, int(text))


def _parse_horizon_seconds(h: str) -> int:
    text = str(h or "").strip().lower()
    if text.endswith("m"):
        return max(1, int(text[:-1])) * 60
    if text.endswith("h"):
        return max(1, int(text[:-1])) * 3600
    if text.endswith("d"):
        return max(1, int(text[:-1])) * 86400
    raise ValueError(f"unsupported_horizon:{h}")


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    n = int(max(2, win))
    for i in range(x.shape[0]):
        lo = max(0, i - n + 1)
        seg = x[lo : i + 1]
        out[i] = float(np.std(seg)) if seg.size > 1 else 0.0
    return out


def _rolling_zscore(x: np.ndarray, win: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    n = int(max(2, win))
    for i in range(x.shape[0]):
        lo = max(0, i - n + 1)
        seg = x[lo : i + 1]
        mu = float(np.mean(seg)) if seg.size else 0.0
        sd = float(np.std(seg)) if seg.size > 1 else 0.0
        out[i] = (float(x[i]) - mu) / max(1e-8, sd)
    return out


def _rolling_skew_proxy(x: np.ndarray, win: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    n = int(max(8, win))
    for i in range(x.shape[0]):
        lo = max(0, i - n + 1)
        seg = x[lo : i + 1]
        if seg.size < 3:
            out[i] = 0.0
            continue
        mu = float(np.mean(seg))
        sd = float(np.std(seg))
        if sd <= 1e-8:
            out[i] = 0.0
            continue
        z = (seg - mu) / sd
        out[i] = float(np.mean(z**3))
    return out


def _safe_idx(name: str) -> int | None:
    v = FEATURE_INDEX.get(name)
    if v is None:
        return None
    return int(v)


def _extract_col(values: np.ndarray, mask: np.ndarray, key: str, default: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    idx = _safe_idx(key)
    if idx is None or int(idx) >= int(values.shape[1]):
        return np.full((values.shape[0],), float(default), dtype=np.float64), np.ones((values.shape[0],), dtype=np.uint8)
    col = values[:, idx].astype(np.float64)
    msk = mask[:, idx].astype(np.uint8)
    col = np.where(msk > 0, float(default), col)
    return col, msk


def compute_regime_features(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ret_1, ret_1_m = _extract_col(values, mask, "ret_1")
    vol_12, vol_12_m = _extract_col(values, mask, "vol_12")
    spread, spread_m = _extract_col(values, mask, "orderbook_spread_bps")
    depth, depth_m = _extract_col(values, mask, "orderbook_depth_total")
    imb, imb_m = _extract_col(values, mask, "orderbook_imbalance")
    funding, funding_m = _extract_col(values, mask, "funding_rate")
    basis, basis_m = _extract_col(values, mask, "basis_rate")
    oi, oi_m = _extract_col(values, mask, "open_interest")
    ret_48, ret_48_m = _extract_col(values, mask, "ret_48")
    ret_288, ret_288_m = _extract_col(values, mask, "ret_288")
    t_cov, t_cov_m = _extract_col(values, mask, "text_coverage")
    t_dis, t_dis_m = _extract_col(values, mask, "text_disagreement")
    e_count, e_count_m = _extract_col(values, mask, "event_count_1h")

    realized_vol = _rolling_std(ret_1, win=24)
    vol_of_vol = _rolling_std(vol_12, win=24)
    skew = _rolling_skew_proxy(ret_1, win=48)
    funding_z = _rolling_zscore(funding, win=96)
    oi_change = np.concatenate([[0.0], np.diff(oi)])
    liq_proxy = np.maximum(0.0, -oi_change) * (1.0 + np.maximum(0.0, spread)) * (1.0 + realized_vol)
    trend = 0.6 * ret_48 + 0.4 * ret_288

    rf = np.stack(
        [
            realized_vol,
            vol_of_vol,
            skew,
            spread,
            np.log1p(np.maximum(0.0, depth)),
            imb,
            funding,
            funding_z,
            basis,
            oi,
            oi_change,
            liq_proxy,
            trend,
            t_cov,
            t_dis,
            e_count,
        ],
        axis=1,
    ).astype(np.float32)
    rm = np.stack(
        [
            np.maximum(ret_1_m, vol_12_m),
            np.maximum(ret_1_m, vol_12_m),
            ret_1_m,
            spread_m,
            depth_m,
            imb_m,
            funding_m,
            funding_m,
            basis_m,
            oi_m,
            oi_m,
            np.maximum(np.maximum(oi_m, spread_m), ret_1_m),
            np.maximum(ret_48_m, ret_288_m),
            t_cov_m,
            t_dis_m,
            e_count_m,
        ],
        axis=1,
    ).astype(np.uint8)
    return rf, rm


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _normalize_symbols(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def normalize_context_timeframes(raw: Any) -> List[str]:
    if raw is None:
        pieces = list(DEFAULT_CONTEXT_TIMEFRAMES)
    elif isinstance(raw, str):
        pieces = [x.strip() for x in raw.split(",")]
    else:
        pieces = [str(x).strip() for x in list(raw)]
    out: List[str] = []
    seen = set()
    for piece in pieces:
        tf = str(piece or "").strip().lower()
        if not tf or tf in seen:
            continue
        if tf.endswith("m") and tf[:-1].isdigit():
            pass
        elif tf.endswith("h") and tf[:-1].isdigit():
            pass
        elif tf.endswith("d") and tf[:-1].isdigit():
            pass
        else:
            continue
        seen.add(tf)
        out.append(tf)
    return out


def build_multi_tf_feature_names(timeframes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tf in normalize_context_timeframes(timeframes):
        key = tf.replace(" ", "")
        for field in MULTI_TF_CONTEXT_FIELDS:
            out.append(f"mtf_{key}_{field}")
    return out


def build_multi_tf_vector_from_payload(
    context_payload: Mapping[str, Any],
    *,
    timeframes: Sequence[str],
    require_complete: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    tfs = normalize_context_timeframes(timeframes)
    if not tfs:
        raise RuntimeError("multi_tf_context_timeframes_empty")
    ctx = dict(context_payload.get("context") or {})
    cov = dict(context_payload.get("coverage") or {})
    values: List[float] = []
    masks: List[int] = []
    for tf in tfs:
        block = dict(ctx.get(tf) or {})
        cov_block = dict(cov.get(tf) or {})
        if not block:
            if require_complete:
                raise RuntimeError(f"multi_tf_context_missing_block:{tf}")
            values.extend([1.0, 0.0, 0.0, 0.0])
            masks.extend([0, 1, 1, 1])
            continue
        miss = int(block.get("missing", cov_block.get("missing", 1)) or 0)
        ret_1 = float(block.get("ret_1", 0.0) or 0.0)
        volume = float(block.get("volume", 0.0) or 0.0)
        lag_sec = float(block.get("lag_sec", cov_block.get("asof_lag_sec", 0.0)) or 0.0)
        values.extend([float(miss), ret_1, volume, lag_sec])
        masks.extend([0, miss, miss, miss])
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=np.uint8)


def _load_universe_symbols(universe_snapshot_file: Path) -> Tuple[List[str], str]:
    if not universe_snapshot_file.exists():
        raise RuntimeError(f"universe_snapshot_missing:{universe_snapshot_file}")
    payload = json.loads(universe_snapshot_file.read_text(encoding="utf-8"))
    syms = _normalize_symbols(payload.get("symbols") or [])
    if not syms:
        raise RuntimeError("universe_snapshot_symbols_empty")
    snap_hash = str(payload.get("snapshot_hash") or "")
    if not snap_hash:
        raise RuntimeError("universe_snapshot_hash_missing")
    return syms, snap_hash


def _table_columns(conn: Any, table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            """,
            (str(table_name),),
        )
        rows = cur.fetchall() or []
    return {str(r.get("column_name") or "") for r in rows if str(r.get("column_name") or "").strip()}


def _pick_col(cols: set[str], names: Iterable[str]) -> str:
    for name in names:
        key = str(name)
        if key in cols:
            return key
    return ""


def _feature_matrix_layout(conn: Any) -> Dict[str, str]:
    cols = _table_columns(conn, "feature_matrix_main")
    values_col = _pick_col(cols, ("values", "feature_values"))
    mask_col = _pick_col(cols, ("mask", "feature_mask"))
    features_col = _pick_col(cols, ("features", "feature_payload"))
    schema_col = _pick_col(cols, ("schema_hash",))
    feature_version_col = _pick_col(cols, ("feature_version",))
    mode = ""
    if values_col and mask_col:
        mode = "vector"
    elif features_col:
        mode = "object"
    else:
        raise RuntimeError("feature_matrix_vector_columns_missing")
    return {
        "mode": mode,
        "values_col": values_col,
        "mask_col": mask_col,
        "features_col": features_col,
        "schema_col": schema_col,
        "feature_version_col": feature_version_col,
    }


def _read_feature_rows(
    *,
    conn: Any,
    symbol: str,
    start_ts: datetime,
    end_ts: datetime,
    layout: Mapping[str, str],
) -> List[Dict[str, Any]]:
    mode = str(layout.get("mode") or "")
    values_col = str(layout.get("values_col") or "")
    mask_col = str(layout.get("mask_col") or "")
    features_col = str(layout.get("features_col") or "")
    schema_col = str(layout.get("schema_col") or "")
    ver_col = str(layout.get("feature_version_col") or "")
    if mode == "vector" and values_col and mask_col:
        schema_sel = f"{schema_col} AS schema_hash" if schema_col else "NULL::text AS schema_hash"
        ver_sel = f"{ver_col} AS feature_version" if ver_col else "NULL::text AS feature_version"
        sql = f"""
            SELECT as_of_ts, {values_col} AS values, {mask_col} AS mask, {schema_sel}, {ver_sel}
            FROM feature_matrix_main
            WHERE UPPER(symbol) = %s
              AND as_of_ts >= %s
              AND as_of_ts <= %s
            ORDER BY as_of_ts ASC
        """
    elif mode == "object" and features_col:
        schema_sel = f"{schema_col} AS schema_hash" if schema_col else "NULL::text AS schema_hash"
        ver_sel = f"{ver_col} AS feature_version" if ver_col else "NULL::text AS feature_version"
        sql = f"""
            SELECT as_of_ts, {features_col} AS features_obj, {schema_sel}, {ver_sel}
            FROM feature_matrix_main
            WHERE symbol = %s
              AND as_of_ts >= %s
              AND as_of_ts <= %s
            ORDER BY as_of_ts ASC
        """
    else:
        raise RuntimeError("feature_matrix_layout_invalid")
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start_ts, end_ts))
        return [dict(r) for r in cur.fetchall()]


def _read_price_rows(
    *,
    conn: Any,
    symbol: str,
    timeframe: str,
    start_ts: datetime,
    end_ts: datetime,
) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, close::double precision AS close
            FROM market_bars
            WHERE UPPER(symbol) = %s
              AND timeframe = %s
              AND ts >= %s
              AND ts <= %s
            ORDER BY ts ASC
            """,
            (symbol, timeframe, start_ts, end_ts),
        )
        return [dict(r) for r in cur.fetchall()]


def _load_existing_symbol_cache(sym_file: Path) -> Dict[str, np.ndarray]:
    if not sym_file.exists():
        return {}
    try:
        with np.load(sym_file, allow_pickle=False) as obj:
            required = ("values", "mask", "close", "end_ts")
            for k in required:
                if k not in obj:
                    return {}
            out = {
                "values": np.asarray(obj["values"], dtype=np.float32),
                "mask": np.asarray(obj["mask"], dtype=np.uint8),
                "close": np.asarray(obj["close"], dtype=np.float64),
                "end_ts": np.asarray(obj["end_ts"], dtype=np.int64),
            }
            if "regime_features" in obj:
                out["regime_features"] = np.asarray(obj["regime_features"], dtype=np.float32)
            if "regime_mask" in obj:
                out["regime_mask"] = np.asarray(obj["regime_mask"], dtype=np.uint8)
            if "multi_tf_context" in obj:
                out["multi_tf_context"] = np.asarray(obj["multi_tf_context"], dtype=np.float32)
            if "multi_tf_mask" in obj:
                out["multi_tf_mask"] = np.asarray(obj["multi_tf_mask"], dtype=np.uint8)
            return out
    except Exception:
        return {}


def write_toy_cache(
    *,
    output_dir: Path,
    symbols: Sequence[str],
    values_map: Mapping[str, np.ndarray],
    mask_map: Mapping[str, np.ndarray],
    close_map: Mapping[str, np.ndarray],
    end_ts_map: Mapping[str, np.ndarray],
    lookback_len: int,
    horizons: Sequence[str],
    universe_snapshot_hash: str,
    feature_contract_hash: str = SCHEMA_HASH,
    bar_size: str = "5m",
    context_timeframes: Sequence[str] = DEFAULT_CONTEXT_TIMEFRAMES,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols_norm = _normalize_symbols(symbols)
    symbol_to_id = {s: i for i, s in enumerate(symbols_norm)}
    index_symbol: List[int] = []
    index_t_idx: List[int] = []
    index_end_ts: List[int] = []
    horizon_steps = {h: max(1, int(round(_parse_horizon_seconds(h) / _parse_bar_seconds(bar_size)))) for h in horizons}
    max_h = max(horizon_steps.values())
    context_names = build_multi_tf_feature_names(context_timeframes)
    context_dim = len(context_names)
    file_hashes: Dict[str, str] = {}
    for sym in symbols_norm:
        values = np.asarray(values_map[sym], dtype=np.float32)
        mask = np.asarray(mask_map[sym], dtype=np.uint8)
        close = np.asarray(close_map[sym], dtype=np.float64)
        end_ts = np.asarray(end_ts_map[sym], dtype=np.int64)
        if values.ndim != 2 or mask.shape != values.shape:
            raise RuntimeError(f"toy_cache_shape_invalid:{sym}")
        regime_features, regime_mask = compute_regime_features(values, mask)
        mtf_context = np.zeros((values.shape[0], context_dim), dtype=np.float32)
        mtf_mask = np.zeros((values.shape[0], context_dim), dtype=np.uint8)
        sym_file = output_dir / f"{sym}.npz"
        np.savez_compressed(
            sym_file,
            values=values,
            mask=mask,
            close=close,
            end_ts=end_ts,
            regime_features=regime_features,
            regime_mask=regime_mask,
            multi_tf_context=mtf_context,
            multi_tf_mask=mtf_mask,
        )
        file_hashes[sym_file.name] = _hash_file(sym_file)
        for t_idx in range(max(lookback_len - 1, 0), values.shape[0] - max_h):
            index_symbol.append(symbol_to_id[sym])
            index_t_idx.append(t_idx)
            index_end_ts.append(int(end_ts[t_idx]))

    idx_file = output_dir / "index.npz"
    np.savez_compressed(
        idx_file,
        symbol_id=np.asarray(index_symbol, dtype=np.int32),
        t_idx=np.asarray(index_t_idx, dtype=np.int32),
        end_ts=np.asarray(index_end_ts, dtype=np.int64),
    )
    file_hashes[idx_file.name] = _hash_file(idx_file)
    manifest = {
        "status": "ok",
        "bar_size": bar_size,
        "lookback_len": int(lookback_len),
        "horizons": [str(x) for x in horizons],
        "horizon_steps": {k: int(v) for k, v in horizon_steps.items()},
        "feature_contract_hash": str(feature_contract_hash),
        "universe_snapshot_hash": str(universe_snapshot_hash),
        "symbols": symbols_norm,
        "symbol_to_id": symbol_to_id,
        "regime_feature_names": list(REGIME_FEATURE_NAMES),
        "context_timeframes": normalize_context_timeframes(context_timeframes),
        "multi_tf_feature_names": context_names,
        "index_file": idx_file.name,
        "files": file_hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "data_audit.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_count": len(index_symbol),
                "missing_ratio": float(
                    np.mean([np.mean(np.asarray(mask_map[s], dtype=np.uint8)) for s in symbols_norm]) if symbols_norm else 1.0
                ),
                "asof_leakage_check": {"passed": True, "violations": 0},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_training_cache_from_db(cfg: CacheBuildConfig) -> Dict[str, Any]:
    symbols, universe_snapshot_hash = _load_universe_symbols(cfg.universe_snapshot_file)
    bar_sec = _parse_bar_seconds(cfg.bar_size)
    horizon_steps = {h: max(1, int(round(_parse_horizon_seconds(h) / bar_sec))) for h in cfg.horizons}
    max_h = max(horizon_steps.values())
    start_needed = cfg.start_ts
    end_needed = cfg.end_ts
    context_timeframes = normalize_context_timeframes(cfg.context_timeframes)
    if not context_timeframes:
        raise RuntimeError("context_timeframes_empty")
    context_feature_names = build_multi_tf_feature_names(context_timeframes)
    context_dim = len(context_feature_names)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sample_symbol_id: List[int] = []
    sample_t_idx: List[int] = []
    sample_end_ts: List[int] = []
    per_symbol_coverage: Dict[str, Dict[str, float]] = {}
    file_hashes: Dict[str, str] = {}
    leakage_violations = 0

    with psycopg2.connect(cfg.database_url, cursor_factory=RealDictCursor) as conn:
        layout = _feature_matrix_layout(conn)
        tables = _table_columns(conn, "market_context_multi_tf")
        has_multi_tf_table = bool(tables)
        if bool(cfg.require_multi_tf_context) and not has_multi_tf_table:
            raise RuntimeError("missing_table:market_context_multi_tf")
        for symbol in symbols:
            sym_file = out_dir / f"{symbol}.npz"
            existing = _load_existing_symbol_cache(sym_file) if bool(cfg.incremental) else {}
            req_start_ts = int(cfg.start_ts.timestamp())
            req_end_ts = int(cfg.end_ts.timestamp())
            query_start_ts = req_start_ts
            reuse_existing_only = False
            incremental_mode = "rebuild"
            if existing:
                existing_end_ts = int(existing["end_ts"][-1]) if existing["end_ts"].size else 0
                existing_start_ts = int(existing["end_ts"][0]) if existing["end_ts"].size else 0
                if existing_end_ts >= req_end_ts and existing_start_ts <= req_start_ts:
                    reuse_existing_only = True
                    incremental_mode = "reuse_existing"
                elif existing_end_ts >= req_start_ts and existing_start_ts <= req_start_ts:
                    warmup_steps = max(1, int(cfg.incremental_warmup_steps))
                    query_start_ts = max(req_start_ts, existing_end_ts - (warmup_steps * bar_sec))
                    incremental_mode = "append_from_overlap"
                if bool(cfg.require_multi_tf_context) and (
                    "multi_tf_context" not in existing or "multi_tf_mask" not in existing
                ):
                    reuse_existing_only = False
                    incremental_mode = "rebuild_missing_multi_tf"

            feat_rows: List[Dict[str, Any]] = []
            px_rows: List[Dict[str, Any]] = []
            mtf_by_ts: Dict[int, Dict[str, Any]] = {}
            if not reuse_existing_only:
                query_start_dt = datetime.fromtimestamp(query_start_ts, tz=timezone.utc)
                feat_rows = _read_feature_rows(
                    conn=conn,
                    symbol=symbol,
                    start_ts=query_start_dt,
                    end_ts=end_needed,
                    layout=layout,
                )
                px_rows = _read_price_rows(conn=conn, symbol=symbol, timeframe=cfg.bar_size, start_ts=query_start_dt, end_ts=end_needed)
                if not feat_rows:
                    raise RuntimeError(f"feature_rows_missing:{symbol}")
                if not px_rows:
                    raise RuntimeError(f"price_rows_missing:{symbol}:{cfg.bar_size}")
                if has_multi_tf_table:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT as_of_ts, context_json, coverage_json
                            FROM market_context_multi_tf
                            WHERE UPPER(symbol) = %s
                              AND primary_timeframe = %s
                              AND as_of_ts >= %s
                              AND as_of_ts <= %s
                            ORDER BY as_of_ts ASC
                            """,
                            (symbol, str(cfg.bar_size).lower(), query_start_dt, end_needed),
                        )
                        for row in cur.fetchall() or []:
                            ts_key = int(_parse_ts(row.get("as_of_ts")).timestamp())
                            mtf_by_ts[ts_key] = {
                                "context": dict(row.get("context_json") or {}),
                                "coverage": dict(row.get("coverage_json") or {}),
                            }
                if bool(cfg.require_multi_tf_context) and not mtf_by_ts:
                    raise RuntimeError(f"multi_tf_context_rows_missing:{symbol}:{cfg.bar_size}")

            px_map: Dict[int, float] = {}
            for r in px_rows:
                ts = int(_parse_ts(r["ts"]).timestamp())
                px_map[ts] = float(r.get("close") or 0.0)

            values_rows: List[np.ndarray] = []
            mask_rows: List[np.ndarray] = []
            end_ts_rows: List[int] = []
            close_rows: List[float] = []
            mtf_rows: List[np.ndarray] = []
            mtf_mask_rows: List[np.ndarray] = []
            mode = str(layout.get("mode") or "")
            min_cov = float(os.getenv("CACHE_MIN_FEATURE_KEY_COVERAGE", "0.60"))
            if not reuse_existing_only:
                for r in feat_rows:
                    schema_hash = str(r.get("schema_hash") or "")
                    feature_version = str(r.get("feature_version") or "")
                    if schema_hash and schema_hash != str(SCHEMA_HASH):
                        raise RuntimeError(f"schema_hash_mismatch_cache:{symbol}")
                    if not schema_hash and feature_version and feature_version != "main":
                        raise RuntimeError(f"feature_version_mismatch_cache:{symbol}:{feature_version}")
                    if mode == "vector":
                        vals = np.asarray(r.get("values") or [], dtype=np.float32).reshape(-1)
                        msk = np.asarray(r.get("mask") or [], dtype=np.uint8).reshape(-1)
                    else:
                        obj = r.get("features_obj")
                        if not isinstance(obj, dict):
                            raise RuntimeError(f"feature_object_missing:{symbol}")
                        vals = np.zeros((FEATURE_DIM,), dtype=np.float32)
                        msk = np.ones((FEATURE_DIM,), dtype=np.uint8)
                        matched = 0
                        for k, v in obj.items():
                            idx = FEATURE_INDEX.get(str(k))
                            if idx is None:
                                continue
                            vals[int(idx)] = float(v or 0.0)
                            msk[int(idx)] = 0
                            matched += 1
                        coverage = float(matched / max(1, FEATURE_DIM))
                        if coverage < min_cov:
                            raise RuntimeError(f"feature_object_coverage_low:{symbol}:{coverage:.4f}")
                    if vals.size != FEATURE_DIM or msk.size != FEATURE_DIM:
                        raise RuntimeError(f"feature_dim_mismatch_cache:{symbol}:{vals.size}:{msk.size}:{FEATURE_DIM}")
                    ts = int(_parse_ts(r["as_of_ts"]).timestamp())
                    px = float(px_map.get(ts, 0.0))
                    if px <= 0:
                        continue
                    ctx_payload = mtf_by_ts.get(ts)
                    if not isinstance(ctx_payload, dict):
                        if bool(cfg.require_multi_tf_context):
                            raise RuntimeError(f"multi_tf_context_missing:{symbol}:{ts}")
                        ctx_payload = {"context": {}, "coverage": {}}
                    mtf_vec, mtf_msk = build_multi_tf_vector_from_payload(
                        ctx_payload,
                        timeframes=context_timeframes,
                        require_complete=bool(cfg.require_multi_tf_context),
                    )
                    values_rows.append(vals)
                    mask_rows.append(msk)
                    end_ts_rows.append(ts)
                    close_rows.append(px)
                    mtf_rows.append(mtf_vec)
                    mtf_mask_rows.append(mtf_msk)

            if reuse_existing_only:
                values = np.asarray(existing["values"], dtype=np.float32)
                mask = np.asarray(existing["mask"], dtype=np.uint8)
                close = np.asarray(existing["close"], dtype=np.float64)
                end_ts_np = np.asarray(existing["end_ts"], dtype=np.int64)
                if "multi_tf_context" in existing and "multi_tf_mask" in existing:
                    multi_tf_context = np.asarray(existing["multi_tf_context"], dtype=np.float32)
                    multi_tf_mask = np.asarray(existing["multi_tf_mask"], dtype=np.uint8)
                elif bool(cfg.require_multi_tf_context):
                    raise RuntimeError(f"existing_cache_missing_multi_tf_context:{symbol}:{sym_file}")
                else:
                    multi_tf_context = np.zeros((values.shape[0], context_dim), dtype=np.float32)
                    multi_tf_mask = np.ones((values.shape[0], context_dim), dtype=np.uint8)
            else:
                if len(values_rows) <= max(cfg.lookback_len + max_h, 16):
                    raise RuntimeError(f"insufficient_symbol_rows_for_cache:{symbol}:{len(values_rows)}")
                fresh_values = np.stack(values_rows, axis=0).astype(np.float32)
                fresh_mask = np.stack(mask_rows, axis=0).astype(np.uint8)
                fresh_close = np.asarray(close_rows, dtype=np.float64)
                fresh_end_ts = np.asarray(end_ts_rows, dtype=np.int64)
                fresh_multi_tf = np.stack(mtf_rows, axis=0).astype(np.float32)
                fresh_multi_tf_mask = np.stack(mtf_mask_rows, axis=0).astype(np.uint8)
                if existing and incremental_mode == "append_from_overlap":
                    old_values = np.asarray(existing["values"], dtype=np.float32)
                    old_mask = np.asarray(existing["mask"], dtype=np.uint8)
                    old_close = np.asarray(existing["close"], dtype=np.float64)
                    old_end_ts = np.asarray(existing["end_ts"], dtype=np.int64)
                    if "multi_tf_context" in existing and "multi_tf_mask" in existing:
                        old_multi_tf = np.asarray(existing["multi_tf_context"], dtype=np.float32)
                        old_multi_tf_mask = np.asarray(existing["multi_tf_mask"], dtype=np.uint8)
                    elif bool(cfg.require_multi_tf_context):
                        raise RuntimeError(f"existing_cache_missing_multi_tf_context:{symbol}:{sym_file}")
                    else:
                        old_multi_tf = np.zeros((old_values.shape[0], context_dim), dtype=np.float32)
                        old_multi_tf_mask = np.ones((old_values.shape[0], context_dim), dtype=np.uint8)
                    merged: Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]] = {}
                    for i, ts in enumerate(old_end_ts.tolist()):
                        merged[int(ts)] = (
                            old_values[i],
                            old_mask[i],
                            float(old_close[i]),
                            old_multi_tf[i],
                            old_multi_tf_mask[i],
                        )
                    for i, ts in enumerate(fresh_end_ts.tolist()):
                        merged[int(ts)] = (
                            fresh_values[i],
                            fresh_mask[i],
                            float(fresh_close[i]),
                            fresh_multi_tf[i],
                            fresh_multi_tf_mask[i],
                        )
                    ts_sorted = sorted(merged.keys())
                    values = np.stack([merged[ts][0] for ts in ts_sorted], axis=0).astype(np.float32)
                    mask = np.stack([merged[ts][1] for ts in ts_sorted], axis=0).astype(np.uint8)
                    close = np.asarray([merged[ts][2] for ts in ts_sorted], dtype=np.float64)
                    multi_tf_context = np.stack([merged[ts][3] for ts in ts_sorted], axis=0).astype(np.float32)
                    multi_tf_mask = np.stack([merged[ts][4] for ts in ts_sorted], axis=0).astype(np.uint8)
                    end_ts_np = np.asarray(ts_sorted, dtype=np.int64)
                else:
                    values = fresh_values
                    mask = fresh_mask
                    close = fresh_close
                    end_ts_np = fresh_end_ts
                    multi_tf_context = fresh_multi_tf
                    multi_tf_mask = fresh_multi_tf_mask
            if values.shape[0] <= max(cfg.lookback_len + max_h, 16):
                raise RuntimeError(f"insufficient_symbol_rows_for_cache:{symbol}:{values.shape[0]}")
            if values.shape != mask.shape:
                raise RuntimeError(f"cache_shape_mismatch_after_merge:{symbol}")
            if multi_tf_context.shape != (values.shape[0], context_dim):
                raise RuntimeError(
                    f"multi_tf_context_shape_mismatch:{symbol}:{tuple(multi_tf_context.shape)}:{(values.shape[0], context_dim)}"
                )
            if multi_tf_mask.shape != (values.shape[0], context_dim):
                raise RuntimeError(
                    f"multi_tf_mask_shape_mismatch:{symbol}:{tuple(multi_tf_mask.shape)}:{(values.shape[0], context_dim)}"
                )
            regime_features, regime_mask = compute_regime_features(values, mask)

            np.savez_compressed(
                sym_file,
                values=values,
                mask=mask,
                close=close,
                end_ts=end_ts_np,
                regime_features=regime_features,
                regime_mask=regime_mask,
                multi_tf_context=multi_tf_context,
                multi_tf_mask=multi_tf_mask,
            )
            file_hashes[sym_file.name] = _hash_file(sym_file)

            missing_ratio = float(mask.mean()) if mask.size else 1.0
            observed_ratio = float(1.0 - missing_ratio)
            per_symbol_coverage[symbol] = {
                "rows": int(values.shape[0]),
                "missing_ratio": missing_ratio,
                "observed_ratio": observed_ratio,
                "multi_tf_missing_ratio": float(multi_tf_mask.mean()) if multi_tf_mask.size else 1.0,
                "incremental_mode": incremental_mode,
            }

            for t_idx in range(max(cfg.lookback_len - 1, 0), values.shape[0] - max_h):
                ts = int(end_ts_np[t_idx])
                ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                if ts_dt < cfg.start_ts or ts_dt > cfg.end_ts:
                    continue
                for h, step in horizon_steps.items():
                    if (t_idx + step) <= t_idx:
                        leakage_violations += 1
                        raise RuntimeError(f"asof_leakage_detected:{symbol}:{h}:{t_idx}:{step}")
                    if (t_idx + step) >= values.shape[0]:
                        leakage_violations += 1
                        raise RuntimeError(f"future_index_missing:{symbol}:{h}:{t_idx}:{step}")
                sample_symbol_id.append(symbol_to_id[symbol])
                sample_t_idx.append(int(t_idx))
                sample_end_ts.append(int(ts))

    if not sample_t_idx:
        raise RuntimeError("cache_sample_index_empty")

    idx_file = out_dir / "index.npz"
    np.savez_compressed(
        idx_file,
        symbol_id=np.asarray(sample_symbol_id, dtype=np.int32),
        t_idx=np.asarray(sample_t_idx, dtype=np.int32),
        end_ts=np.asarray(sample_end_ts, dtype=np.int64),
    )
    file_hashes[idx_file.name] = _hash_file(idx_file)

    manifest = {
        "status": "ok",
        "bar_size": str(cfg.bar_size),
        "lookback_len": int(cfg.lookback_len),
        "horizons": [str(x) for x in cfg.horizons],
        "horizon_steps": {k: int(v) for k, v in horizon_steps.items()},
        "incremental": bool(cfg.incremental),
        "incremental_warmup_steps": int(cfg.incremental_warmup_steps),
        "feature_contract_hash": str(cfg.feature_contract_hash),
        "universe_snapshot_hash": str(universe_snapshot_hash),
        "universe_snapshot_file": str(cfg.universe_snapshot_file),
        "symbols": symbols,
        "symbol_to_id": symbol_to_id,
        "regime_feature_names": list(REGIME_FEATURE_NAMES),
        "context_timeframes": context_timeframes,
        "multi_tf_feature_names": context_feature_names,
        "index_file": idx_file.name,
        "files": file_hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest["cache_hash"] = manifest_hash
    manifest_path = out_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = {
        "status": "ok",
        "sample_count": int(len(sample_t_idx)),
        "symbol_count": int(len(symbols)),
        "feature_contract_hash": str(cfg.feature_contract_hash),
        "incremental": bool(cfg.incremental),
        "universe_snapshot_hash": str(universe_snapshot_hash),
        "coverage": {
            "per_symbol": per_symbol_coverage,
            "avg_missing_ratio": float(np.mean([v["missing_ratio"] for v in per_symbol_coverage.values()])),
            "avg_observed_ratio": float(np.mean([v["observed_ratio"] for v in per_symbol_coverage.values()])),
        },
        "asof_leakage_check": {
            "passed": leakage_violations == 0,
            "violations": int(leakage_violations),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
