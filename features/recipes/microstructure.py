from __future__ import annotations

from typing import Dict, Optional


def compute_microstructure(
    *,
    bid_px: Optional[float],
    ask_px: Optional[float],
    bid_sz: Optional[float],
    ask_sz: Optional[float],
    spread_bps: Optional[float],
    imbalance: Optional[float],
) -> Dict[str, float]:
    bp = float(bid_px) if bid_px is not None else 0.0
    ap = float(ask_px) if ask_px is not None else 0.0
    bs = float(bid_sz) if bid_sz is not None else 0.0
    a_s = float(ask_sz) if ask_sz is not None else 0.0
    if spread_bps is None and bp > 0 and ap > 0:
        mid = max(1e-9, (bp + ap) * 0.5)
        spr = (ap - bp) / mid * 10000.0
    else:
        spr = float(spread_bps or 0.0)
    if imbalance is None:
        imb = (bs - a_s) / max(1e-9, (bs + a_s)) if (bs + a_s) > 0 else 0.0
    else:
        imb = float(imbalance)
    return {
        "orderbook_spread_bps": float(spr),
        "orderbook_imbalance": float(imb),
        "orderbook_depth_total": float(max(0.0, bs + a_s)),
    }
