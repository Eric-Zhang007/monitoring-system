from __future__ import annotations

from typing import Dict, Optional


def compute_derivatives(
    *,
    funding_rate: Optional[float],
    basis_rate: Optional[float],
    taker_buy_sell_ratio: Optional[float],
    open_interest: Optional[float],
) -> Dict[str, float]:
    return {
        "funding_rate": float(funding_rate or 0.0),
        "basis_rate": float(basis_rate or 0.0),
        "taker_buy_sell_ratio": float(taker_buy_sell_ratio or 0.0),
        "open_interest": float(open_interest or 0.0),
    }
