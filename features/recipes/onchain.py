from __future__ import annotations

from typing import Dict, Optional


def compute_onchain(
    *,
    net_inflow: Optional[float],
    active_addresses: Optional[float],
) -> Dict[str, float]:
    return {
        "onchain_net_inflow": float(net_inflow or 0.0),
        "onchain_active_addresses": float(active_addresses or 0.0),
    }
