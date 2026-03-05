from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class MultiHorizonDistOutput:
    """Unified strict output contract for liquid model inference."""

    mu: torch.Tensor
    log_sigma: torch.Tensor
    q: Optional[torch.Tensor] = None
    direction_logit: Optional[torch.Tensor] = None
    expert_weights: Optional[torch.Tensor] = None
    regime_probs: Optional[torch.Tensor] = None
    df: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = field(default_factory=dict)

    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    @property
    def quantiles(self) -> Optional[torch.Tensor]:
        return self.q

    @property
    def uncertainty(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def direction_prob(self) -> Optional[torch.Tensor]:
        if self.direction_logit is None:
            return None
        return torch.sigmoid(self.direction_logit)
