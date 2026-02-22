from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BackboneBase(nn.Module, ABC):
    @property
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Return sequence representation with shape [B, L, D_model]."""
        raise NotImplementedError
