from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseConnector(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self) -> List[Dict]:
        """Fetch raw events from a data source."""

    @abstractmethod
    def normalize(self, raw: Dict) -> Dict:
        """Normalize a raw event to canonical schema."""
