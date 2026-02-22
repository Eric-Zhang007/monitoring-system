from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ExecutionAdapterBase(ABC):
    name: str = "base"

    @abstractmethod
    def prepare(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, child_order: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def poll_order(self, venue_order_id: str, timeout: float) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, venue_order_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fetch_fills(self, venue_order_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def fetch_positions(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
