from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main as main_mod  # noqa: E402
from metrics import WEBSOCKET_DROPPED_MESSAGES_TOTAL  # noqa: E402


class _FakeWebSocket:
    def __init__(self, fail_send: bool = False):
        self.fail_send = fail_send
        self.accepted = False
        self.closed = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(payload)

    async def close(self, code: int = 1000):
        self.closed = True


def test_queue_full_disconnects_and_increments_counter():
    async def _case():
        mgr = main_mod.ConnectionManager()
        mgr.queue_max = 1
        ws = _FakeWebSocket()
        await mgr.connect(ws, "signals")
        # Keep queue full deterministically.
        mgr.sender_tasks[ws].cancel()
        mgr.queues[ws].put_nowait({"type": "seed"})
        before = WEBSOCKET_DROPPED_MESSAGES_TOTAL.labels(reason="queue_full")._value.get()
        mgr._enqueue(ws, {"type": "next"})
        await asyncio.sleep(0.01)
        after = WEBSOCKET_DROPPED_MESSAGES_TOTAL.labels(reason="queue_full")._value.get()
        assert after >= before + 1
        assert ws not in mgr.active_connections

    asyncio.run(_case())


def test_sender_error_disconnects_and_increments_counter():
    async def _case():
        mgr = main_mod.ConnectionManager()
        ws = _FakeWebSocket(fail_send=True)
        await mgr.connect(ws, "signals")
        before = WEBSOCKET_DROPPED_MESSAGES_TOTAL.labels(reason="send_error")._value.get()
        mgr._enqueue(ws, {"type": "evt"})
        for _ in range(20):
            if ws not in mgr.active_connections:
                break
            await asyncio.sleep(0.01)
        after = WEBSOCKET_DROPPED_MESSAGES_TOTAL.labels(reason="send_error")._value.get()
        assert after >= before + 1
        assert ws not in mgr.active_connections

    asyncio.run(_case())


def test_slow_connection_isolated_without_blocking_others():
    async def _case():
        mgr = main_mod.ConnectionManager()
        mgr.queue_max = 1
        mgr.flush_ms = 10
        ws_bad = _FakeWebSocket()
        ws_good = _FakeWebSocket()
        await mgr.connect(ws_bad, "signals")
        await mgr.connect(ws_good, "signals")
        mgr.subscribe(ws_bad, "BTC")
        mgr.subscribe(ws_good, "BTC")

        # Simulate a stuck/buffered client queue.
        mgr.sender_tasks[ws_bad].cancel()
        mgr.queues[ws_bad].put_nowait({"type": "seed"})
        await mgr.broadcast_symbol("BTC", {"type": "prediction", "symbol": "BTC"})
        await asyncio.sleep(0.2)

        assert ws_bad not in mgr.active_connections
        assert ws_good in mgr.active_connections
        assert "BTC" in mgr.subscriptions.get(ws_good, set())
        mgr.disconnect(ws_good, reason="test_cleanup")

    asyncio.run(_case())
