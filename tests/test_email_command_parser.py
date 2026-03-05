from __future__ import annotations

from typing import Any, Dict, List

from backend.risk_command_center import RiskCommandExecutor, parse_risk_command


class _FakeRepo:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.kill_rows: List[Dict[str, Any]] = []

    def save_risk_command_log(self, **kwargs):
        self.logs.append(dict(kwargs))
        return len(self.logs)

    def upsert_kill_switch_state(self, **kwargs):
        row = dict(kwargs)
        self.kill_rows.append(row)
        return row


class _FakePM:
    def restart(self, process_id: str, created_by: str = "api"):
        return {"process_id": process_id, "status": "running", "created_by": created_by}


class _FakeConfig:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def upsert_config(self, **kwargs):
        row = dict(kwargs)
        row["version"] = len(self.items) + 1
        self.items.append(row)
        return row


def test_email_command_parser_success_and_failure():
    ok = parse_risk_command("CMD KILL_SWITCH ON track=liquid strategy=global")
    assert ok.ok is True
    assert ok.action == "kill_switch"
    assert ok.params["enabled"] is True

    bad = parse_risk_command("KILL_SWITCH ON")
    assert bad.ok is False
    assert "command_must_start_with_CMD" in str(bad.error)


def test_email_command_executor_branches():
    repo = _FakeRepo()
    executor = RiskCommandExecutor(repo=repo, process_manager=_FakePM(), config_service=_FakeConfig())  # type: ignore[arg-type]

    out1 = executor.execute(
        source="email",
        command_text="CMD KILL_SWITCH OFF track=liquid strategy=global",
        actor="tester",
    )
    assert out1["parse_ok"] is True
    assert out1["execute_ok"] is True
    assert len(repo.kill_rows) == 1

    out2 = executor.execute(
        source="frontend",
        command_text="CMD SET CONFIG key=paper.initial_usdt value=750 scope=global",
        actor="tester2",
    )
    assert out2["parse_ok"] is True
    assert out2["execute_ok"] is True

    out3 = executor.execute(
        source="email",
        command_text="CMD PAUSE LIVE",
        actor="tester3",
    )
    assert out3["parse_ok"] is False
    assert out3["execute_ok"] is False
    assert len(repo.logs) >= 3
