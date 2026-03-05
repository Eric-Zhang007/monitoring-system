from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from process_manager import ProcessManager
from runtime_config_service import RuntimeConfigService
from v2_repository import V2Repository


SUPPORTED_COMMANDS = [
    "CMD KILL_SWITCH ON track=<track> strategy=<strategy>",
    "CMD KILL_SWITCH OFF track=<track> strategy=<strategy>",
    "CMD PAUSE LIVE account=<account_id>",
    "CMD RESUME LIVE account=<account_id>",
    "CMD RESTART PROCESS id=<process_id>",
    "CMD SET CONFIG key=<config_key> value=<json-or-scalar> scope=<global|track|process|account> [scope_id=<id>] [requires_restart=0|1]",
]


@dataclass
class ParseResult:
    ok: bool
    action: str
    params: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "action": self.action, "params": self.params, "error": self.error}


def _kv_tokens(tokens: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def parse_risk_command(raw: str) -> ParseResult:
    text = str(raw or "").strip()
    if not text:
        return ParseResult(ok=False, action="invalid", params={}, error="empty_command")
    try:
        tokens = shlex.split(text)
    except Exception as exc:
        return ParseResult(ok=False, action="invalid", params={}, error=f"tokenize_failed:{exc}")
    if len(tokens) < 3 or tokens[0].upper() != "CMD":
        return ParseResult(ok=False, action="invalid", params={}, error="command_must_start_with_CMD")
    head = [t.upper() for t in tokens[:3]]
    extras = tokens[3:]
    kv = _kv_tokens(extras)

    if head[1] == "KILL_SWITCH" and head[2] in {"ON", "OFF"}:
        track = str(kv.get("track") or "").strip().lower()
        strategy = str(kv.get("strategy") or kv.get("strategy_id") or "").strip()
        if not track:
            return ParseResult(ok=False, action="kill_switch", params={}, error="missing_track")
        if not strategy:
            return ParseResult(ok=False, action="kill_switch", params={}, error="missing_strategy")
        return ParseResult(ok=True, action="kill_switch", params={"enabled": head[2] == "ON", "track": track, "strategy_id": strategy})

    if head[1] in {"PAUSE", "RESUME"} and head[2] == "LIVE":
        account_id = str(kv.get("account") or "").strip()
        if not account_id:
            return ParseResult(ok=False, action="live_control", params={}, error="missing_account")
        return ParseResult(ok=True, action="live_control", params={"enabled": head[1] == "RESUME", "account_id": account_id})

    if head[1] == "RESTART" and head[2] == "PROCESS":
        pid = str(kv.get("id") or "").strip()
        if not pid:
            return ParseResult(ok=False, action="restart_process", params={}, error="missing_process_id")
        return ParseResult(ok=True, action="restart_process", params={"process_id": pid})

    if head[1] == "SET" and head[2] == "CONFIG":
        key = str(kv.get("key") or "").strip()
        scope = str(kv.get("scope") or "global").strip().lower()
        scope_id = str(kv.get("scope_id") or "").strip()
        value_raw = kv.get("value")
        if not key:
            return ParseResult(ok=False, action="set_config", params={}, error="missing_key")
        if value_raw is None:
            return ParseResult(ok=False, action="set_config", params={}, error="missing_value")
        if scope not in {"global", "track", "process", "account"}:
            return ParseResult(ok=False, action="set_config", params={}, error="invalid_scope")
        try:
            if value_raw.startswith("{") or value_raw.startswith("["):
                value_json = json.loads(value_raw)
            elif value_raw.lower() in {"true", "false"}:
                value_json = {"value": value_raw.lower() == "true"}
            else:
                try:
                    value_json = {"value": float(value_raw)}
                except Exception:
                    value_json = {"value": value_raw}
        except Exception as exc:
            return ParseResult(ok=False, action="set_config", params={}, error=f"invalid_value_json:{exc}")
        req_restart = str(kv.get("requires_restart") or "0").strip().lower() in {"1", "true", "yes", "on"}
        return ParseResult(
            ok=True,
            action="set_config",
            params={"config_key": key, "value_json": value_json, "scope": scope, "scope_id": scope_id, "requires_restart": req_restart},
        )

    return ParseResult(ok=False, action="invalid", params={}, error="unsupported_command")


class RiskCommandExecutor:
    def __init__(self, repo: V2Repository, process_manager: ProcessManager, config_service: RuntimeConfigService):
        self.repo = repo
        self.process_manager = process_manager
        self.config_service = config_service

    def execute(self, *, source: str, command_text: str, actor: str = "system") -> Dict[str, Any]:
        parsed = parse_risk_command(command_text)
        if not parsed.ok:
            payload = {
                "parse_ok": False,
                "execute_ok": False,
                "error": parsed.error,
                "supported_commands": list(SUPPORTED_COMMANDS),
            }
            try:
                self.repo.save_risk_command_log(
                    source=source,
                    command_text=command_text,
                    parse_ok=False,
                    execute_ok=False,
                    result=payload,
                    error=parsed.error,
                    created_by=actor,
                )
            except Exception:
                pass
            return payload

        execute_ok = False
        result: Dict[str, Any] = {"action": parsed.action, "params": parsed.params}
        error: Optional[str] = None
        try:
            if parsed.action == "kill_switch":
                enabled = bool(parsed.params.get("enabled"))
                if enabled:
                    state = self.repo.upsert_kill_switch_state(
                        track=str(parsed.params["track"]),
                        strategy_id=str(parsed.params["strategy_id"]),
                        state="triggered",
                        reason=f"command:{source}",
                        metadata={"source": source, "actor": actor, "via": "risk_command"},
                    )
                else:
                    state = self.repo.upsert_kill_switch_state(
                        track=str(parsed.params["track"]),
                        strategy_id=str(parsed.params["strategy_id"]),
                        state="armed",
                        reason=f"command:{source}",
                        metadata={"source": source, "actor": actor, "via": "risk_command"},
                    )
                result["kill_switch_state"] = state
                execute_ok = True
            elif parsed.action == "live_control":
                account_id = str(parsed.params["account_id"])
                cfg = self.config_service.upsert_config(
                    config_key=f"live.account.{account_id}.enabled",
                    value_json={"enabled": bool(parsed.params.get("enabled"))},
                    scope="account",
                    scope_id=account_id,
                    requires_restart=False,
                    description="Live trader pause/resume command",
                    updated_by=actor,
                )
                result["config"] = cfg
                execute_ok = True
            elif parsed.action == "restart_process":
                out = self.process_manager.restart(str(parsed.params["process_id"]), created_by=actor)
                result["process"] = out
                execute_ok = True
            elif parsed.action == "set_config":
                cfg = self.config_service.upsert_config(
                    config_key=str(parsed.params["config_key"]),
                    value_json=dict(parsed.params["value_json"]),
                    scope=str(parsed.params["scope"]),
                    scope_id=str(parsed.params["scope_id"]),
                    requires_restart=bool(parsed.params["requires_restart"]),
                    description=f"updated by {source} command",
                    updated_by=actor,
                )
                result["config"] = cfg
                execute_ok = True
            else:
                error = f"unsupported_action:{parsed.action}"
        except Exception as exc:
            error = str(exc)

        payload = {
            "parse_ok": True,
            "execute_ok": bool(execute_ok),
            "result": result,
            "error": error,
            "supported_commands": list(SUPPORTED_COMMANDS),
        }
        try:
            self.repo.save_risk_command_log(
                source=source,
                command_text=command_text,
                parse_ok=True,
                execute_ok=bool(execute_ok),
                result=payload,
                error=error,
                created_by=actor,
            )
        except Exception:
            pass
        return payload
