from __future__ import annotations

from typing import Literal

DecisionState = Literal["created", "running", "completed", "failed"]
ParentState = Literal["submitted", "executing", "filled", "partially_filled", "rejected", "canceled"]
ChildState = Literal["new", "submitted", "filled", "partially_filled", "canceled", "rejected", "expired"]


_DECISION_TRANSITIONS = {
    "created": {"start": "running", "fail": "failed"},
    "running": {"complete": "completed", "fail": "failed"},
    "completed": {},
    "failed": {},
}

_PARENT_TRANSITIONS = {
    "submitted": {"start_exec": "executing", "reject": "rejected", "cancel": "canceled"},
    "executing": {
        "fill": "filled",
        "partial_fill": "partially_filled",
        "reject": "rejected",
        "cancel": "canceled",
    },
    "partially_filled": {"fill": "filled", "cancel": "canceled", "reject": "rejected"},
    "filled": {},
    "rejected": {},
    "canceled": {},
}

_CHILD_TRANSITIONS = {
    "new": {"submit": "submitted", "reject": "rejected", "expire": "expired"},
    "submitted": {
        "fill": "filled",
        "partial_fill": "partially_filled",
        "cancel": "canceled",
        "reject": "rejected",
        "expire": "expired",
    },
    "partially_filled": {"fill": "filled", "cancel": "canceled", "reject": "rejected", "expire": "expired"},
    "filled": {},
    "canceled": {},
    "rejected": {},
    "expired": {},
}


def _transition(table: dict, prev: str, event: str, state_name: str) -> str:
    if prev not in table:
        raise ValueError(f"invalid_{state_name}_state:{prev}")
    next_state = table[prev].get(event)
    if not next_state:
        raise ValueError(f"illegal_{state_name}_transition:{prev}:{event}")
    return str(next_state)


def transition_decision(prev: DecisionState, event: str) -> DecisionState:
    return transition(prev, event, kind="decision")  # type: ignore[return-value]


def transition_parent(prev: ParentState, event: str) -> ParentState:
    return transition(prev, event, kind="parent")  # type: ignore[return-value]


def transition_child(prev: ChildState, event: str) -> ChildState:
    return transition(prev, event, kind="child")  # type: ignore[return-value]


def transition(prev: str, event: str, *, kind: Literal["decision", "parent", "child"]) -> str:
    if kind == "decision":
        return _transition(_DECISION_TRANSITIONS, prev, event, "decision")
    if kind == "parent":
        return _transition(_PARENT_TRANSITIONS, prev, event, "parent")
    if kind == "child":
        return _transition(_CHILD_TRANSITIONS, prev, event, "child")
    raise ValueError(f"unknown_fsm_kind:{kind}")


__all__ = ["transition", "transition_decision", "transition_parent", "transition_child"]
