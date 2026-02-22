from __future__ import annotations

import pytest

from execution_fsm import transition_child, transition_decision, transition_parent


def test_decision_fsm_paths():
    s = transition_decision("created", "start")
    assert s == "running"
    s = transition_decision(s, "complete")
    assert s == "completed"

    with pytest.raises(ValueError):
        transition_decision("completed", "start")


def test_parent_fsm_paths():
    s = transition_parent("submitted", "start_exec")
    assert s == "executing"
    s = transition_parent(s, "partial_fill")
    assert s == "partially_filled"
    s = transition_parent(s, "fill")
    assert s == "filled"

    with pytest.raises(ValueError):
        transition_parent("submitted", "fill")


def test_child_fsm_paths():
    s = transition_child("new", "submit")
    assert s == "submitted"
    s = transition_child(s, "partial_fill")
    assert s == "partially_filled"
    s = transition_child(s, "cancel")
    assert s == "canceled"

    with pytest.raises(ValueError):
        transition_child("filled", "cancel")
