from __future__ import annotations

from typing import Any, Dict, List

from backend.mailer import send_risk_email


class _Repo:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def save_mail_delivery_log(self, **kwargs):
        self.logs.append(dict(kwargs))
        return len(self.logs)


class _SMTP:
    def __init__(self, host=None, port=None, timeout=None):  # noqa: ANN001
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self):
        return None

    def login(self, user, password):  # noqa: ANN001
        _ = (user, password)
        return None

    def send_message(self, msg):
        self.sent.append(msg)


def test_email_risk_notification_success(monkeypatch):
    repo = _Repo()
    monkeypatch.setenv("RISK_EMAIL_RECIPIENTS", "a@test.local,b@test.local")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.local")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setenv("SMTP_FROM", "monitor@test.local")
    monkeypatch.setattr("backend.mailer.smtplib.SMTP", _SMTP)

    out = send_risk_email(
        event_type="kill_switch_trigger",
        subject="[risk] trigger",
        body="body",
        repo=repo,  # type: ignore[arg-type]
    )
    assert out["send_ok"] is True
    assert len(repo.logs) == 1
    assert repo.logs[0]["send_ok"] is True


def test_email_risk_notification_failure_logged(monkeypatch):
    repo = _Repo()
    monkeypatch.setenv("RISK_EMAIL_RECIPIENTS", "a@test.local")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    out = send_risk_email(
        event_type="recon_drift",
        subject="[risk] recon",
        body="body",
        repo=repo,  # type: ignore[arg-type]
    )
    assert out["send_ok"] is False
    assert len(repo.logs) == 1
    assert repo.logs[0]["send_ok"] is False
