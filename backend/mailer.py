from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

from v2_repository import V2Repository


def _recipients_from_env() -> List[str]:
    raw = str(os.getenv("RISK_EMAIL_RECIPIENTS", "")).strip()
    out: List[str] = []
    for piece in raw.split(","):
        addr = piece.strip()
        if addr:
            out.append(addr)
    return out


def send_risk_email(
    *,
    event_type: str,
    subject: str,
    body: str,
    repo: Optional[V2Repository] = None,
    recipients: Optional[List[str]] = None,
    smtp_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    to_list = list(recipients or _recipients_from_env())
    if not to_list:
        payload = {"send_ok": False, "reason": "no_recipients", "event_type": event_type}
        if repo:
            try:
                repo.save_mail_delivery_log(
                    event_type=event_type,
                    recipients=[],
                    subject=subject,
                    body_preview=body[:240],
                    send_ok=False,
                    error="no_recipients",
                )
            except Exception:
                pass
        return payload

    over = dict(smtp_overrides or {})
    smtp_host = str(over.get("SMTP_HOST") or os.getenv("SMTP_HOST", "")).strip()
    smtp_port = int(over.get("SMTP_PORT") or os.getenv("SMTP_PORT", "587"))
    smtp_user = str(over.get("SMTP_USER") or os.getenv("SMTP_USER", "")).strip()
    smtp_pass = str(over.get("SMTP_PASS") or os.getenv("SMTP_PASS", "")).strip()
    smtp_from = str(over.get("SMTP_FROM") or os.getenv("SMTP_FROM", smtp_user or "monitor@localhost")).strip()
    smtp_tls = str(over.get("SMTP_TLS") or os.getenv("SMTP_TLS", "1")).strip().lower() in {"1", "true", "yes", "on"}

    if not smtp_host:
        err = "smtp_host_missing"
        if repo:
            try:
                repo.save_mail_delivery_log(
                    event_type=event_type,
                    recipients=to_list,
                    subject=subject,
                    body_preview=body[:240],
                    send_ok=False,
                    error=err,
                )
            except Exception:
                pass
        return {"send_ok": False, "reason": err, "event_type": event_type, "recipients": to_list}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    try:
        with smtplib.SMTP(host=smtp_host, port=smtp_port, timeout=10) as s:
            if smtp_tls:
                s.starttls()
            if smtp_user:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        if repo:
            try:
                repo.save_mail_delivery_log(
                    event_type=event_type,
                    recipients=to_list,
                    subject=subject,
                    body_preview=body[:240],
                    send_ok=True,
                    error=None,
                )
            except Exception:
                pass
        return {"send_ok": True, "event_type": event_type, "recipients": to_list}
    except Exception as exc:
        if repo:
            try:
                repo.save_mail_delivery_log(
                    event_type=event_type,
                    recipients=to_list,
                    subject=subject,
                    body_preview=body[:240],
                    send_ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        return {"send_ok": False, "event_type": event_type, "recipients": to_list, "reason": str(exc)}
