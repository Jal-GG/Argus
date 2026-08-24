"""Alert channel handlers.

``console`` is always available; webhook/email are opt-in and fail soft so a
missing SMTP server never breaks the pipeline.
"""

from __future__ import annotations

import contextlib
import json
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Fans out rule violations to configured channels."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        self.enabled = [c.lower() for c in cfg.get("enabled_channels", ["console"])]
        if "console" not in self.enabled:
            self.enabled.append("console")
        self.channels_cfg: dict = cfg
        self._recent: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def dispatch(self, violation_dict: dict, snapshot_bgr: bytes | None = None) -> None:
        channels = violation_dict.get("channels") or self.default_channels_for(violation_dict)
        for channel in channels:
            handler = getattr(self, f"_send_{channel}", None)
            if handler is None:
                logger.warning("Unknown alert channel '%s'", channel)
                continue
            try:
                handler(violation_dict, snapshot_bgr)
            except Exception as exc:
                logger.error("Alert via %s failed: %s", channel, exc)

    def default_channels_for(self, violation_dict: dict) -> list[str]:
        return self.enabled

    # ------------------------------------------------------------------ #
    def _send_console(self, alert: dict, _snapshot: bytes | None = None) -> None:
        logger.warning(
            "ALERT [%s] zone=%s person=%s :: %s",
            alert.get("rule_type"),
            alert.get("zone_name"),
            alert.get("global_id"),
            alert.get("reason"),
        )

    def _send_webhook(self, alert: dict, snapshot: bytes | None = None) -> None:
        cfg = self.channels_cfg.get("webhook") or {}
        for endpoint in cfg.get("endpoints", []):
            url = endpoint.get("url")
            if not url:
                continue
            files = None
            if snapshot is not None and endpoint.get("attach_snapshot", True):
                files = {"snapshot": ("snapshot.jpg", snapshot, "image/jpeg")}
            resp = requests.post(
                url,
                data={"payload": json.dumps(alert, default=str)},
                files=files,
                timeout=endpoint.get("timeout", 5.0),
            )
            resp.raise_for_status()

    def _send_email(self, alert: dict, snapshot: bytes | None = None) -> None:
        cfg = self.channels_cfg.get("email") or {}
        host, port = cfg.get("smtp_server", "localhost"), int(cfg.get("smtp_port", 587))
        username, password = cfg.get("username"), cfg.get("password")

        msg = MIMEMultipart()
        msg["Subject"] = f"[Security] {alert.get('rule_type')} — {alert.get('zone_name')}"
        msg["From"] = cfg.get("from_email", username or "surveillance@localhost")
        msg["To"] = ", ".join(cfg.get("to_emails", []))

        body = (
            f"Rule: {alert.get('rule_type')}\n"
            f"Zone: {alert.get('zone_name')}\n"
            f"Person: #{alert.get('global_id')}\n"
            f"Reason: {alert.get('reason')}\n"
            f"At: {alert.get('timestamp')}\n"
        )
        msg.attach(MIMEText(body, "plain"))
        if snapshot is not None:
            msg.attach(MIMEImage(snapshot, name="snapshot.jpg"))

        with smtplib.SMTP(host, port, timeout=10) as server:
            with contextlib.suppress(smtplib.SMTPNotSupportedError):
                server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
