"""Structured, request-traced logging.

One JSON line per pipeline stage on stdout, all sharing a `request_id` so a
single `/chat` call can be reconstructed from interleaved logs:

    {"ts": ..., "level": "INFO", "request_id": "a1b2c3d4",
     "stage": "expand_query", "event": "expanded", "queries": [...], "n": 4}

INFO carries the one-line stage events; DEBUG adds the heavy payloads (full
chunk text, full prompt). SSNs are redacted unless LOG_PII=true.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar

from app.config import get_settings

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_configured = False

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


def redact_pii(value):
    if isinstance(value, str):
        return _SSN.sub("[redacted-ssn]", value)
    if isinstance(value, dict):
        return {k: redact_pii(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_pii(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", _request_id.get()),
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    s = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if s.log_json
                         else logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(s.log_level.upper())
    for noisy in ("sentence_transformers", "chromadb", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # chromadb 0.5.x logs a broken-posthog-wrapper stack at ERROR on every client
    # start ("capture() takes 1 positional argument but 3 were given"); it is
    # cosmetic and unfixable from our side. Silence just that logger.
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    _configured = True


_log = logging.getLogger("agentic_rag")


def log_event(stage: str, event: str, *, level: int = logging.INFO, **fields) -> None:
    if not get_settings().log_pii:
        fields = {k: redact_pii(v) for k, v in fields.items()}
    _log.log(level, f"{stage}.{event}",
             extra={"stage": stage, "event": event, "request_id": _request_id.get(), **fields})
