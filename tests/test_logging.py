import json
import logging

from app.observability.logging import (
    JsonFormatter,
    log_event,
    new_request_id,
    redact_pii,
    set_request_id,
)


def _format_one(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def test_json_formatter_emits_expected_keys():
    rec = logging.LogRecord("agentic_rag", logging.INFO, __file__, 1, "triage.routed",
                            None, None)
    rec.stage, rec.event, rec.request_id, rec.route = "triage", "routed", "abc123", "rag_only"
    out = _format_one(rec)
    assert out["level"] == "INFO"
    assert out["stage"] == "triage" and out["event"] == "routed"
    assert out["request_id"] == "abc123" and out["route"] == "rag_only"


def test_redact_pii_masks_ssn_recursively():
    assert redact_pii("SSN 123-45-6789") == "SSN [redacted-ssn]"
    assert redact_pii({"q": "my ssn is 123-45-6789", "n": 5}) == \
        {"q": "my ssn is [redacted-ssn]", "n": 5}
    assert redact_pii(["123-45-6789"]) == ["[redacted-ssn]"]


def test_log_event_carries_request_id_and_redacts(caplog):
    set_request_id("req-42")
    with caplog.at_level(logging.INFO, logger="agentic_rag"):
        log_event("api", "request.received", question="pin 123-45-6789")
    rec = caplog.records[-1]
    assert rec.request_id == "req-42"
    assert rec.stage == "api" and rec.event == "request.received"
    assert rec.question == "pin [redacted-ssn]"


def test_new_request_id_is_short_and_unique():
    a, b = new_request_id(), new_request_id()
    assert a != b and len(a) == 8
