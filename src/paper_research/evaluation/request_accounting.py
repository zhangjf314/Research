"""Idempotent request reservation closure for every terminal outcome."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class RequestTerminalState(StrEnum):
    COMPLETED = "completed"
    PROVIDER_FAILED = "provider_failed"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_FAILED = "schema_failed"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    POLICY_FAILED = "policy_failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


def _terminal_events(events: list[dict[str, Any]], reservation_id: str) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("reservation_id") == reservation_id
        and event.get("event")
        in {"reservation_settled", "reservation_released", "billing_unknown_terminal"}
    ]


def finalize_request_accounting(
    events: list[dict[str, Any]],
    *,
    reservation_id: str,
    request_id: str,
    reserved_tokens: int,
    terminal_state: RequestTerminalState | str,
    provider_usage: dict[str, Any] | None,
    request_sent: bool,
    billing_known_no_charge: bool = False,
) -> dict[str, Any]:
    """Return one terminal accounting event, reusing an existing event idempotently."""
    state = RequestTerminalState(terminal_state)
    existing = _terminal_events(events, reservation_id)
    if len(existing) > 1:
        raise ValueError(f"duplicate terminal accounting for {reservation_id}")
    if existing:
        return existing[0]
    idempotency_key = f"request-accounting-v1:{reservation_id}"
    common = {
        "reservation_id": reservation_id,
        "request_id": request_id,
        "terminal_state": state.value,
        "idempotency_key": idempotency_key,
        "reserved_tokens": int(reserved_tokens),
        "remaining_active_tokens": 0,
    }
    if provider_usage:
        return {
            **common,
            "event": "reservation_settled",
            "settled_tokens": int(provider_usage["total_tokens"]),
            "released_tokens": 0,
            "usage_source": provider_usage.get("usage_source", "provider_reported"),
            "billing_liability": "settled",
        }
    if not request_sent or billing_known_no_charge:
        return {
            **common,
            "event": "reservation_released",
            "settled_tokens": 0,
            "released_tokens": int(reserved_tokens),
            "usage_source": None,
            "billing_liability": "none",
        }
    return {
        **common,
        "event": "billing_unknown_terminal",
        "settled_tokens": 0,
        "released_tokens": 0,
        "usage_source": None,
        "billing_liability": "manual_review",
        "liability_ceiling_tokens": int(reserved_tokens),
    }


def close_reservation_for_terminal_run(
    events: list[dict[str, Any]],
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append at most one terminal event and report the effective active balance."""
    terminal = finalize_request_accounting(events, **kwargs)
    if terminal not in events:
        events = [*events, terminal]
    return events, {
        "reservation_id": kwargs["reservation_id"],
        "effective_active_tokens": 0,
        "terminal_event": terminal["event"],
        "idempotency_key": terminal["idempotency_key"],
    }
