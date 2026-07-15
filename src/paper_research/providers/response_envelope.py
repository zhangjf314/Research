"""Durable provider response envelope written before business-schema parsing."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderUsageRecord(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    usage_source: str = "provider_reported"
    monetary_cost_usd: str = "0"
    cost_basis: str = "explicit_free_provider"


class ProviderResponseEnvelope(BaseModel):
    request_id: str
    provider: str
    model: str
    received_at: str
    raw_body_hash: str
    parsed_provider_payload: dict[str, Any]
    usage: ProviderUsageRecord
    finish_reason: str | None = None
    response_status: str = "completed"
    parse_status: str = "not_started"
    parse_error_type: str | None = None
    parse_error_message: str | None = None
    schema_version: str = "provider-response-envelope-v1"


class ProviderResponseEnvelopeStore:
    """Persist raw response and settled usage before parsing generated QA content."""

    def __init__(self, run_dir: Path, ledger_path: Path) -> None:
        self.run_dir = run_dir
        self.ledger_path = ledger_path
        self.raw_path = run_dir / "raw-provider-response.json"
        self.envelope_path = run_dir / "provider-response-envelope.json"

    def _event(self, event: str, **values: Any) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event_id": uuid.uuid4().hex,
                        "event": event,
                        "timestamp": datetime.now(UTC).isoformat(),
                        **values,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def record_received(
        self,
        *,
        request_id: str,
        provider: str,
        model: str,
        raw_body: bytes,
    ) -> ProviderResponseEnvelope:
        """Settle provider usage and persist raw response before business parsing."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        body_hash = hashlib.sha256(raw_body).hexdigest()
        self._event(
            "raw_response_received",
            request_id=request_id,
            raw_body_hash=body_hash,
        )
        # Preserve the exact provider bytes before attempting even transport-level
        # JSON decoding.  A malformed body is still evidence and must survive.
        self.raw_path.write_bytes(raw_body)
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._event(
                "raw_response_persisted",
                request_id=request_id,
                path=self.raw_path.name,
            )
            raise
        usage_body = payload.get("usage") or {}
        usage = ProviderUsageRecord(
            input_tokens=int(usage_body.get("prompt_tokens", 0)),
            output_tokens=int(usage_body.get("completion_tokens", 0)),
            total_tokens=int(
                usage_body.get(
                    "total_tokens",
                    int(usage_body.get("prompt_tokens", 0))
                    + int(usage_body.get("completion_tokens", 0)),
                )
            ),
        )
        self._event(
            "provider_usage_recorded",
            request_id=request_id,
            usage=usage.model_dump(),
            active_reserved_tokens=0,
        )
        self._event(
            "raw_response_persisted",
            request_id=request_id,
            path=self.raw_path.name,
        )
        choices = payload.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        envelope = ProviderResponseEnvelope(
            request_id=request_id,
            provider=provider,
            model=str(payload.get("model") or model),
            received_at=datetime.now(UTC).isoformat(),
            raw_body_hash=body_hash,
            parsed_provider_payload=payload,
            usage=usage,
            finish_reason=finish_reason,
        )
        self._write(envelope)
        return envelope

    def parsing_started(
        self, envelope: ProviderResponseEnvelope
    ) -> ProviderResponseEnvelope:
        updated = envelope.model_copy(update={"parse_status": "started"})
        self._write(updated)
        self._event(
            "response_parsing_started", request_id=updated.request_id
        )
        return updated

    def parsed(
        self, envelope: ProviderResponseEnvelope
    ) -> ProviderResponseEnvelope:
        updated = envelope.model_copy(
            update={
                "parse_status": "parsed",
                "parse_error_type": None,
                "parse_error_message": None,
            }
        )
        self._write(updated)
        self._event("response_parsed", request_id=updated.request_id)
        return updated

    def post_processing_failed(
        self, envelope: ProviderResponseEnvelope, error: Exception
    ) -> ProviderResponseEnvelope:
        updated = envelope.model_copy(
            update={
                "parse_status": "post_processing_failed",
                "parse_error_type": type(error).__name__,
                "parse_error_message": str(error)[:1000],
            }
        )
        self._write(updated)
        self._event(
            "post_processing_failed",
            request_id=updated.request_id,
            parse_error_type=updated.parse_error_type,
            active_reserved_tokens=0,
        )
        return updated

    def _write(self, envelope: ProviderResponseEnvelope) -> None:
        temporary = self.envelope_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(envelope.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.envelope_path)
