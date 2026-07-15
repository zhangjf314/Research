"""Fail-closed provider structured-output capability snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    REPORTED_NOT_VERIFIED = "reported_not_verified"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    supports_json_object: bool | None
    supports_json_schema: bool | None
    supports_tool_calling: bool | None
    supports_strict_schema: bool | None
    capability_source: str
    verified_at: date | None
    verification_status: VerificationStatus
    fallback_policy: str = "fail_closed"

    @property
    def snapshot_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def require(self, capability: str) -> None:
        value = getattr(self, capability)
        if value is not True or self.verification_status != VerificationStatus.VERIFIED:
            raise RuntimeError(f"provider capability is not verified: {capability}")


def siliconflow_qwen3_8b_stage13_5_snapshot() -> ProviderCapabilities:
    """Only json_object transport was observed; stronger modes remain unverified."""
    return ProviderCapabilities(
        provider="siliconflow",
        model="Qwen/Qwen3-8B",
        supports_json_object=True,
        supports_json_schema=None,
        supports_tool_calling=None,
        supports_strict_schema=False,
        capability_source=(
            "Stage 13.5 saved request configuration and 10/10 valid JSON responses; "
            "no official json_schema or tool-calling verification was performed"
        ),
        verified_at=date(2026, 7, 15),
        verification_status=VerificationStatus.VERIFIED,
    )
