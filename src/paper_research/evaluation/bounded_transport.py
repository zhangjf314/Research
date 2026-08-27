"""Bounded, observable HTTP transport for the RQ17-R2 development harness.

It is deliberately an evaluation-only adapter.  It does not change a Jina
provider's model, endpoint, payload, or response validation; it only bounds
retry/timeout behaviour and records transport facts without credentials or
response bodies.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class TransportLedger:
    """Thread-safe, sanitized call/retry records suitable for an artifact."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)


class RequestPacer:
    """A single-concurrency pacing limiter; it never changes request payloads."""

    def __init__(self, minimum_interval_seconds: float) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("minimum interval must be non-negative")
        self.minimum_interval_seconds = minimum_interval_seconds
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.minimum_interval_seconds
        if delay:
            time.sleep(delay)
        return delay


class BoundedTransportClient:
    """``httpx.Client``-shaped adapter with <=4 exact-payload attempts."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        ledger: TransportLedger,
        stage: str,
        provider: str,
        model: str,
        request_type: str,
        pacer: RequestPacer,
        max_attempts: int = 4,
        max_backoff_seconds: float = 8.0,
        sleep: Any = time.sleep,
    ) -> None:
        if not 1 <= max_attempts <= 4:
            raise ValueError("max_attempts must be between 1 and 4")
        self.client = client
        self.ledger = ledger
        self.stage = stage
        self.provider = provider
        self.model = model
        self.request_type = request_type
        self.pacer = pacer
        self.max_attempts = max_attempts
        self.max_backoff_seconds = max_backoff_seconds
        self.sleep = sleep

    def post(
        self, url: str, *, headers: dict[str, str] | None = None, json: Any = None, **kwargs: Any
    ) -> httpx.Response:
        payload_hash = canonical_hash(json)
        last_response: httpx.Response | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            paced_seconds = self.pacer.wait()
            started = time.perf_counter()
            status: int | None = None
            retry_after: float | None = None
            retryable = False
            try:
                # Forward the original object, not a reconstructed payload.
                response = self.client.post(url, headers=headers, json=json, **kwargs)
                last_response = response
                status = response.status_code
                retryable = status in {408, 429} or status >= 500
                retry_after = _retry_after_seconds(response.headers) if retryable else None
                if not retryable or attempt == self.max_attempts:
                    self._record(
                        attempt,
                        payload_hash,
                        status,
                        retry_after,
                        paced_seconds,
                        started,
                        "response",
                    )
                    return response
                self._record(
                    attempt, payload_hash, status, retry_after, paced_seconds, started, "retry"
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                retryable = True
                self._record(
                    attempt, payload_hash, None, None, paced_seconds, started, type(exc).__name__
                )
                if attempt == self.max_attempts:
                    raise
            if retryable:
                delay = retry_after if retry_after is not None else self._backoff_seconds(attempt)
                self.sleep(delay)
        if last_response is not None:
            return last_response
        assert last_error is not None
        raise last_error

    def _record(
        self,
        attempt: int,
        payload_hash: str,
        status: int | None,
        retry_after: float | None,
        paced_seconds: float,
        started: float,
        outcome: str,
    ) -> None:
        self.ledger.append(
            {
                "stage": self.stage,
                "provider": self.provider,
                "model": self.model,
                "request_type": self.request_type,
                "attempt": attempt,
                "max_attempts": self.max_attempts,
                "http_status": status,
                "retry_after_seconds": retry_after,
                "paced_seconds": round(paced_seconds, 4),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "payload_hash": payload_hash,
                "outcome": outcome,
            }
        )

    def _backoff_seconds(self, attempt: int) -> float:
        # A deterministic small jitter is reproducible and remains bounded.
        jitter = random.Random(f"{self.stage}:{attempt}").uniform(0.0, 0.125)
        return min(self.max_backoff_seconds, 0.5 * (2 ** (attempt - 1)) + jitter)


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, min(120.0, float(value)))
    except ValueError:
        return None
