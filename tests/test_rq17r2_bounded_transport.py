from __future__ import annotations

import httpx
import pytest

from paper_research.evaluation.bounded_transport import (
    BoundedTransportClient,
    RequestPacer,
    TransportLedger,
)


def transport(responses: list[httpx.Response], *, sleeps: list[float]):
    seen: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return responses.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bounded = BoundedTransportClient(
        client=client,
        ledger=TransportLedger(),
        stage="test",
        provider="jina",
        model="model",
        request_type="test",
        pacer=RequestPacer(0),
        max_attempts=4,
        sleep=sleeps.append,
    )
    return bounded, seen


def test_retries_429_honours_retry_after_and_preserves_exact_body() -> None:
    sleeps: list[float] = []
    bounded, seen = transport(
        [
            httpx.Response(429, headers={"Retry-After": "1.5"}),
            httpx.Response(200, json={"ok": True}),
        ],
        sleeps=sleeps,
    )
    response = bounded.post("https://example.test/v1/rerank", json={"input": ["same"]})
    assert response.status_code == 200
    assert sleeps == [1.5]
    assert seen == [seen[0], seen[0]]
    events = bounded.ledger.snapshot()
    assert [event["http_status"] for event in events] == [429, 200]
    assert events[0]["payload_hash"] == events[1]["payload_hash"]


def test_retries_transient_network_error_at_most_four_times() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timed out")

    bounded = BoundedTransportClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        ledger=TransportLedger(),
        stage="test",
        provider="jina",
        model="model",
        request_type="test",
        pacer=RequestPacer(0),
        max_attempts=4,
        sleep=lambda _delay: None,
    )
    with pytest.raises(httpx.ConnectTimeout):
        bounded.post("https://example.test/v1/embeddings", json={"input": ["x"]})
    assert calls == 4
    assert len(bounded.ledger.snapshot()) == 4


def test_non_transient_response_is_not_retried() -> None:
    sleeps: list[float] = []
    bounded, _seen = transport([httpx.Response(400)], sleeps=sleeps)
    assert bounded.post("https://example.test", json={"x": 1}).status_code == 400
    assert sleeps == []
    assert len(bounded.ledger.snapshot()) == 1
