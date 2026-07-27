from __future__ import annotations

from starlette.requests import Request

from paper_research.api.rate_limit import resolve_client_identity
from paper_research.config import Settings


def _request(client_host: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/papers",
            "headers": headers or [],
            "client": (client_host, 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_trusted_proxy_uses_forwarded_for_first_valid_address() -> None:
    settings = Settings(trusted_proxy_cidrs="172.16.0.0/12")
    request = _request(
        "172.18.0.6",
        [(b"x-forwarded-for", b"203.0.113.8, 172.18.0.6")],
    )

    assert resolve_client_identity(request, settings) == "203.0.113.8"


def test_untrusted_client_cannot_spoof_forwarded_for() -> None:
    settings = Settings(trusted_proxy_cidrs="172.16.0.0/12")
    request = _request(
        "198.51.100.10",
        [(b"x-forwarded-for", b"203.0.113.8")],
    )

    assert resolve_client_identity(request, settings) == "198.51.100.10"


def test_invalid_forwarded_for_falls_back_to_direct_proxy() -> None:
    settings = Settings(trusted_proxy_cidrs="172.16.0.0/12")
    request = _request(
        "172.18.0.6",
        [(b"x-forwarded-for", b"not-an-ip")],
    )

    assert resolve_client_identity(request, settings) == "172.18.0.6"
