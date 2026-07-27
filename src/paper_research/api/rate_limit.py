import ipaddress
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from paper_research.config import Settings, get_settings
from paper_research.infrastructure.redis_service import get_redis_service

EXEMPT_GET_PATHS = {
    "/api/v1/health",
    "/api/v1/capabilities",
    "/docs",
    "/openapi.json",
}


def _trusted_networks(settings: Settings) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for raw in settings.trusted_proxy_cidrs.split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            continue
    return networks


def _is_trusted_proxy(host: str | None, settings: Settings) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in _trusted_networks(settings))


def resolve_client_identity(request: Request, settings: Settings) -> str:
    """Resolve the rate-limit identity without trusting spoofed proxy headers."""

    direct_host = request.client.host if request.client else None
    if not _is_trusted_proxy(direct_host, settings):
        return direct_host or "unknown"
    forwarded_for = request.headers.get("x-forwarded-for", "")
    for candidate in forwarded_for.split(","):
        value = candidate.strip()
        if not value:
            continue
        try:
            ipaddress.ip_address(value)
        except ValueError:
            continue
        return value
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        try:
            ipaddress.ip_address(real_ip)
        except ValueError:
            pass
        else:
            return real_ip
    return direct_host or "unknown"


def rate_limit_bucket_for_request(request: Request) -> str | None:
    path = request.url.path.rstrip("/") or "/"
    method = request.method.upper()
    if method == "GET" and (
        path in EXEMPT_GET_PATHS
        or path == "/api/v1/ui"
        or path.startswith("/api/v1/ui/")
    ):
        return None
    if path.startswith("/api/v1/research") or path.startswith("/api/v1/deep"):
        return "deep_research"
    if path.startswith("/api/v1/search"):
        return "search"
    if path.startswith("/api/v1/papers/upload") or (
        path == "/api/v1/papers" and method == "POST"
    ):
        return "upload"
    if "enrich-metadata" in path:
        return "metadata_enrichment"
    if method == "GET":
        return "read_api"
    return "default"


def limit_for_bucket(settings: Settings, bucket: str) -> int:
    return {
        "ui_page": settings.api_rate_limit_ui_page_per_minute,
        "read_api": settings.api_rate_limit_read_api_per_minute,
        "search": settings.api_rate_limit_search_per_minute,
        "upload": settings.api_rate_limit_upload_per_minute,
        "metadata_enrichment": settings.api_rate_limit_metadata_enrichment_per_minute,
        "deep_research": settings.api_rate_limit_deep_research_per_minute,
    }.get(bucket, settings.api_rate_limit_per_minute)


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        bucket = rate_limit_bucket_for_request(request)
        if bucket is None:
            return await call_next(request)
        identity = resolve_client_identity(request, settings)
        allowed, retry_after = get_redis_service().allow_request(
            identity,
            bucket_name=bucket,
            limit_per_minute=limit_for_bucket(settings, bucket),
        )
        if not allowed:
            request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
                content={
                    "error_code": "RATE_LIMITED",
                    "request_id": request_id,
                    "retry_after_seconds": retry_after,
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"rate limit exceeded; retry after {retry_after} seconds",
                        "request_id": request_id,
                    },
                },
            )
        return await call_next(request)
