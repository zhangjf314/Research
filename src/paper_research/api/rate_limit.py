from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from paper_research.infrastructure.redis_service import get_redis_service


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.endswith(("/health", "/capabilities")):
            return await call_next(request)
        identity = request.client.host if request.client else "unknown"
        if not get_redis_service().allow_request(identity):
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "rate limit exceeded"}},
            )
        return await call_next(request)
