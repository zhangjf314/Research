import hashlib
import json
import time
from pathlib import Path

import httpx

from paper_research.infrastructure.redis_service import RedisService


class CachedRetryClient:
    def __init__(
        self,
        cache_dir: Path,
        ttl_seconds: int = 3600,
        retries: int = 3,
        client: httpx.Client | None = None,
        redis_cache: RedisService | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.retries = retries
        self.client = client or httpx.Client(follow_redirects=True, timeout=30)
        self.redis_cache = redis_cache
        self.telemetry: dict[str, object] = {
            "cache_hit": False,
            "retry_count": 0,
            "rate_limited": False,
            "fallback_used": False,
        }

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> dict:
        cached = self._read_cache(url, params)
        if cached is not None:
            return cached
        response = self._get(url, params=params, headers=headers)
        payload = response.json()
        self._write_cache(url, params, payload)
        return payload

    def get_text(self, url: str, *, params: dict[str, object]) -> str:
        path = self._cache_path(url, params).with_suffix(".txt")
        if self.redis_cache:
            cached = self.redis_cache.get_json(path.stem)
            if isinstance(cached, dict) and "text" in cached:
                self.telemetry["cache_hit"] = True
                return str(cached["text"])
        if path.exists() and time.time() - path.stat().st_mtime <= self.ttl_seconds:
            self.telemetry["cache_hit"] = True
            return path.read_text(encoding="utf-8")
        response = self._get(url, params=params)
        if self.redis_cache:
            self.redis_cache.set_json(path.stem, {"text": response.text}, self.ttl_seconds)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return response.text

    def get_bytes(self, url: str) -> bytes:
        return self._get(url).content

    def _get(self, url: str, **kwargs: object) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.get(url, **kwargs)
                if response.status_code == 429:
                    self.telemetry["rate_limited"] = True
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    self.telemetry["retry_count"] = int(self.telemetry["retry_count"]) + 1
                    retry_after = 0.0
                    if isinstance(exc, httpx.HTTPStatusError):
                        value = exc.response.headers.get("Retry-After")
                        retry_after = float(value) if value and value.isdigit() else 0.0
                    time.sleep(min(10.0, max(retry_after, 0.25 * (2**attempt))))
        assert last_error is not None
        raise last_error

    def _cache_path(self, url: str, params: dict[str, object]) -> Path:
        key = json.dumps([url, params], sort_keys=True, ensure_ascii=True).encode()
        return self.cache_dir / f"{hashlib.sha256(key).hexdigest()}.json"

    def _read_cache(self, url: str, params: dict[str, object]) -> dict | None:
        key = self._cache_path(url, params).stem
        if self.redis_cache:
            cached = self.redis_cache.get_json(key)
            if cached is not None:
                self.telemetry["cache_hit"] = True
                return cached
        path = self._cache_path(url, params)
        if path.exists() and time.time() - path.stat().st_mtime <= self.ttl_seconds:
            self.telemetry["cache_hit"] = True
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, url: str, params: dict[str, object], payload: dict) -> None:
        if self.redis_cache:
            self.redis_cache.set_json(self._cache_path(url, params).stem, payload, self.ttl_seconds)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(url, params).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
