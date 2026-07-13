import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import redis

from paper_research.config import Settings, get_settings


class RedisService:
    """Best-effort Redis cache, rate limiter, lock, and short-lived task state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = (
            redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.3,
                socket_timeout=0.5,
            )
            if settings.redis_url
            else None
        )
        self.last_error: str | None = None

    def ping(self) -> bool:
        if self.client is None:
            self.last_error = "REDIS_URL is not configured"
            return False
        try:
            result = bool(self.client.ping())
            self.last_error = None
            return result
        except redis.RedisError as exc:
            self.last_error = type(exc).__name__
            return False

    def get_json(self, key: str) -> Any | None:
        if self.client is None:
            return None
        try:
            value = self.client.get(f"paperresearch:cache:{key}")
            self.client.hincrby("paperresearch:metrics:cache", "hit" if value else "miss", 1)
            return json.loads(value) if value else None
        except (redis.RedisError, json.JSONDecodeError) as exc:
            self.last_error = type(exc).__name__
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        if self.client is None:
            return False
        try:
            self._enforce_cache_limit()
            self.client.setex(
                f"paperresearch:cache:{key}",
                ttl_seconds or self.settings.redis_cache_ttl_seconds,
                json.dumps(value, ensure_ascii=False),
            )
            self.client.hincrby("paperresearch:metrics:cache", "write", 1)
            return True
        except redis.RedisError as exc:
            self.last_error = type(exc).__name__
            return False

    def allow_request(self, identity: str) -> bool:
        if self.client is None:
            return True
        bucket = int(time.time() // 60)
        key = f"paperresearch:rate:{bucket}:{identity}"
        try:
            count = self.client.incr(key)
            if count == 1:
                self.client.expire(key, 120)
            return count <= self.settings.api_rate_limit_per_minute
        except redis.RedisError as exc:
            self.last_error = type(exc).__name__
            return True

    @contextmanager
    def lock(self, name: str, timeout: int = 300) -> Iterator[bool]:
        if self.client is None:
            yield False
            return
        lock = self.client.lock(f"paperresearch:lock:{name}", timeout=timeout, blocking_timeout=1)
        acquired = False
        try:
            acquired = bool(lock.acquire(blocking=True))
            yield acquired
        except redis.RedisError as exc:
            self.last_error = type(exc).__name__
            yield False
        finally:
            if acquired:
                try:
                    lock.release()
                except redis.RedisError:
                    pass

    def set_task_state(self, task_id: str, state: dict, ttl_seconds: int = 86400) -> bool:
        return self.set_json(f"task:{task_id}", state, ttl_seconds)

    def stats(self) -> dict[str, Any]:
        if self.client is None:
            return {"available": False, "used": False, "error": self.last_error}
        try:
            counters = self.client.hgetall("paperresearch:metrics:cache")
            hits = int(counters.get("hit", 0))
            misses = int(counters.get("miss", 0))
            keys = sum(1 for _ in self.client.scan_iter("paperresearch:*", count=200))
            return {
                "available": True,
                "used": bool(hits or misses or int(counters.get("write", 0))),
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_rate": round(hits / max(1, hits + misses), 6),
                "writes": int(counters.get("write", 0)),
                "key_count": keys,
                "ttl_seconds": self.settings.redis_cache_ttl_seconds,
            }
        except redis.RedisError as exc:
            return {"available": False, "used": False, "error": type(exc).__name__}

    def _enforce_cache_limit(self) -> None:
        assert self.client is not None
        count = int(self.client.dbsize())
        if count < self.settings.redis_max_cache_keys:
            return
        for key in self.client.scan_iter("paperresearch:cache:*", count=100):
            self.client.delete(key)
            count -= 1
            if count < self.settings.redis_max_cache_keys:
                break


@lru_cache
def get_redis_service() -> RedisService:
    return RedisService(get_settings())
