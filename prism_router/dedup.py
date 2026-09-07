"""请求去重：短时间内相同请求返回 429"""

from __future__ import annotations

import hashlib
import json
import logging
import time

logger = logging.getLogger("prism_router.dedup")


class InFlightDedup:
    """追踪在途请求，短时间内相同请求返回 429"""

    def __init__(self, window_seconds: int = 10):
        self.window_seconds = window_seconds
        self._inflight: dict[str, float] = {}  # key -> register_time

    def _make_key(self, model: str, messages: list[dict], tools: list | None) -> str:
        raw = json.dumps(
            {"model": model, "messages": messages, "tools": tools},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def check_and_register(self, model: str, messages: list[dict], tools: list | None) -> str | None:
        """检查是否有重复请求。返回 key（重复）或 None（不重复，已注册）"""
        key = self._make_key(model, messages, tools)
        now = time.time()

        # 清理过期条目
        self._cleanup(now)

        logger.debug("Dedup check: key=%s, model=%s, inflight=%d", key[:12], model, len(self._inflight))

        if key in self._inflight:
            elapsed = now - self._inflight[key]
            remaining = max(1, self.window_seconds - int(elapsed))
            logger.info("Duplicate request detected (key=%s, %.0fs ago, retry after %ds)", key[:12], elapsed, remaining)
            return key

        self._inflight[key] = now
        return None

    def complete(self, model: str, messages: list[dict], tools: list | None) -> None:
        """请求完成，移除追踪"""
        key = self._make_key(model, messages, tools)
        self._inflight.pop(key, None)

    def complete_by_key(self, key: str) -> None:
        """请求完成，直接用 key 移除追踪（避免重算）"""
        self._inflight.pop(key, None)

    def _cleanup(self, now: float) -> None:
        """清理过期条目"""
        expired = [k for k, t in self._inflight.items() if now - t > self.window_seconds]
        for k in expired:
            del self._inflight[k]


# 全局实例（容器列表，避免 global）
_dedup: list[InFlightDedup | None] = [None]


def get_dedup(window_seconds: int = 10) -> InFlightDedup:
    instance = _dedup[0]
    if instance is None or instance.window_seconds != window_seconds:
        instance = InFlightDedup(window_seconds=window_seconds)
        _dedup[0] = instance
    return instance


def reset_dedup() -> None:
    _dedup[0] = None
