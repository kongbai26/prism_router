"""Tool Call 缓存 — call_id → (name, arguments) 持久化映射。

解决 Codex 多轮工具调用时，对话历史被压缩导致孤儿 tool 消息的问题。
参考 codex-app-transfer 的 ToolCallCache 实现。

写时机：流式 tool call 完成时（_wrap_stream_as_responses 的 [DONE] 处理）
读时机：repair_tool_call_ids 重建缺失的 tool_call
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock

logger = logging.getLogger("prism_router.tool_call_cache")


class ToolCallCache:
    """持久化 tool call 缓存，call_id → (name, arguments)。"""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        persist_path: Path | None = None,
    ) -> None:
        self.max_size = max(1, max_size)
        self.ttl_ms = ttl_seconds * 1000
        self.persist_path = persist_path
        self._lock = Lock()
        self._entries: dict[str, dict] = {}  # call_id → {name, arguments, inserted_at_ms, access_count}
        if persist_path:
            self._load_from_disk()

    def save(self, call_id: str, name: str, arguments: str) -> None:
        """写入 tool call 缓存（流式 tool call 完成时调用）"""
        if not call_id or not call_id.strip():
            return
        with self._lock:
            self._evict_expired()
            if len(self._entries) >= self.max_size and call_id not in self._entries:
                self._evict_oldest()
            self._entries[call_id] = {
                "name": name,
                "arguments": arguments,
                "inserted_at_ms": int(time.time() * 1000),
                "access_count": 0,
            }
            snapshot = dict(self._entries)
        if self.persist_path:
            self._write_to_disk(snapshot)

    def get(self, call_id: str) -> tuple[str, str] | None:
        """查询 tool call 缓存，返回 (name, arguments) 或 None"""
        if not call_id or not call_id.strip():
            return None
        with self._lock:
            entry = self._entries.get(call_id)
            if not entry:
                return None
            if int(time.time() * 1000) - entry["inserted_at_ms"] > self.ttl_ms:
                del self._entries[call_id]
                return None
            entry["access_count"] += 1
            return (entry["name"], entry["arguments"])

    def _evict_expired(self) -> None:
        """淘汰过期条目（调用方需持有锁）"""
        now_ms = int(time.time() * 1000)
        expired = [k for k, v in self._entries.items() if now_ms - v["inserted_at_ms"] > self.ttl_ms]
        for k in expired:
            del self._entries[k]

    def _evict_oldest(self) -> None:
        """淘汰最旧/最少访问的条目（调用方需持有锁）"""
        if not self._entries:
            return
        oldest = min(
            self._entries, key=lambda k: (self._entries[k]["access_count"], self._entries[k]["inserted_at_ms"])
        )
        del self._entries[oldest]

    def _load_from_disk(self) -> None:
        """从磁盘加载缓存"""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                logger.warning("ToolCallCache: unknown version, starting fresh")
                return
            now_ms = int(time.time() * 1000)
            for call_id, entry in data.get("entries", {}).items():
                if now_ms - entry.get("inserted_at_ms", 0) <= self.ttl_ms:
                    self._entries[call_id] = entry
            logger.info("ToolCallCache: loaded %d entries from %s", len(self._entries), self.persist_path)
        except Exception as e:
            logger.warning("ToolCallCache: failed to load from disk: %s", e)

    def _write_to_disk(self, entries: dict[str, dict]) -> None:
        """原子写入磁盘（tmp + rename）"""
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persist_path.with_suffix(".tmp")
            payload = {"version": 1, "entries": entries}
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.rename(self.persist_path)
        except Exception as e:
            logger.warning("ToolCallCache: failed to write to disk: %s", e)


# 全局单例（容器列表，避免 global）
_cache: list[ToolCallCache | None] = [None]


def get_tool_call_cache() -> ToolCallCache:
    """获取全局 ToolCallCache 单例"""
    if _cache[0] is None:
        ttl = 3600
        max_size = 1000
        try:
            from prism_router.server import get_settings

            cfg = get_settings().logging.tool_call_cache
            ttl = cfg.get("ttl_seconds", 3600)
            max_size = cfg.get("max_size", 1000)
            persist = bool(cfg.get("persist", False))
        except Exception:
            persist = False
        persist_path = Path("logs/tool_call_cache.json") if persist else None
        _cache[0] = ToolCallCache(max_size=max_size, ttl_seconds=ttl, persist_path=persist_path)
    instance = _cache[0]
    assert instance is not None
    return instance
