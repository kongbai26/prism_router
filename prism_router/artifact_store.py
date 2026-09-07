"""ToolArtifactStore — 大 tool output 外部存储，防止 context 膨胀。

当 tool output 超过阈值时，将完整内容存入 SQLite，messages 中只保留摘要 + 引用。
发送给上游前自动恢复完整内容。
"""

import logging
import re
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

_ARTIFACT_REF_PATTERN = re.compile(r"\[Artifact: (art_[A-Za-z0-9]+)\]")
_DEFAULT_MAX_INLINE = 10240  # 10KB
_DEFAULT_TTL_HOURS = 24


class ToolArtifactStore:
    """tool output 大 payload 存储"""

    def __init__(
        self,
        db_path: str = "logs/requests.db",
        max_inline_size: int = _DEFAULT_MAX_INLINE,
        ttl_hours: int = _DEFAULT_TTL_HOURS,
    ):
        self._db_path = db_path
        self._max_inline = max_inline_size
        self._ttl_seconds = ttl_hours * 3600
        self._lock = threading.Lock()
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_table(self) -> None:
        conn = None
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    call_id TEXT,
                    output TEXT NOT NULL,
                    output_size INTEGER,
                    created_at REAL,
                    expires_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_expires ON tool_artifacts(expires_at)")
            conn.commit()
        except Exception as e:
            logger.warning("Failed to init tool_artifacts table: %s", e)
        finally:
            if conn:
                conn.close()

    def should_store(self, output: str) -> bool:
        """判断是否需要存入 artifact（超过阈值）"""
        return len(output.encode("utf-8")) > self._max_inline

    def store(self, call_id: str, output: str) -> str:
        """存储大 payload，返回 artifact_id"""
        from prism_router.db import gen_ulid

        artifact_id = f"art_{gen_ulid()[:20]}"
        now = time.time()
        output_size = len(output.encode("utf-8"))
        conn = None
        try:
            with self._lock:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO tool_artifacts (artifact_id, call_id, output, output_size, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (artifact_id, call_id, output, output_size, now, now + self._ttl_seconds),
                )
                conn.commit()
            logger.debug("Stored artifact %s (%d bytes, call_id=%s)", artifact_id, output_size, call_id)
        except Exception as e:
            logger.warning("Failed to store artifact: %s", e)
        finally:
            if conn:
                conn.close()
        return artifact_id

    def retrieve(self, artifact_id: str) -> str | None:
        """按 artifact_id 恢复完整内容"""
        conn = None
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT output FROM tool_artifacts WHERE artifact_id = ? AND expires_at > ?",
                (artifact_id, time.time()),
            ).fetchone()
            if row:
                return str(row[0])
        except Exception as e:
            logger.warning("Failed to retrieve artifact %s: %s", artifact_id, e)
        finally:
            if conn:
                conn.close()
        return None

    def make_summary(self, output: str, artifact_id: str) -> str:
        """生成截断摘要 + artifact 引用"""
        # 保留前 200 字符作为摘要
        summary = output[:200]
        if len(output) > 200:
            summary += "..."
        return f"{summary}\n[Artifact: {artifact_id}]"

    def is_artifact_ref(self, content: str) -> bool:
        """检查内容是否包含 artifact 引用"""
        return bool(_ARTIFACT_REF_PATTERN.search(content))

    def expand_artifacts(self, messages: list[dict]) -> list[dict]:
        """展开 messages 中所有 artifact 引用为完整内容"""
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not self.is_artifact_ref(content):
                continue
            match = _ARTIFACT_REF_PATTERN.search(content)
            if match:
                artifact_id = match.group(1)
                full = self.retrieve(artifact_id)
                if full is not None:
                    msg["content"] = full
                else:
                    logger.warning("Artifact %s not found or expired, keeping summary", artifact_id)
        return messages

    def purge_expired(self) -> int:
        """清理过期 artifacts，返回删除数量"""
        conn = None
        try:
            with self._lock:
                conn = self._get_conn()
                cursor = conn.execute("DELETE FROM tool_artifacts WHERE expires_at < ?", (time.time(),))
                deleted = cursor.rowcount
                conn.commit()
            if deleted:
                logger.info("Purged %d expired artifacts", deleted)
            return deleted
        except Exception as e:
            logger.warning("Failed to purge artifacts: %s", e)
            return 0
        finally:
            if conn:
                conn.close()


_store_instance: list[ToolArtifactStore | None] = [None]


def get_artifact_store(db_path: str = "logs/requests.db") -> ToolArtifactStore:
    """获取全局 ToolArtifactStore 单例"""
    if _store_instance[0] is None:
        # 从配置读取 TTL
        try:
            from prism_router.server import get_settings

            retention = get_settings().logging.retention_days
            ttl_hours = retention.get("artifacts", 1) * 24  # 天数转小时
        except Exception:
            ttl_hours = _DEFAULT_TTL_HOURS
        _store_instance[0] = ToolArtifactStore(db_path=db_path, ttl_hours=ttl_hours)
    instance = _store_instance[0]
    assert instance is not None
    return instance
