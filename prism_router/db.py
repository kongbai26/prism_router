"""SQLite 持久化：请求日志、渠道健康状态"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("prism_router.db")

_db_path: str | None = None
_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def gen_ulid() -> str:
    """生成 26 字符的 ULID（时间有序、全局唯一）"""
    milli = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")
    C32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    ts = ""
    for _ in range(10):
        ts = C32[milli & 0x1F] + ts
        milli >>= 5
    rand = ""
    for _ in range(16):
        rand = C32[randomness & 0x1F] + rand
        randomness >>= 5
    return ts + rand


# request_logs 完整列定义（新建/重建表时使用）
_REQUEST_LOGS_COLUMNS = """
    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 自增主键
    request_id TEXT UNIQUE,                  -- 请求唯一标识（ULID）
    timestamp REAL,                          -- 时间戳（秒）
    time_str TEXT,                           -- 可读时间（北京时间）
    client_model TEXT,                       -- 客户端请求的模型
    final_model TEXT,                        -- 最终使用的模型
    tier TEXT,                               -- 路由层级（simple/mid/complex）
    classifier_method TEXT,                  -- 分类方式（model/heuristic/passthrough）
    classifier_model TEXT,                   -- 分类使用的模型
    status_code INTEGER,                     -- HTTP 状态码
    latency_ms REAL,                         -- 总耗时（毫秒）
    upstream_latency_ms REAL,                -- 上游耗时（毫秒）
    prompt_tokens INTEGER,                   -- 输入 token 数
    completion_tokens INTEGER,               -- 输出 token 数
    total_tokens INTEGER,                    -- 总 token 数
    cost REAL,                               -- 费用（美元）
    is_stream INTEGER DEFAULT 0,             -- 是否流式请求
    is_fallback INTEGER DEFAULT 0,           -- 是否触发了降级
    tried_tiers TEXT,                        -- 尝试过的层级列表（JSON）
    error TEXT,                              -- 错误信息
    is_rewritten INTEGER DEFAULT 0,          -- 是否经过 prompt 改写
    rewriter_latency_ms REAL DEFAULT 0,      -- 改写耗时（毫秒）
    last_user_msg TEXT,                      -- 最后一条用户消息摘要（前200字符）
    msg_count INTEGER DEFAULT 0,             -- 消息总数
    has_tools INTEGER DEFAULT 0,             -- 是否携带工具
    tool_count INTEGER DEFAULT 0,            -- 工具数量
    tools_upgraded INTEGER DEFAULT 0         -- 是否触发工具升级
"""


_TZ_BEIJING = timezone(timedelta(hours=8))


def _now_str() -> str:
    return datetime.now(_TZ_BEIJING).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def init_logger(db_path: str, retention_days: dict | None = None) -> None:
    """初始化 SQLite 连接和表结构（兼容旧表，自动补列）"""
    global _db_path, _conn
    _db_path = db_path

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    except sqlite3.OperationalError:
        # DB 文件损坏或残留 WAL/SHM 文件不匹配，自动清理重建
        logger.warning("DB corrupted or stale WAL/SHM files, rebuilding: %s", db_path)
        _conn = None
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(db_path + suffix).unlink(missing_ok=True)
            except Exception:
                pass
        _conn = sqlite3.connect(db_path, check_same_thread=False)

    # WAL 模式：提升读写并发性能
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA synchronous=NORMAL")

    table_exists = (
        _conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='request_logs'").fetchone()
        is not None
    )

    if not table_exists:
        _conn.execute(f"CREATE TABLE request_logs ({_REQUEST_LOGS_COLUMNS})")
        _conn.commit()
        logger.info("Created request_logs table: %s", db_path)
        # 新表直接跳到创建其他表
    else:
        cursor = _conn.execute("PRAGMA table_info(request_logs)")
        columns = {row[1] for row in cursor.fetchall()}
        if "request_id" not in columns:
            old_columns = columns.copy()
            col_mapping = {
                "model_key": "final_model",
                "classifier_score": None,
            }
            _conn.execute("ALTER TABLE request_logs RENAME TO request_logs_old")
            _conn.execute(f"CREATE TABLE request_logs ({_REQUEST_LOGS_COLUMNS})")
            try:
                select_cols = []
                insert_cols = []
                for old_col in sorted(old_columns):
                    if old_col == "id":
                        continue
                    new_col = col_mapping.get(old_col, old_col)
                    if new_col is None:
                        continue
                    select_cols.append(old_col)
                    insert_cols.append(new_col)
                if select_cols:
                    sql = f"INSERT INTO request_logs ({', '.join(insert_cols)}) SELECT {', '.join(select_cols)} FROM request_logs_old"
                    _conn.execute(sql)
            except Exception as e:
                logger.warning("Failed to migrate old data: %s", e)
            _conn.execute("DROP TABLE IF EXISTS request_logs_old")
            logger.info("Migrated request_logs: recreated with new schema")
        else:
            for col, typedef in [
                ("time_str", "TEXT"),
                ("upstream_latency_ms", "REAL"),
                ("classifier_method", "TEXT"),
                ("classifier_model", "TEXT"),
                ("prompt_tokens", "INTEGER"),
                ("completion_tokens", "INTEGER"),
                ("total_tokens", "INTEGER"),
                ("cost", "REAL"),
                ("is_stream", "INTEGER DEFAULT 0"),
                ("is_fallback", "INTEGER DEFAULT 0"),
                ("tried_tiers", "TEXT"),
                ("error", "TEXT"),
                ("is_rewritten", "INTEGER DEFAULT 0"),
                ("rewriter_latency_ms", "REAL DEFAULT 0"),
                ("last_user_msg", "TEXT"),
                ("msg_count", "INTEGER DEFAULT 0"),
                ("has_tools", "INTEGER DEFAULT 0"),
                ("tool_count", "INTEGER DEFAULT 0"),
                ("tools_upgraded", "INTEGER DEFAULT 0"),
            ]:
                if col not in columns:
                    _conn.execute(f"ALTER TABLE request_logs ADD COLUMN {col} {typedef}")

    _conn.commit()

    # request_bodies 表：请求体/响应体，方便排查
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS request_bodies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 自增主键
            request_id TEXT UNIQUE,                   -- 关联 request_logs
            timestamp REAL,                           -- 时间戳（秒）
            time_str TEXT,                            -- 可读时间（北京时间）
            user_messages TEXT,                       -- 用户消息列表（JSON数组）
            request_raw TEXT,                         -- 精简请求体（结构摘要）
            request_raw_size INTEGER DEFAULT 0,       -- 精简请求体大小（字节）
            request_raw_truncated INTEGER DEFAULT 0,  -- 精简请求体是否被截断
            response_content TEXT,                    -- 助手回复内容
            response_model TEXT,                      -- 实际响应模型
            response_finish_reason TEXT,              -- 停止原因（stop/length）
            response_raw TEXT,                        -- 精简响应体（结构摘要）
            response_raw_size INTEGER DEFAULT 0,      -- 精简响应体大小（字节）
            response_raw_truncated INTEGER DEFAULT 0, -- 精简响应体是否被截断
            full_request TEXT,                        -- 完整请求体 JSON
            full_request_size INTEGER DEFAULT 0,      -- 完整请求体大小（字节）
            full_request_truncated INTEGER DEFAULT 0, -- 完整请求体是否被截断
            full_response TEXT,                       -- 完整响应体 JSON
            full_response_size INTEGER DEFAULT 0,     -- 完整响应体大小（字节）
            full_response_truncated INTEGER DEFAULT 0 -- 完整响应体是否被截断
        )
    """)
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_bodies_request_id ON request_bodies(request_id)")
    _conn.commit()

    # request_bodies 表迁移：补全新列
    try:
        cursor = _conn.execute("PRAGMA table_info(request_bodies)")
        body_columns = {row[1] for row in cursor.fetchall()}
        for col, typedef in [
            ("full_request", "TEXT"),
            ("full_request_size", "INTEGER DEFAULT 0"),
            ("full_request_truncated", "INTEGER DEFAULT 0"),
            ("full_response", "TEXT"),
            ("full_response_size", "INTEGER DEFAULT 0"),
            ("full_response_truncated", "INTEGER DEFAULT 0"),
        ]:
            if col not in body_columns:
                _conn.execute(f"ALTER TABLE request_bodies ADD COLUMN {col} {typedef}")
        _conn.commit()
    except Exception:
        logger.debug("request_bodies migration: columns may already exist")

    # rewrite_logs 表：改写前后对比
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS rewrite_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 自增主键
            request_id TEXT,                         -- 关联 request_logs
            timestamp REAL,                          -- 时间戳（秒）
            time_str TEXT,                           -- 可读时间（北京时间）
            rewriter_key TEXT,                       -- 改写模型标识
            target_key TEXT,                         -- 目标模型标识
            latency_ms REAL,                         -- 改写耗时（毫秒）
            original_text TEXT,                      -- 原始 prompt（前500字符）
            rewritten_text TEXT                      -- 改写后 prompt（前500字符）
        )
    """)
    _conn.commit()

    # error_logs 表：错误详情 + 降级链
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 自增主键
            request_id TEXT,                         -- 关联 request_logs
            timestamp REAL,                          -- 时间戳（秒）
            time_str TEXT,                           -- 可读时间（北京时间）
            status_code INTEGER,                     -- HTTP 状态码
            error_type TEXT,                         -- 错误类型（upstream_error/server_error）
            error_message TEXT,                      -- 错误信息（前500字符）
            tried_tiers TEXT,                        -- 尝试过的层级列表（降级链，JSON）
            client_model TEXT,                       -- 客户端请求的模型
            final_model TEXT                         -- 最终使用的模型
        )
    """)
    _conn.commit()

    # channel_health 表：渠道熔断器状态
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS channel_health (
            channel_id INTEGER PRIMARY KEY,          -- 渠道 ID
            state TEXT NOT NULL DEFAULT 'closed',    -- 熔断状态（closed/open/half_open）
            cooldown_until REAL NOT NULL DEFAULT 0,  -- 冷却结束时间戳
            failure_count INTEGER NOT NULL DEFAULT 0,        -- 累计失败次数
            consecutive_successes INTEGER NOT NULL DEFAULT 0, -- 连续成功次数
            slow_response_count INTEGER NOT NULL DEFAULT 0    -- 慢响应次数
        )
    """)
    try:
        _conn.execute("ALTER TABLE channel_health ADD COLUMN slow_response_count INTEGER NOT NULL DEFAULT 0")
    except Exception:
        logger.debug("channel_health migration: slow_response_count column may already exist")
    _conn.commit()
    logger.info("Request logger initialized: %s", db_path)

    # 后台线程清理旧数据，不阻塞启动
    threading.Thread(target=purge_old_logs, args=(retention_days,), daemon=True).start()


def close_db() -> None:
    """关闭 SQLite 连接，确保 WAL 文件合并"""
    global _conn
    if _conn is not None:
        try:
            _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


_ALLOWED_TABLES = {"request_bodies", "request_logs", "error_logs", "rewrite_logs"}


def purge_old_logs(retention_days: dict | None = None) -> None:
    """分级清理旧日志数据（SQLite 表 + 文本日志文件）"""
    if _conn is None:
        return
    if not retention_days:
        retention_days = {
            "request_bodies": 7,
            "request_logs": 30,
            "error_logs": 90,
            "rewrite_logs": 30,
        }

    # 1. 清理 SQLite 表
    batch_size = 10000
    for table, days in retention_days.items():
        if table not in _ALLOWED_TABLES:
            continue
        total_deleted = 0
        while True:
            try:
                with _lock:
                    result = _conn.execute(
                        f"DELETE FROM {table} WHERE id IN (  SELECT id FROM {table} WHERE timestamp < ? LIMIT ?)",
                        (time.time() - days * 86400, batch_size),
                    )
                    _conn.commit()
                    deleted = result.rowcount
                    total_deleted += deleted
                    if deleted == 0:
                        break
            except Exception as e:
                logger.warning("Failed to purge %s: %s", table, e)
                break
        if total_deleted > 0:
            logger.info("Purged %d rows from %s (older than %d days)", total_deleted, table, days)

    # 2. 清理 artifact 表（仅在启用过该功能时表才存在）
    try:
        with _lock:
            table_exists = _conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tool_artifacts'"
            ).fetchone()
            purged_count = 0
            if table_exists:
                result = _conn.execute("DELETE FROM tool_artifacts WHERE expires_at < ?", (time.time(),))
                _conn.commit()
                purged_count = result.rowcount
        if purged_count > 0:
            logger.info("Purged %d expired artifacts", purged_count)
    except Exception as e:
        logger.warning("Failed to purge artifacts: %s", e)

    # 3. 清理文本日志文件
    _purge_log_files("logs/server", retention_days.get("server_logs", 7))
    _purge_log_files("logs/rewrites", retention_days.get("rewrite_files", 7))


def _purge_log_files(log_dir: str, max_days: int) -> None:
    """删除指定目录下超过 max_days 天的 .log 文件（跳过今天的）"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    today = datetime.now().strftime("%Y%m%d")
    cutoff = time.time() - max_days * 86400
    deleted = 0
    try:
        for f in log_path.glob("*.log"):
            # 跳过今天的日志文件
            if today in f.name:
                continue
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        if deleted > 0:
            logger.info("Purged %d old log files from %s (older than %d days)", deleted, log_dir, max_days)
    except Exception as e:
        logger.warning("Failed to purge log files from %s: %s", log_dir, e)


# ── 渠道健康 ──


def save_channel_health(
    channel_id: int,
    state: str,
    cooldown_until: float,
    failure_count: int,
    consecutive_successes: int,
    slow_response_count: int = 0,
) -> None:
    """保存单个渠道的熔断器状态到 DB"""
    if _conn is None:
        return
    try:
        with _lock:
            _conn.execute(
                "INSERT OR REPLACE INTO channel_health "
                "(channel_id, state, cooldown_until, failure_count, consecutive_successes, slow_response_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (channel_id, state, cooldown_until, failure_count, consecutive_successes, slow_response_count),
            )
            _conn.commit()
    except Exception as e:
        logger.warning("Failed to save channel health: %s", e)


def load_all_channel_health() -> list[dict]:
    """从 DB 加载所有渠道的熔断器状态"""
    if _conn is None:
        return []
    try:
        with _lock:
            rows = _conn.execute(
                "SELECT channel_id, state, cooldown_until, failure_count, consecutive_successes, "
                "slow_response_count "
                "FROM channel_health"
            ).fetchall()
        return [
            {
                "channel_id": row[0],
                "state": row[1],
                "cooldown_until": row[2],
                "failure_count": row[3],
                "consecutive_successes": row[4],
                "slow_response_count": row[5] if len(row) > 5 else 0,
            }
            for row in rows
        ]
    except Exception as e:
        logger.warning("Failed to load channel health: %s", e)
        return []


def delete_channel_health(channel_id: int) -> None:
    """从 DB 删除单个渠道的熔断器状态"""
    if _conn is None:
        return
    try:
        with _lock:
            _conn.execute("DELETE FROM channel_health WHERE channel_id = ?", (channel_id,))
            _conn.commit()
    except Exception as e:
        logger.warning("Failed to delete channel health: %s", e)


# ── 请求日志 ──


def log_request(
    client_model: str | None = None,
    final_model: str = "",
    tier: str = "",
    classifier_method: str = "passthrough",
    classifier_model: str = "",
    status_code: int = 0,
    latency_ms: float = 0,
    upstream_latency_ms: float = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float = 0,
    is_stream: bool = False,
    is_fallback: bool = False,
    tried_tiers: list[str] | None = None,
    error: str | None = None,
    is_rewritten: bool = False,
    rewriter_latency_ms: float = 0,
    request_id: str = "",
    last_user_msg: str = "",
    msg_count: int = 0,
    has_tools: bool = False,
    tool_count: int = 0,
    tools_upgraded: bool = False,
) -> str:
    """记录一次请求的结果，返回 request_id"""
    if not request_id:
        request_id = gen_ulid()
    time_str = _now_str()

    if _conn is not None:
        try:
            with _lock:
                _conn.execute(
                    """INSERT INTO request_logs
                       (request_id, timestamp, time_str, client_model, final_model, tier,
                        classifier_method, classifier_model, status_code, latency_ms,
                        upstream_latency_ms, prompt_tokens, completion_tokens, total_tokens,
                        cost, is_stream, is_fallback, tried_tiers, error,
                        is_rewritten, rewriter_latency_ms,
                        last_user_msg, msg_count, has_tools, tool_count, tools_upgraded)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        request_id,
                        time.time(),
                        time_str,
                        client_model,
                        final_model,
                        tier,
                        classifier_method,
                        classifier_model,
                        status_code,
                        latency_ms,
                        upstream_latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        prompt_tokens + completion_tokens,
                        cost,
                        1 if is_stream else 0,
                        1 if is_fallback else 0,
                        json.dumps(tried_tiers) if tried_tiers else None,
                        error,
                        1 if is_rewritten else 0,
                        rewriter_latency_ms,
                        last_user_msg[:200] if last_user_msg else "",
                        msg_count,
                        1 if has_tools else 0,
                        tool_count,
                        1 if tools_upgraded else 0,
                    ),
                )
                _conn.commit()
        except Exception as e:
            logger.error("Failed to log request to DB: %s", e)

    return request_id


def log_request_bodies(
    request_id: str,
    request_body: dict,
    response_body: dict | None = None,
    body_max_size: int = 10240,
    full_body_max_size: int = 5120,
) -> None:
    """记录请求体/响应体到 DB，拆分用户消息和响应体方便排查"""
    if _conn is None:
        return

    time_str = _now_str()

    # ── 提取用户消息 ──
    messages = request_body.get("messages", [])
    user_msgs = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_msgs.append(content[:2000])
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", "")[:2000])
                if text_parts:
                    user_msgs.append(" ".join(text_parts))
    user_messages_json = json.dumps(user_msgs, ensure_ascii=False)

    # ── 完整请求体 ──
    full_req = json.dumps(request_body, ensure_ascii=False)
    full_req_size = len(full_req.encode("utf-8"))
    full_req_truncated = 1 if full_req_size > full_body_max_size else 0
    if full_req_truncated:
        full_req = full_req[:full_body_max_size]

    # ── 精简请求体（去掉 messages 内容，只保留结构） ──
    req_summary = dict(request_body)
    if "messages" in req_summary:
        req_summary["messages"] = [
            {
                "role": m.get("role", ""),
                "content_length": len(m.get("content") or "")
                if isinstance(m.get("content"), str)
                else len(m.get("content") or []),
            }
            for m in req_summary["messages"]
        ]
    req_raw = json.dumps(req_summary, ensure_ascii=False)
    req_raw_size = len(req_raw.encode("utf-8"))
    req_raw_truncated = 1 if req_raw_size > body_max_size else 0
    if req_raw_truncated:
        req_raw = req_raw[:body_max_size]

    # ── 提取响应体字段 + 完整响应体 ──
    resp_content = ""
    resp_model = ""
    resp_finish_reason = ""
    resp_raw = ""
    resp_raw_size = 0
    resp_raw_truncated = 0
    full_resp = ""
    full_resp_size = 0
    full_resp_truncated = 0
    if response_body:
        resp_model = response_body.get("model", "")
        choices = response_body.get("choices", [])
        if choices:
            first = choices[0]
            resp_finish_reason = first.get("finish_reason", "")
            msg = first.get("message", {})
            resp_content = msg.get("content", "")[:2000] if msg else ""

        # 完整响应体
        full_resp = json.dumps(response_body, ensure_ascii=False)
        full_resp_size = len(full_resp.encode("utf-8"))
        full_resp_truncated = 1 if full_resp_size > full_body_max_size else 0
        if full_resp_truncated:
            full_resp = full_resp[:full_body_max_size]

        # 精简响应体（去掉 content 内容，只保留结构）
        resp_summary = dict(response_body)
        if "choices" in resp_summary:
            resp_summary["choices"] = [
                {
                    k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                    for k, v in c.items()
                    if k != "message"
                }
                for c in resp_summary["choices"]
            ]
        resp_raw = json.dumps(resp_summary, ensure_ascii=False)
        resp_raw_size = len(resp_raw.encode("utf-8"))
        resp_raw_truncated = 1 if resp_raw_size > body_max_size else 0
        if resp_raw_truncated:
            resp_raw = resp_raw[:body_max_size]

    try:
        with _lock:
            _conn.execute(
                """INSERT OR REPLACE INTO request_bodies
                   (request_id, timestamp, time_str,
                    user_messages,
                    request_raw, request_raw_size, request_raw_truncated,
                    response_content, response_model, response_finish_reason,
                    response_raw, response_raw_size, response_raw_truncated,
                    full_request, full_request_size, full_request_truncated,
                    full_response, full_response_size, full_response_truncated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    time.time(),
                    time_str,
                    user_messages_json,
                    req_raw,
                    req_raw_size,
                    req_raw_truncated,
                    resp_content,
                    resp_model,
                    resp_finish_reason,
                    resp_raw,
                    resp_raw_size,
                    resp_raw_truncated,
                    full_req,
                    full_req_size,
                    full_req_truncated,
                    full_resp,
                    full_resp_size,
                    full_resp_truncated,
                ),
            )
            _conn.commit()
    except Exception as e:
        logger.error("Failed to log request bodies to DB: %s", e)


def log_rewrite(
    request_id: str = "",
    rewriter_key: str = "",
    target_key: str = "",
    latency_ms: float = 0,
    original_text: str = "",
    rewritten_text: str = "",
) -> None:
    """记录一次改写操作的前后对比到 DB"""
    if _conn is None:
        return
    time_str = _now_str()
    try:
        with _lock:
            _conn.execute(
                """INSERT INTO rewrite_logs
                   (request_id, timestamp, time_str, rewriter_key, target_key,
                    latency_ms, original_text, rewritten_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    time.time(),
                    time_str,
                    rewriter_key,
                    target_key,
                    latency_ms,
                    original_text[:500] if original_text else "",
                    rewritten_text[:500] if rewritten_text else "",
                ),
            )
            _conn.commit()
    except Exception as e:
        logger.error("Failed to log rewrite to DB: %s", e)


def log_error(
    request_id: str = "",
    status_code: int = 0,
    error_type: str = "",
    error_message: str = "",
    tried_tiers: list[str] | None = None,
    client_model: str = "",
    final_model: str = "",
) -> None:
    """记录一次错误请求的详细信息到 DB"""
    if _conn is None:
        return
    time_str = _now_str()
    try:
        with _lock:
            _conn.execute(
                """INSERT INTO error_logs
                   (request_id, timestamp, time_str, status_code, error_type,
                    error_message, tried_tiers, client_model, final_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    time.time(),
                    time_str,
                    status_code,
                    error_type,
                    error_message[:500] if error_message else "",
                    json.dumps(tried_tiers) if tried_tiers else None,
                    client_model or "",
                    final_model or "",
                ),
            )
            _conn.commit()
    except Exception as e:
        logger.error("Failed to log error to DB: %s", e)


def log_request_detail(
    request_id: str,
    time_str: str,
    client_model: str | None,
    final_model: str,
    tier: str,
    classifier_method: str,
    classifier_model: str,
    latency_ms: float,
    upstream_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    status_code: int,
    is_stream: bool,
    is_fallback: bool,
    tried_tiers: list[str] | None = None,
    error: str | None = None,
    tools_upgraded: bool = False,
    is_rewritten: bool = False,
    rewriter_latency_ms: float = 0,
    rewriter_model: str = "",
    target_model: str = "",
    last_user_msg: str = "",
) -> None:
    """记录请求排查摘要到日志（终端 RichHandler + 文件 FileHandler）"""
    summary_parts = []

    if client_model and final_model and client_model != final_model and client_model != "auto":
        summary_parts.append(f"FALLBACK: {client_model} → {final_model}")

    if tools_upgraded:
        summary_parts.append("Tools upgrade")

    if classifier_method != "passthrough":
        summary_parts.append(f"Classifier: {classifier_model} → output: {tier} ({classifier_method})")

    if upstream_latency_ms > 0:
        summary_parts.append(f"Upstream: {upstream_latency_ms:.0f}ms")

    if prompt_tokens > 0 or completion_tokens > 0:
        summary_parts.append(f"Tokens: {prompt_tokens}+{completion_tokens}={prompt_tokens + completion_tokens}")

    if is_fallback and tried_tiers:
        summary_parts.append(f"Fallback: {' → '.join(tried_tiers)}")

    if is_rewritten:
        if rewriter_model and target_model:
            summary_parts.append(f"Rewrite: {rewriter_model} → {target_model} ({rewriter_latency_ms:.0f}ms)")
        else:
            summary_parts.append(f"Rewrite: yes ({rewriter_latency_ms:.0f}ms)")
    elif rewriter_model:
        summary_parts.append("Rewrite: skip")

    if error:
        summary_parts.append(f"Error: {error}")

    if last_user_msg:
        summary_parts.append(f"Req: {last_user_msg!r}")

    # 单次 logger.info 输出（RichHandler 渲染终端，FileHandler 写文件）
    log_lines = [
        "",
        f"[{request_id}] {time_str}",
        f"Model: {client_model or 'auto'} → {final_model}",
        f"Tier: {tier} | Status: {status_code} | Time: {latency_ms:.0f}ms",
    ]
    log_lines.extend(summary_parts)
    logger.info("\n".join(log_lines))
    logger.info("")
