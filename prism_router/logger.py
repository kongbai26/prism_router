"""日志格式化：Rich 控制台 handler、请求级日志适配器"""

from __future__ import annotations

import logging
import re

from rich.console import Console
from rich.logging import RichHandler

_RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z][a-zA-Z0-9 ]*\]")

# ── Rich console (stderr) ──

console = Console(stderr=True, highlighter=None)


class CompactFormatter(logging.Formatter):
    """丢弃堆栈，只保留日志消息本身，去除 Rich 标记"""

    def format(self, record: logging.LogRecord) -> str:
        saved_exc = record.exc_info
        saved_stack = record.stack_info
        record.exc_info = None
        record.stack_info = None
        try:
            msg = super().format(record)
            return _RICH_TAG_RE.sub("", msg)
        finally:
            record.exc_info = saved_exc
            record.stack_info = saved_stack


def make_console_handler() -> RichHandler:
    """创建 Rich 控制台 handler — 只显示一行错误摘要"""
    handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        markup=True,
        rich_tracebacks=False,
    )
    handler.setFormatter(CompactFormatter("%(message)s"))
    return handler


# ── 请求级日志适配器 ──


class RequestLogger(logging.LoggerAdapter):
    """为每个请求添加 [req_id] 前缀，方便区分不同请求的日志"""

    def process(self, msg, kwargs):
        rid = self.extra.get("request_id", "")
        # 取 UUID 部分（唯一），跳过时间戳前缀
        short_id = rid.split("-")[-1] if "-" in rid else rid[:8]
        if short_id:
            return f"[bold cyan]\\[{short_id}][/bold cyan] {msg}", kwargs
        return msg, kwargs
