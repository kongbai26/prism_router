"""本地后端适配器基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LocalBackend(ABC):
    """本地推理后端的抽象基类"""

    @abstractmethod
    async def chat_completions(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict | AsyncIterator:
        """
        发送 chat completions 请求。

        Args:
            model: 模型名
            messages: 消息列表
            stream: 是否 streaming
            **kwargs: 其他 OpenAI 兼容参数

        Returns:
            非 streaming: OpenAI 格式 dict
            streaming: AsyncIterator yielding SSE data dicts
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """检查后端是否可用"""
        ...
