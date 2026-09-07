"""会话上下文：跟踪每个对话的最近路由决策"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import md5


@dataclass
class RouteDecision:
    """单次路由决策记录"""

    tier: str  # simple/mid/complex
    model_key: str  # 使用的模型，如 "zhipu/glm-4.5-flash"
    user_msg: str  # 用户消息摘要（前120字符）
    success: bool  # 请求是否成功（200）
    timestamp: float  # 时间戳
    has_tools: bool = False  # 是否带 tools


class ConversationContext:
    """
    会话上下文管理器

    存储策略：
    - 用消息列表的 hash 作为会话标识（前3条消息的MD5）
    - 每个会话保留最近 N 条路由决策，每条摘要 120 字符
    - 内存存储，最多跟踪 M 个会话
    - 淘汰策略：LRU（最久未访问的会话先淘汰）
    """

    def __init__(self, max_conversations: int = 200, history_window: int = 3):
        self.max_conversations = max_conversations
        self.history_window = history_window
        self._store: OrderedDict[str, list[RouteDecision]] = OrderedDict()
        # 会话级决策缓存：{conv_key: {choice_name: choice_value}}
        self._choices: dict[str, dict[str, str]] = {}

    def get_choice(self, messages: list[dict], choice_name: str) -> str | None:
        """获取会话级决策（如 rewrite_only_model_choice）"""
        key = self._make_key(messages)
        return self._choices.get(key, {}).get(choice_name)

    def set_choice(self, messages: list[dict], choice_name: str, value: str) -> None:
        """记录会话级决策"""
        key = self._make_key(messages)
        if key not in self._choices:
            self._choices[key] = {}
        self._choices[key][choice_name] = value

    def _make_key(self, messages: list[dict]) -> str:
        """
        用消息列表的 hash 作为会话标识

        策略：只用第一条用户消息作为会话标识
        这样即使对话历史增长，只要对话从同一条消息开始，就是同一个会话
        """
        # 找到第一条用户消息
        first_user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            first_user_msg = part.get("text", "")[:100]
                            break
                elif isinstance(content, str):
                    first_user_msg = content[:100]
                break

        # 如果没有找到用户消息，用前几条消息的 hash 作为 key
        if not first_user_msg:
            sample = []
            for msg in messages[:3]:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            content = part.get("text", "")[:50]
                            break
                elif isinstance(content, str):
                    content = content[:50]
                sample.append({"role": msg.get("role"), "content": content})
            raw = json.dumps(sample, sort_keys=True, ensure_ascii=False)
            return md5(raw.encode()).hexdigest()[:16]

        return md5(first_user_msg.encode()).hexdigest()[:16]

    def get_history(self, messages: list[dict]) -> list[RouteDecision]:
        """获取当前会话的历史路由决策"""
        key = self._make_key(messages)
        history = self._store.get(key, [])
        # 移到末尾（LRU）
        if key in self._store:
            self._store.move_to_end(key)
        return history

    def record(
        self,
        messages: list[dict],
        tier: str,
        model_key: str,
        success: bool,
        has_tools: bool = False,
    ):
        """记录一次路由决策"""
        key = self._make_key(messages)

        # 获取用户消息摘要（120字符，平衡信息量和长度）
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_msg = content[:120]
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_msg = part.get("text", "")[:120]
                            break
                break

        decision = RouteDecision(
            tier=tier,
            model_key=model_key,
            user_msg=user_msg,
            success=success,
            timestamp=time.time(),
            has_tools=has_tools,
        )

        if key not in self._store:
            self._store[key] = []

        history = self._store[key]
        history.append(decision)

        # 只保留最近 N 条
        if len(history) > self.history_window:
            self._store[key] = history[-self.history_window :]

        # 淘汰最旧会话（LRU）
        while len(self._store) > self.max_conversations:
            self._store.popitem(last=False)
