"""智能改写器：在请求发送前改写最后一条 user message"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from prism_router.prompts import load_prompt
from prism_router.settings import Settings

logger = logging.getLogger("prism_router.rewriter")


class RewriterBreaker:
    """改写器熔断器：连续失败 → 跳过改写 → 冷却后自动探测 → 恢复"""

    def __init__(
        self, enabled: bool = True, failure_threshold: int = 3, cooldown_base: float = 30, cooldown_max: float = 600
    ):
        self.enabled = enabled
        self.failure_threshold = failure_threshold
        self.cooldown_base = cooldown_base
        self.cooldown_max = cooldown_max
        self._consecutive_failures: int = 0
        self._cooldown_until: float = 0
        self._cooldown_seconds: float = cooldown_base

    def is_available(self) -> bool:
        """改写器是否可用（未熔断或冷却已到期）"""
        if not self.enabled:
            return True
        if self._consecutive_failures < self.failure_threshold:
            return True
        if time.time() >= self._cooldown_until:
            return True  # 冷却到期，允许探测
        return False

    def record_success(self):
        """改写成功，重置状态"""
        if not self.enabled:
            return
        if self._consecutive_failures > 0:
            logger.info("Rewriter breaker: recovered (failures reset)")
        self._consecutive_failures = 0
        self._cooldown_until = 0
        self._cooldown_seconds = self.cooldown_base

    def record_failure(self):
        """改写失败，累加计数，达到阈值则熔断"""
        if not self.enabled:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._cooldown_seconds = min(
                self._cooldown_seconds * 2,
                self.cooldown_max,
            )
            self._cooldown_until = time.time() + self._cooldown_seconds
            logger.warning(
                "Rewriter breaker: OPEN (failures=%d, cooldown=%.0fs)",
                self._consecutive_failures,
                self._cooldown_seconds,
            )
        else:
            logger.debug("Rewriter breaker: failure %d/%d", self._consecutive_failures, self.failure_threshold)

    @property
    def state(self) -> str:
        if self._consecutive_failures >= self.failure_threshold:
            if time.time() >= self._cooldown_until:
                return "half_open"
            return "open"
        return "closed"


# 全局改写器熔断器（容器列表，避免 global）
_rewriter_breaker: list[RewriterBreaker | None] = [None]


def get_rewriter_breaker(settings: Settings | None = None) -> RewriterBreaker:
    if _rewriter_breaker[0] is None:
        cfg = settings.rewriting if settings else None
        _rewriter_breaker[0] = RewriterBreaker(
            enabled=cfg.breaker_enabled if cfg else True,
            failure_threshold=cfg.breaker_failure_threshold if cfg else 3,
            cooldown_base=cfg.breaker_cooldown_base if cfg else 60,
            cooldown_max=cfg.breaker_cooldown_max if cfg else 600,
        )
    breaker = _rewriter_breaker[0]
    assert breaker is not None
    return breaker


@dataclass
class RewriteMetadata:
    rewritten: bool
    original_text: str
    rewritten_text: str
    latency_ms: float
    rewriter_key: str = ""
    target_key: str = ""


def _extract_user_text(content: str | list | None) -> str:
    """从 OpenAI 格式的 content 中提取纯文本（支持 string 和 list 格式）"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return ""


def _find_last_user_message(messages: list[dict]) -> int | None:
    """找到最后一条 user message 的索引"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _has_agent_activity_after(messages: list[dict], last_user_idx: int) -> bool:
    """检测 last_user_idx 之后是否有 agent 自主执行证据。

    如果最后一条 user message 之后的消息包含：
    - role=tool 或 role=function（tool result）
    - role=assistant 且有 tool_calls / function_call
    则说明当前处于 agent tool-calling 循环中，不应改写。
    """
    for i in range(last_user_idx + 1, len(messages)):
        msg = messages[i]
        role = msg.get("role", "")
        if role in ("tool", "function"):
            return True
        if role == "assistant" and (msg.get("tool_calls") or msg.get("function_call")):
            return True
    return False


def _parse_chat_response(result) -> str:
    """从 chat completions 响应中提取文本，失败返回空字符串"""
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content", "") or ""


def _format_transforms(transforms: object) -> str:
    """将 RewritingTransforms 格式化为模板变量字符串"""
    clarity = getattr(transforms, "clarity", True)
    structure = getattr(transforms, "structure", True)
    completeness = getattr(transforms, "completeness", False)
    return f"clarity={clarity}, structure={structure}, completeness={completeness}"


def _build_rewrite_context(
    messages: list[dict],
    last_user_idx: int,
    max_rounds: int,
    max_chars: int,
) -> str:
    """构建改写器的多轮对话上下文。

    从 last_user_idx 之前的 messages 中取最后 max_rounds 轮，
    排除 system/tool/function 消息，每轮截断到 max_chars。
    """
    prior = messages[:last_user_idx]
    rows = []
    for msg in prior:
        role = msg.get("role", "")
        if role in ("system", "tool", "function"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            text = " ".join(parts)
        else:
            continue
        text = text.strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        rows.append(f"{role}: {text}")
    if len(rows) > max_rounds:
        rows = rows[-max_rounds:]
    return "\n".join(rows)


async def rewrite_prompt(
    messages: list[dict],
    target_model_key: str,
    settings: Settings,
) -> tuple[list[dict], RewriteMetadata]:
    """改写最后一条 user message，返回 (改写后的 messages, metadata)。

    - target_model_key: 用户请求中的目标模型（用于模板变量）
    - rewriter_model: config 中配置的改写器模型（用于实际调用）
    - 根据目标模型 tier 选择模板（小 tier 模板 / 大 tier 模板）
    """
    from prism_router.local_backends import get_backend
    from prism_router.local_backends.pool import get_cloud_client

    cfg = settings.get_model_config(target_model_key)
    if cfg is None:
        return messages, RewriteMetadata(
            rewritten=False,
            original_text="",
            rewritten_text="",
            latency_ms=0,
            rewriter_key="",
            target_key=target_model_key,
        )

    rewriting_cfg = settings.rewriting

    # 0. 熔断器检查
    breaker = get_rewriter_breaker(settings)
    if not breaker.is_available():
        logger.info("Skip rewrite: rewriter breaker %s", breaker.state)
        return messages, RewriteMetadata(
            rewritten=False,
            original_text="",
            rewritten_text="",
            latency_ms=0,
            rewriter_key=rewriting_cfg.rewriter_model,
            target_key=target_model_key,
        )

    # 1. 根据 tier 选择模板类型
    template_type = rewriting_cfg.tier_template_map.get(cfg.tier, "small")

    # 2. 加载模板
    prompt_path = getattr(rewriting_cfg, f"rewrite_{template_type}_prompt", "")
    template = load_prompt(prompt_path, purpose="改写提示词")

    # 3. 如果 add_plan=True 且是大 tier 模板，追加 plan generation 段落
    if rewriting_cfg.add_plan and template_type == "large":
        plan_path = Path(__file__).parent / "prompts" / "plan_generation.md"
        try:
            template += plan_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Plan generation prompt not found: %s", plan_path)
        except Exception as e:
            logger.warning("Failed to load plan generation prompt: %s", e)

    # 4. 填充模板变量
    transforms_str = _format_transforms(rewriting_cfg.transforms)
    source = cfg.source
    if source == "auto":
        source = "local" if settings.is_local(target_model_key) else "cloud"

    from collections import defaultdict

    _template_vars = defaultdict(
        str,
        {
            "model_key": target_model_key,
            "model_description": cfg.description or "",
            "model_tier": cfg.tier,
            "source": source,
            "transforms": transforms_str,
        },
    )
    template = template.format_map(_template_vars)

    # 5. 找到最后一条 user message
    msg_idx = _find_last_user_message(messages)
    if msg_idx is None:
        return messages, RewriteMetadata(
            rewritten=False,
            original_text="",
            rewritten_text="",
            latency_ms=0,
            rewriter_key="",
            target_key=target_model_key,
        )

    # 5.5 检测 agent 自主执行，跳过改写
    if _has_agent_activity_after(messages, msg_idx):
        return messages, RewriteMetadata(
            rewritten=False,
            original_text="",
            rewritten_text="",
            latency_ms=0,
            rewriter_key=rewriting_cfg.rewriter_model,
            target_key=target_model_key,
        )

    original_text = _extract_user_text(messages[msg_idx].get("content", ""))
    if not original_text.strip():
        return messages, RewriteMetadata(
            rewritten=False,
            original_text="",
            rewritten_text="",
            latency_ms=0,
            rewriter_key="",
            target_key=target_model_key,
        )

    # 6. 获取改写器模型配置（构建上下文和调用都需要）
    rewriter_key = rewriting_cfg.rewriter_model
    rewriter_cfg = settings.get_model_config(rewriter_key)
    if rewriter_cfg is None:
        logger.warning("Rewriter model '%s' not found in config", rewriter_key)
        if rewriting_cfg.fallback_to_original:
            return messages, RewriteMetadata(
                rewritten=False,
                original_text=original_text,
                rewritten_text="",
                latency_ms=0,
                rewriter_key=rewriter_key,
                target_key=target_model_key,
            )
        raise ValueError(f"Rewriter model '{rewriter_key}' not found in config")

    # 7. 构建上下文并构造改写请求
    rewriter_source = rewriter_cfg.source
    if rewriter_source == "auto":
        rewriter_source = "local" if settings.is_local(rewriter_key) else "cloud"

    if rewriter_source == "local":
        ctx_rounds = rewriting_cfg.context_local_rounds
        ctx_max_chars = rewriting_cfg.context_local_max_chars
    else:
        ctx_rounds = rewriting_cfg.context_cloud_rounds
        ctx_max_chars = rewriting_cfg.context_cloud_max_chars

    context_str = _build_rewrite_context(messages, msg_idx, ctx_rounds, ctx_max_chars)

    api_messages = [{"role": "system", "content": template}]
    if context_str:
        api_messages.append({"role": "user", "content": f"Conversation context:\n{context_str}"})
    api_messages.append({"role": "user", "content": original_text})

    # 8. 调用 rewriter 模型
    start = time.time()
    rewritten_text = ""

    try:
        import asyncio

        if settings.is_local(rewriter_key):
            backend = get_backend(rewriter_cfg.backend or "", rewriter_cfg.base_url)

            async def _local_call():
                kwargs = {
                    "model": rewriter_cfg.model,
                    "messages": api_messages,
                    "stream": False,
                    "max_tokens": rewriting_cfg.max_rewriter_tokens,
                    "temperature": rewriting_cfg.temperature,
                    "top_p": rewriting_cfg.top_p,
                }
                if rewriting_cfg.thinking_enabled:
                    kwargs.update(rewriting_cfg.thinking_params)
                return await backend.chat_completions(**kwargs)

            if rewriting_cfg.rewriter_timeout_enabled:
                result = await asyncio.wait_for(
                    _local_call(),
                    timeout=rewriting_cfg.rewriter_timeout_seconds,
                )
            else:
                result = await _local_call()
            rewritten_text = _parse_chat_response(result)
        else:
            url = rewriter_cfg.base_url.rstrip("/")
            headers = {"Content-Type": "application/json"}
            if rewriter_cfg.api_key:
                headers["Authorization"] = f"Bearer {rewriter_cfg.api_key}"

            payload = {
                "model": rewriter_cfg.model,
                "messages": api_messages,
                "stream": False,
                "max_tokens": rewriting_cfg.max_rewriter_tokens,
                "temperature": rewriting_cfg.temperature,
                "top_p": rewriting_cfg.top_p,
            }
            # thinking 预算（如 API 支持，如 Qwen）
            if rewriting_cfg.thinking_enabled:
                payload.update(rewriting_cfg.thinking_params)

            client = get_cloud_client(
                timeout=rewriting_cfg.timeout_seconds,
                proxy=settings.server.proxy,
                verify=settings.server.verify_ssl,
            )

            async def _cloud_call():
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()

            if rewriting_cfg.rewriter_timeout_enabled:
                data = await asyncio.wait_for(
                    _cloud_call(),
                    timeout=rewriting_cfg.rewriter_timeout_seconds,
                )
            else:
                data = await _cloud_call()
            rewritten_text = _parse_chat_response(data)

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.warning("Rewrite failed (%.0fms): %s", latency_ms, e)
        breaker.record_failure()
        if rewriting_cfg.fallback_to_original:
            return messages, RewriteMetadata(
                rewritten=False,
                original_text=original_text,
                rewritten_text="",
                latency_ms=latency_ms,
                rewriter_key=rewriter_key,
                target_key=target_model_key,
            )
        raise

    latency_ms = (time.time() - start) * 1000
    rewritten_text = rewritten_text.strip()

    if not rewritten_text:
        logger.warning("Rewriter returned empty output, using original")
        return messages, RewriteMetadata(
            rewritten=False,
            original_text=original_text,
            rewritten_text="",
            latency_ms=latency_ms,
            rewriter_key=rewriter_key,
            target_key=target_model_key,
        )

    # 8. 替换最后一条 user message 的 content
    new_messages = [dict(m) for m in messages]
    original_content = new_messages[msg_idx].get("content", "")
    if isinstance(original_content, str):
        new_messages[msg_idx]["content"] = rewritten_text
    elif isinstance(original_content, list):
        # 替换第一个 text part，保留其他 parts（如 image）
        new_parts = []
        replaced = False
        for part in original_content:
            if isinstance(part, dict) and part.get("type") == "text" and not replaced:
                new_parts.append({"type": "text", "text": rewritten_text})
                replaced = True
            else:
                new_parts.append(part)
        if not replaced:
            new_parts.append({"type": "text", "text": rewritten_text})
        new_messages[msg_idx]["content"] = new_parts
    else:
        new_messages[msg_idx]["content"] = rewritten_text

    # 9. 仅在用户明确启用时写入包含提示词内容的日志文件
    if rewriting_cfg.write_rewrite_logs:
        await _write_rewrite_log(
            original_text, rewritten_text, rewriter_key, target_model_key, latency_ms, rewriting_cfg.rewrite_log_dir
        )

    breaker.record_success()

    return new_messages, RewriteMetadata(
        rewritten=True,
        original_text=original_text,
        rewritten_text=rewritten_text,
        latency_ms=latency_ms,
        rewriter_key=rewriter_key,
        target_key=target_model_key,
    )


async def _write_rewrite_log(
    original: str,
    rewritten: str,
    rewriter_key: str,
    target_key: str,
    latency_ms: float,
    log_dir: str = "logs/rewrites",
) -> None:
    """将改写前后文本追加到当天的日志文件中（异步写入，避免阻塞事件循环）"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    filepath = log_path / f"{time.strftime('%Y%m%d')}.log"

    content = (
        f"[{ts}] rewriter: {rewriter_key} | target: {target_key} | latency: {latency_ms:.0f}ms\n"
        f"[Original]\n{original}\n"
        f"[Rewritten]\n{rewritten}\n"
        f"{'─' * 50}\n"
    )

    def _write():
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content)

    await asyncio.to_thread(_write)
