"""分类器：判断请求复杂度并推荐模型

分类方式:
- local: 调用本地模型（多轮上下文，~50-200ms，零成本）
- cloud: 调用云端模型（多轮上下文，200-500ms，有网络延迟）
- rules: 启发式规则（模型不可用时的 fallback）

设计原则:
- 本地模型传 3 轮上下文（每轮 300 字符），适合小模型快速判断
- 云端模型传 6 轮上下文（每轮 900 字符），分类更精准
- 启发式规则基于文本长度，作为兜底
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prism_router.context import RouteDecision
from prism_router.prompts import load_prompt
from prism_router.status import record_classifier_failure, record_classifier_success

if TYPE_CHECKING:
    from prism_router.local_backends.base import LocalBackend
    from prism_router.settings import ModelConfig, RoutingConfig, Settings

logger = logging.getLogger("prism_router.classifier")

# 预编译正则（避免每次调用都编译）
_TIER_PATTERN = re.compile(r"\b(simple|mid|complex)\b", re.IGNORECASE)


def _extract_prompt_block(content: str, index: int = 0) -> str:
    """从 markdown 文件中提取第 index 个代码块的内容"""
    blocks = re.findall(r"```\n(.*?)```", content, re.DOTALL)
    if index < len(blocks):
        return str(blocks[index]).strip()
    return ""


# 默认分类器提示词路径（惰性加载）
_DEFAULT_CLASSIFY_PATH = "prism_router/prompts/classify.md"
_classify_prompt_cache: dict[str, tuple[str, str]] = {}


def _get_classify_prompts(classify_prompt: str = "") -> tuple[str, str]:
    """获取分类器提示词，支持自定义路径（惰性加载 + 缓存）"""
    path = classify_prompt or _DEFAULT_CLASSIFY_PATH
    if path not in _classify_prompt_cache:
        raw = load_prompt(path, purpose="分类器提示词")
        _classify_prompt_cache[path] = (
            _extract_prompt_block(raw, 0),
            _extract_prompt_block(raw, 1),
        )
    return _classify_prompt_cache[path]


def _format_history(history: list[RouteDecision]) -> str:
    """格式化历史记录为 prompt 可读的文本"""
    if not history:
        return "No previous context."

    lines = []
    for i, h in enumerate(history[-3:], 1):  # 只显示最近3条
        status = "[ok]" if h.success else "[fail]"
        tools_str = " [tools]" if h.has_tools else ""
        lines.append(f"{i}. [{h.tier}] {h.user_msg}... ({status}){tools_str}")

    return "\n".join(lines)


def _extract_content_text(content: str | list | Any) -> str:
    """从 OpenAI 格式的 content 中提取纯文本（支持 string 和 list 格式）"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "tool_use":
                    parts.append(f"[tool call: {part.get('name', 'unknown')}]")
                elif part.get("type") == "tool_result":
                    tool_content = part.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = " ".join(
                            p.get("text", "") for p in tool_content if isinstance(p, dict) and p.get("type") == "text"
                        )
                    parts.append(f"[tool result] {tool_content}")
        return " ".join(parts)
    return str(content)


def _extract_user_text(messages: list[dict]) -> str:
    """提取最后一条 user message 的文本（用于校验和 fallback）"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return " ".join(parts)
            break
    return ""


def _build_context(messages: list[dict], max_chars: int = 300, actual_tool_use: bool = False) -> str:
    """构建分类器的多轮对话上下文（只传 user/assistant/tool 消息，排除 system）

    从传入的 messages 中提取非空消息，每轮格式: role: content
    调用方负责切片（messages[-max_rounds:]），本函数只负责格式化和截断。

    actual_tool_use: messages 中是否有实际 tool 调用证据（role:tool 或 assistant.tool_calls）。
    为 True 时注入提示，让分类器知道当前请求正在使用工具。
    """
    rows = []
    for msg in messages:
        role = msg.get("role", "")
        # 排除 system 消息 — agent 的系统提示词对分类无用，且会干扰分类器
        if role == "system":
            continue
        content = msg.get("content", "")
        text = _extract_content_text(content)
        text = text.strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        rows.append(f"{role}: {text}")
    # 仅当 messages 中有实际 tool 调用证据时才注入提示
    # 分类器会根据 [supports_tools] 标签选择支持工具的模型
    if actual_tool_use:
        rows.append("[system] This conversation uses tools — pick a [supports_tools] model")
    return "\n".join(rows) if rows else "No prior context."


@dataclass
class ClassificationResult:
    tier: str  # "simple" / "mid" / "complex"
    score: float  # 1.0 = confident, 0.0 = fallback
    signals: dict  # 保留兼容性，模型分类时为空
    recommended_model: str = ""  # 分类器推荐的模型 key


def build_model_descriptions(
    models_config: dict, model_pool: list[str] | None = None, routing: RoutingConfig | None = None
) -> str:
    """从 models 配置生成模型描述，带给分类器。只展示 pool 里的模型。

    Args:
        routing: RoutingConfig 对象（可选），传入时使用 routing 层的 tier 而非 cfg.tier
    """
    lines = []
    for key, cfg in models_config.items():
        # 只展示 pool 里的模型（如果指定了 pool）
        if model_pool and key not in model_pool:
            continue
        if hasattr(cfg, "description") and cfg.description:
            tags = list(cfg.tags) if hasattr(cfg, "tags") and cfg.tags else []
            if cfg.supports_tools:
                tags.append("supports_tools")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            # 优先使用 routing 层的 tier（决定实际路由），fallback 到 cfg.tier
            tier = cfg.tier or "mid"
            if routing is not None:
                from prism_router.routing import _get_routing_tier_for_model

                tier = _get_routing_tier_for_model(key, routing)
            lines.append(f"- {key} (tier:{tier}): {cfg.description}{tag_str}")
    return "\n".join(lines) if lines else "No model descriptions available."


def _extract_model_key(text: str, model_keys: list[str]) -> str | None:
    """从模型输出中精确提取模型 key（用正则边界匹配，避免子串误匹配）"""
    for model_key in model_keys:
        # 精确匹配：模型 key 作为独立词出现（被引号、空格、换行包围）
        pattern = (
            r'[\s"\'`/]('
            + re.escape(model_key)
            + r')[\s"\'`/\n]|^('
            + re.escape(model_key)
            + r')[\s"\'`/\n]|[\s"\'`/]('
            + re.escape(model_key)
            + r")$|^("
            + re.escape(model_key)
            + r")$"
        )
        if re.search(pattern, text):
            return model_key
    return None


async def classify_request(
    messages: list[dict],
    backend: LocalBackend | None = None,
    model_id: str = "",
    models_config: dict | None = None,
    model_pool: list[str] | None = None,
    conversation_history: list[RouteDecision] | None = None,
    max_rounds: int = 3,
    max_chars: int = 300,
    max_tokens: int = 256,
    actual_tool_use: bool = False,
    tools_tier: str = "mid",
    effective_tools: bool = False,
    routing: RoutingConfig | None = None,
    classifier_failure_threshold: int = 3,
    classify_prompt: str = "",
    settings: Settings | None = None,
) -> ClassificationResult:
    """
    用本地小模型对请求进行分类并推荐模型。

    Args:
        messages: OpenAI 格式的消息列表
        backend: 本地推理后端（OllamaBackend / LlamaCppBackend），不传则走启发式
        model_id: 本地模型的 model ID
        models_config: models 配置（用于生成模型描述）
        model_pool: 参与路由的模型列表（只推荐这些模型）
        conversation_history: 会话历史路由决策（用于上下文感知分类）
        max_rounds: 传给分类器的对话轮数
        max_chars: 每轮截断字符数
        actual_tool_use: messages 中是否有实际 tool 调用证据
        tools_tier: 支持 tool calling 的 tier（由配置决定）
        effective_tools: 是否应因 tools 而升级 tier（受 tools_policy 控制）
        settings: Settings 对象（用于读取超时配置）
    """
    # 提取最后一条 user message（仅用于校验和 fallback）
    user_text = _extract_user_text(messages)

    if not user_text:
        return _fallback("", tools_tier=tools_tier, effective_tools=effective_tools, actual_tool_use=actual_tool_use)

    # 尝试用本地模型分类（瞬时错误重试）
    if backend and model_id:
        import asyncio

        max_retries = getattr(routing, "classifier_max_retries", 3) if routing else 3

        async def _retry_loop() -> ClassificationResult | None:
            for attempt in range(max_retries):
                try:
                    result = await _classify_with_model(
                        backend,
                        model_id,
                        messages,
                        models_config,
                        model_pool,
                        conversation_history,
                        max_rounds=max_rounds,
                        max_chars=max_chars,
                        max_tokens=max_tokens,
                        actual_tool_use=actual_tool_use,
                        routing=routing,
                        classify_prompt=classify_prompt,
                    )
                    if result:
                        return _validate_tier(result, user_text)
                except (ConnectionError, TimeoutError, OSError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Model classification attempt %d/%d failed, retrying: %s", attempt + 1, max_retries, e
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        logger.warning("Model classification failed after %d attempts: %s", max_retries, e)
                        raise
                except Exception as e:
                    logger.warning("Model classification failed (non-retryable): %s", e)
                    raise
            return None

        try:
            # 总超时覆盖整个重试循环，防止深度思考模型无限挂起
            if settings and settings.routing.classifier_timeout_enabled:
                result = await asyncio.wait_for(
                    _retry_loop(),
                    timeout=settings.routing.classifier_timeout_seconds,
                )
            else:
                result = await _retry_loop()
            if result:
                record_classifier_success()
                return result
            else:
                record_classifier_failure(classifier_failure_threshold)
        except asyncio.TimeoutError:
            logger.warning(
                "Classifier total timeout (%ds)", settings.routing.classifier_timeout_seconds if settings else 15
            )
            record_classifier_failure(classifier_failure_threshold)
        except Exception:
            record_classifier_failure(classifier_failure_threshold)

    # Fallback: 启发式（文本长度 + tools 检测）
    return _fallback(user_text, tools_tier=tools_tier, effective_tools=effective_tools, actual_tool_use=actual_tool_use)


async def _classify_with_model(
    backend: LocalBackend,
    model_id: str,
    messages: list[dict],
    models_config: dict | None = None,
    model_pool: list[str] | None = None,
    conversation_history: list[RouteDecision] | None = None,
    max_rounds: int = 3,
    max_chars: int = 300,
    max_tokens: int = 256,
    actual_tool_use: bool = False,
    routing: RoutingConfig | None = None,
    classify_prompt: str = "",
) -> ClassificationResult | None:
    """调用本地模型分类，注入模型描述辅助决策"""
    # 生成模型描述
    model_desc = ""
    candidate_keys = list(model_pool) if model_pool else []
    if models_config and candidate_keys:
        model_desc = build_model_descriptions(models_config, model_pool, routing=routing)

    # 选择 prompt 并注入变量（支持自定义路径）
    no_history, with_history = _get_classify_prompts(classify_prompt)
    if conversation_history:
        history_text = _format_history(conversation_history)
        prompt = with_history.format(
            model_descriptions=model_desc,
            history_context=history_text,
        )
    else:
        prompt = no_history.format(
            model_descriptions=model_desc,
        )

    # 构建多轮对话上下文
    last_n = messages[-max_rounds:] if len(messages) >= max_rounds else messages
    context = _build_context(last_n, max_chars=max_chars, actual_tool_use=actual_tool_use)

    api_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    logger.debug("Classifying with %s, context=%r", model_id, context[:200])

    result = await backend.chat_completions(
        model=model_id,
        messages=api_messages,
        stream=False,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    if isinstance(result, dict):
        choices = result.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
    else:
        content = str(result)
        reasoning = ""

    # 解析输出：优先从 content，fallback 到 reasoning_content
    for text in (content, reasoning):
        text = text.strip()
        parsed = _parse_classification_output(text, candidate_keys, models_config)
        if parsed:
            logger.info("Classification result: tier=%s, model=%s", parsed.tier, parsed.recommended_model or "auto")
            return parsed

    logger.warning("Classifier returned unparseable output: content=%r, reasoning=%r", content[:200], reasoning[:200])
    return None


def _parse_classification_output(
    text: str, candidate_keys: list[str], models_config: dict | None = None
) -> ClassificationResult | None:
    """解析分类器输出，支持新格式 (tier:/model:) 和旧格式 (纯 tier 关键词)"""
    tier = None
    model_key = None

    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("tier:"):
            t = line.split(":", 1)[1].strip()
            if t in ("simple", "mid", "complex"):
                tier = t
        elif line.startswith("model:"):
            m = line.split(":", 1)[1].strip().strip('"').strip("'")
            if candidate_keys and m in candidate_keys:
                model_key = m

    # 有推荐模型 → 直接用
    if model_key:
        if not tier:
            tier = _get_tier_for_model(model_key, models_config)
        return ClassificationResult(
            tier=tier,
            score=1.0,
            signals={"method": "model"},
            recommended_model=model_key,
        )

    # 只有 tier → 兼容旧行为
    if tier:
        return ClassificationResult(tier=tier, score=1.0, signals={"method": "model"})

    # 兜底：从全文匹配 tier 关键词
    match = _TIER_PATTERN.search(text)
    if match:
        return ClassificationResult(tier=match.group(1).lower(), score=1.0, signals={"method": "model"})

    return None


def _get_tier_for_model(model_key: str, models_config: dict | None = None) -> str:
    """获取模型的默认 tier（仅用于分类器内部，最终 tier 由 routing 层决定）"""
    if models_config:
        cfg = models_config.get(model_key)
        if cfg and hasattr(cfg, "tier") and cfg.tier:
            return str(cfg.tier)
    return "mid"


def _validate_tier(result: ClassificationResult, user_text: str) -> ClassificationResult:
    """启发式校验：防止明显错误"""
    length = len(user_text.strip())

    # 模型说 simple 但消息很长 → 可能不准，改为 mid
    if result.tier == "simple" and length > 200:
        logger.info("Correcting tier from simple to mid (long text: %d chars)", length)
        result.tier = "mid"
        result.signals["corrected"] = "long_text_but_simple"

    # 模型说 complex 但消息很短 → 可能不准，改为 mid
    if result.tier == "complex" and length < 30:
        logger.info("Correcting tier from complex to mid (short text: %d chars)", length)
        result.tier = "mid"
        result.signals["corrected"] = "short_text_but_complex"

    return result


def _fallback(
    user_text: str, tools_tier: str = "mid", effective_tools: bool = False, actual_tool_use: bool = False
) -> ClassificationResult:
    """启发式 fallback — 文本内容 + tools 检测

    Args:
        tools_tier: 支持 tool calling 的 tier（由配置决定，非硬编码）
        effective_tools: 是否应因 tools 而升级 tier（受 tools_policy 控制）
        actual_tool_use: 消息中是否实际包含 tool 调用证据
    """
    # 有实际 tool 调用 → 路由到支持 tool calling 的 tier
    if (effective_tools or actual_tool_use) and tools_tier:
        return ClassificationResult(tier=tools_tier, score=0.0, signals={"method": "heuristic", "reason": "has_tools"})

    text = user_text.lower().strip()
    length = len(text)

    # 短文本（< 50字符）→ simple
    if length < 50:
        return ClassificationResult(tier="simple", score=0.0, signals={"method": "heuristic"})

    # 长文本 → complex（真正的复杂任务通常消息较长）
    if length > 500:
        return ClassificationResult(tier="complex", score=0.0, signals={"method": "heuristic"})

    # 中等长度 → mid
    return ClassificationResult(tier="mid", score=0.0, signals={"method": "heuristic"})


async def classify_request_cloud(
    messages: list[dict],
    model_config: ModelConfig,
    models_config: dict | None = None,
    model_pool: list[str] | None = None,
    conversation_history: list[RouteDecision] | None = None,
    max_rounds: int = 6,
    max_chars: int = 900,
    max_tokens: int = 256,
    actual_tool_use: bool = False,
    tools_tier: str = "mid",
    effective_tools: bool = False,
    routing: RoutingConfig | None = None,
    classifier_failure_threshold: int = 3,
    classify_prompt: str = "",
    settings: Settings | None = None,
) -> ClassificationResult:
    """用云端模型分类（httpx 直连，多轮对话上下文）"""
    user_text = _extract_user_text(messages)
    if not user_text:
        return _fallback("", tools_tier=tools_tier, effective_tools=effective_tools, actual_tool_use=actual_tool_use)

    try:
        import asyncio

        max_retries = getattr(routing, "classifier_max_retries", 3) if routing else 3

        async def _cloud_call():
            return await _classify_with_cloud(
                model_config,
                messages,
                models_config,
                model_pool,
                conversation_history,
                max_rounds=max_rounds,
                max_chars=max_chars,
                max_tokens=max_tokens,
                actual_tool_use=actual_tool_use,
                routing=routing,
                classify_prompt=classify_prompt,
                settings=settings,
            )

        for attempt in range(max_retries):
            try:
                # 总超时覆盖云端分类器调用
                if settings and settings.routing.classifier_timeout_enabled:
                    result = await asyncio.wait_for(
                        _cloud_call(),
                        timeout=settings.routing.classifier_timeout_seconds,
                    )
                else:
                    result = await _cloud_call()

                if result:
                    result = _validate_tier(result, user_text)
                    if effective_tools and tools_tier and result.tier != tools_tier:
                        result.tier = tools_tier
                        result.signals["corrected"] = "tools_required"
                    return result
                break
            except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Cloud classification attempt %d/%d failed, retrying: %s", attempt + 1, max_retries, e
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.warning("Cloud classification failed after %d attempts: %s", max_retries, e)
                    record_classifier_failure(classifier_failure_threshold)
            except Exception as e:
                logger.warning("Cloud classification failed (non-retryable): %s", e)
                record_classifier_failure(classifier_failure_threshold)
                break
    except Exception as e:
        logger.warning("Cloud classification setup failed: %s", e)
        record_classifier_failure(classifier_failure_threshold)

    return _fallback(user_text, tools_tier=tools_tier, effective_tools=effective_tools, actual_tool_use=actual_tool_use)


async def _classify_with_cloud(
    model_config,
    messages: list[dict],
    models_config: dict | None = None,
    model_pool: list[str] | None = None,
    conversation_history: list[RouteDecision] | None = None,
    max_rounds: int = 6,
    max_chars: int = 900,
    max_tokens: int = 256,
    actual_tool_use: bool = False,
    routing=None,
    classify_prompt: str = "",
    settings=None,
) -> ClassificationResult | None:
    """调用云端模型分类（多轮对话上下文）"""
    candidate_keys = list(model_pool) if model_pool else []
    model_desc = ""
    if models_config and candidate_keys:
        model_desc = build_model_descriptions(models_config, model_pool, routing=routing)

    # system prompt（支持自定义路径）
    no_history, with_history = _get_classify_prompts(classify_prompt)
    if conversation_history:
        history_text = _format_history(conversation_history)
        system_prompt = with_history.format(model_descriptions=model_desc, history_context=history_text)
    else:
        system_prompt = no_history.format(model_descriptions=model_desc)

    # user message：多轮对话上下文
    last_n = messages[-max_rounds:] if len(messages) >= max_rounds else messages
    context = _build_context(last_n, max_chars=max_chars, actual_tool_use=actual_tool_use)

    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]

    from prism_router.cloud import build_cloud_url
    from prism_router.local_backends.pool import get_cloud_client
    from prism_router.server import get_settings

    url = build_cloud_url(model_config.base_url)
    headers = {"Content-Type": "application/json"}
    if model_config.api_key:
        headers["Authorization"] = f"Bearer {model_config.api_key}"

    _settings = settings or get_settings()
    _verify = _settings.server.verify_ssl
    client = get_cloud_client(timeout=_settings.routing.classifier_timeout_seconds, verify=_verify)
    resp = await client.post(
        url,
        json={
            "model": model_config.model,
            "messages": api_messages,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    # 解析响应（与本地分类器相同）
    if isinstance(data, dict):
        choices = data.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
    else:
        content = str(data)
        reasoning = ""

    for text in (content, reasoning):
        text = text.strip()
        result = _parse_classification_output(text, candidate_keys, models_config)
        if result:
            return result

    logger.warning("Cloud model returned unexpected classification")
    return None
