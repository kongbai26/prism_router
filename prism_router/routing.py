"""路由逻辑：解析 model 字段 → 分类 → 渠道选择（优先级 + 权重 + 熔断器）"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from prism_router.channel import select_channel
from prism_router.classifier import ClassificationResult, classify_request, classify_request_cloud
from prism_router.health import ChannelStatus, CircuitBreaker, CircuitBreakerConfig
from prism_router.settings import (
    TIER_FALLBACK,
    ChannelEntry,
    ModelConfig,
    RoutingConfig,
    Settings,
    model_key_to_channel_id,
)

logger = logging.getLogger("prism_router.routing")

# 强制 tier 的特殊 model 名
TIER_MODEL_MAP = {
    "prism-simple": "simple",
    "prism-mid": "mid",
    "prism-complex": "complex",
}


def resolve_model_alias(model: str, settings: Settings) -> str:
    """解析模型别名：优先级 tier 别名 > 用户自定义别名 > 原样返回"""
    if model in TIER_MODEL_MAP:
        return model
    # 大小写不敏感匹配别名
    aliases = settings.routing.aliases
    if model in aliases:
        return aliases[model]
    model_lower = model.lower()
    for alias_key, alias_val in aliases.items():
        if alias_key.lower() == model_lower:
            return alias_val
    return model


def estimate_tokens(messages: list[dict], tools: list | None = None) -> int:
    """粗估请求的 token 数（~3.5 chars/token，更保守的估算）"""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
    if tools:
        # Tools 的 JSON schema 结构复杂，token 化效率低，按 3 chars/token 估算
        total_chars += int(len(json.dumps(tools, ensure_ascii=False)) * 1.2)
    # 使用更保守的 3.5 chars/token（原 4 chars/token 偏乐观）
    return int(total_chars / 3.5) + 50  # +50 作为安全余量


def check_context_window(messages: list[dict], tools: list | None, model_config: ModelConfig | None) -> bool:
    """检查请求是否超过模型上下文窗口，返回 True 表示在范围内"""
    if model_config is None:
        return True
    estimated = estimate_tokens(messages, tools)
    return estimated <= model_config.context


@dataclass
class RouteResult:
    model_key: str  # 配置中的 key，如 "llamacpp/qwen2.5-1.5b"
    model_id: str  # 发给 API 的实际 model ID
    tier: str
    base_url: str
    api_key: str
    is_local: bool
    classification: ClassificationResult | None
    model_config: ModelConfig | None
    # 新增：动态路由
    channel_id: int = 0  # 选中的渠道 ID（用于健康度追踪）
    attempt: int = 0  # 第几次尝试
    fallback_from: str = ""  # 从哪个渠道 fallback 过来的
    use_tools: bool = False  # 是否传 tools 给模型（由 tools_policy + 模型 supports_tools 决定）


# 全局熔断器实例（容器列表，避免 global）
_circuit_breaker: list[CircuitBreaker | None] = [None]


def reset_circuit_breaker():
    """重置全局熔断器实例（用于测试）"""
    _circuit_breaker[0] = None


def get_circuit_breaker(settings: Settings | None = None) -> CircuitBreaker:
    if _circuit_breaker[0] is None:
        config = CircuitBreakerConfig(
            enabled=settings.circuit_breaker.enabled if settings else True,
            failure_threshold=settings.circuit_breaker.failure_threshold if settings else 2,
            local_failure_threshold=settings.circuit_breaker.local_failure_threshold if settings else 3,
            success_threshold=settings.circuit_breaker.success_threshold if settings else 1,
            cooldown_base=settings.circuit_breaker.cooldown_base if settings else 30,
            cooldown_max=settings.circuit_breaker.cooldown_max if settings else 3000,
            jitter=settings.circuit_breaker.jitter if settings else 0.2,
            window_size=settings.circuit_breaker.window_size if settings else 20,
            slow_response_enabled=settings.circuit_breaker.slow_response_enabled if settings else True,
            slow_response_threshold_ms=settings.circuit_breaker.slow_response_threshold_ms if settings else 15000,
            slow_response_failure_threshold=settings.circuit_breaker.slow_response_failure_threshold if settings else 3,
        )
        _circuit_breaker[0] = CircuitBreaker(config)
    cb = _circuit_breaker[0]
    assert cb is not None
    return cb


def _resolve_tier(tier: str, routing: RoutingConfig) -> tuple[str, str]:
    """
    把 tier 名转为 (tier, model_key)。
    优先从 routing.tiers 读取（auto 模式写入），fallback 到 simple/mid/complex 字符串字段。
    """
    # 1. 优先从 routing.tiers 读取
    if routing.tiers and tier in routing.tiers:
        entries = routing.tiers[tier]
        if entries:
            return tier, entries[0].model

    # 2. Fallback 到 simple/mid/complex 字符串字段
    key = getattr(routing, tier, "")
    if key:
        return tier, key

    # 3. 都没有 → 按 fallback 链查找
    for fallback_tier in TIER_FALLBACK.get(tier, []):
        if routing.tiers and fallback_tier in routing.tiers:
            entries = routing.tiers[fallback_tier]
            if entries:
                return fallback_tier, entries[0].model
        fb_key = getattr(routing, fallback_tier, "")
        if fb_key:
            return fallback_tier, fb_key

    # 4. 所有 tier 都为空 — 尝试从任意非空 tier 中找一个
    if routing.tiers:
        for t, entries in routing.tiers.items():
            if entries:
                logger.warning("Tier '%s' is empty, falling back to '%s'", tier, t)
                return t, entries[0].model

    logger.warning("No model available for tier '%s' or its fallback chain", tier)
    raise ValueError(f"No model available for tier '{tier}' or its fallback chain. Check routing config.")


def _get_routing_tier_for_model(model_key: str, routing: RoutingConfig) -> str:
    """从 routing 配置中查找模型所在的 tier（优先于模型默认 tier）"""
    # 1. 从 routing.tiers 查找（大小写不敏感）
    if routing.tiers:
        for tier, entries in routing.tiers.items():
            for entry in entries:
                if entry.model.lower() == model_key.lower():
                    return tier
    # 2. 从 simple/mid/complex 字符串字段查找（大小写不敏感）
    for tier in ("simple", "mid", "complex"):
        if getattr(routing, tier, "").lower() == model_key.lower():
            return tier
    # 3. 都找不到，返回 mid 作为默认
    logger.warning("Model '%s' not found in any tier, defaulting to 'mid'", model_key)
    return "mid"


def detect_actual_tool_use(messages: list[dict]) -> bool:
    """检测当前轮对话是否正在使用工具

    从后往前扫描：
    1. 先找最后一条 assistant 消息
    2. 如果 assistant 有 tool_calls → 当前轮正在 tool-calling（等待 tool 结果）
    3. 如果 assistant 是普通文本 → tool-calling 已结束
    4. 如果最后一条不是 assistant（是 tool 或 user）→ 看后续逻辑
    """
    if not messages:
        return False
    for msg in reversed(messages):
        if msg.get("role") == "user":
            # user 消息标志着新轮开始，之前的 tool-calling 已结束
            return False
        if msg.get("role") == "assistant":
            # 最后一条 assistant 有 tool_calls → 当前轮正在 tool-calling
            return bool(msg.get("tool_calls") or msg.get("function_call"))
        if msg.get("role") in ("tool", "function"):
            # tool 消息在 assistant 之前出现 → 继续找 assistant
            continue
    return False


async def route(
    model: str | None,
    messages: list[dict],
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
    conversation_history: list | None = None,
) -> RouteResult:
    """
    路由请求到目标模型。

    model 字段:
      - None / "prism-auto" → 自动分类
      - "prism-simple" / "prism-mid" / "prism-complex" → 强制 tier
      - 用户别名（如 "fast"）→ 解析后走对应逻辑
      - "provider/model" → 直接指定（透传）
    """
    if settings is None:
        from prism_router.settings import load_settings

        settings = load_settings()

    # ── 0. 模型别名解析 ──
    if model:
        model = resolve_model_alias(model, settings)

    routing = settings.routing

    # ── 提前计算 tools 策略（所有路径共用） ──
    tools_policy = routing.tools_policy
    tools_enabled = getattr(routing, "tools_enabled", True)
    has_tools = bool(tools)

    # 检测 messages 中是否有 tool-calling 证据（即使 body 无 tools）
    _has_tool_evidence = False
    if not has_tools and messages:
        for msg in messages:
            role = msg.get("role", "")
            if role in ("tool", "function"):
                _has_tool_evidence = True
                break
            if role == "assistant" and (msg.get("tool_calls") or msg.get("function_call")):
                _has_tool_evidence = True
                break

    # tools 透传：body 有 tools 就传给模型（never 模式除外）
    # 这是代理行为 — tools 是客户端请求的一部分，应原样透传
    pass_tools = has_tools and tools_policy != "never"

    # tier 升级和分类器提示：受 tools_enabled 开关控制
    if not tools_enabled:
        upgrade_tier = False
        actual_tool_use = False
        tools_tier = ""
    else:
        # tier 升级：由 tools_policy 控制（成本优化）
        if tools_policy == "always":
            upgrade_tier = has_tools
        elif tools_policy == "auto":
            upgrade_tier = has_tools and detect_actual_tool_use(messages)
        else:  # "never"
            upgrade_tier = False

        # 分类器提示：messages 有 tool 调用证据时提示选支持 tools 的模型
        if has_tools:
            actual_tool_use = tools_policy == "always" or detect_actual_tool_use(messages)
        else:
            actual_tool_use = _has_tool_evidence

        tools_tier = settings.find_tools_tier()

    # ── 1. 强制指定 tier ──
    if model in TIER_MODEL_MAP:
        tier = TIER_MODEL_MAP[model]
        tier, key = _resolve_tier(tier, routing)
        entry = select_channel(tier, settings)
        if entry:
            res = _build_result_from_entry(
                entry, tier, settings, None, pass_tools=pass_tools, upgrade_tier=upgrade_tier
            )
        else:
            res = _build_result(key, tier, settings, None, pass_tools=pass_tools, upgrade_tier=upgrade_tier)
        return _apply_circuit_breaker_fallback(res, settings)

    # ── 2. 直接指定模型（透传） ──
    if model and not model.startswith("prism-"):
        cfg = settings.get_model_config(model)
        if cfg:
            use_tools = pass_tools and cfg.supports_tools
            return RouteResult(
                model_key=model,
                model_id=cfg.model,
                tier=cfg.tier,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                is_local=settings.is_local(model),
                classification=None,
                model_config=cfg,
                channel_id=model_key_to_channel_id(model),
                use_tools=use_tools,
            )
        # 未在 models 中定义 → 继续走分类路由（不丢弃，分类器会判断复杂度）
        logger.debug("Model %r not in config, classifying request", model)

    # ── 3. 自动分类 ──
    # 根据 routing mode 构建候选列表
    if routing.mode == "manual":
        candidate_models = []
        # 从 routing.tiers 收集模型（新格式，多渠道）
        if routing.tiers:
            for tier_entries in routing.tiers.values():
                for entry in tier_entries:
                    if entry.model and entry.model not in candidate_models:
                        candidate_models.append(entry.model)
        # 兼容旧格式（simple/mid/complex 字符串字段）
        for tier in ("simple", "mid", "complex"):
            key = getattr(routing, tier, "")
            if key and key not in candidate_models:
                candidate_models.append(key)
        model_pool_for_classifier = candidate_models if candidate_models else None
    else:
        model_pool_for_classifier = list(routing.model_pool) if routing.model_pool else None

    if routing.classifier in ("local", "cloud") and routing.classifier_model:
        classifier_cfg = settings.get_model_config(routing.classifier_model)
        if classifier_cfg:
            if routing.classifier == "local":
                from prism_router.status import is_classifier_ok

                if not is_classifier_ok():
                    logger.warning("Classifier backend unavailable, falling back to heuristic rules")
                    result = await classify_request(
                        messages=messages,
                        models_config=settings.models,
                        model_pool=model_pool_for_classifier,
                        actual_tool_use=actual_tool_use,
                        tools_tier=tools_tier,
                        effective_tools=upgrade_tier,
                        routing=routing,
                        classifier_failure_threshold=routing.classifier_failure_threshold,
                        classify_prompt=routing.classify_prompt,
                        settings=settings,
                    )
                else:
                    logger.debug("Classifying with local model: %s", routing.classifier_model)
                    from prism_router.local_backends import get_backend

                    backend = get_backend(classifier_cfg.backend or "", classifier_cfg.base_url)
                    result = await classify_request(
                        messages=messages,
                        backend=backend,
                        model_id=classifier_cfg.model,
                        models_config=settings.models,
                        model_pool=model_pool_for_classifier,
                        conversation_history=conversation_history,
                        max_rounds=routing.classifier_local_rounds,
                        max_chars=routing.classifier_local_max_chars,
                        max_tokens=routing.classifier_local_max_tokens,
                        actual_tool_use=actual_tool_use,
                        tools_tier=tools_tier,
                        effective_tools=upgrade_tier,
                        routing=routing,
                        classifier_failure_threshold=routing.classifier_failure_threshold,
                        classify_prompt=routing.classify_prompt,
                        settings=settings,
                    )
            else:  # cloud
                logger.debug("Classifying with cloud model: %s", routing.classifier_model)
                result = await classify_request_cloud(
                    messages=messages,
                    model_config=classifier_cfg,
                    models_config=settings.models,
                    model_pool=model_pool_for_classifier,
                    conversation_history=conversation_history,
                    max_rounds=routing.classifier_cloud_rounds,
                    max_chars=routing.classifier_cloud_max_chars,
                    max_tokens=routing.classifier_cloud_max_tokens,
                    actual_tool_use=actual_tool_use,
                    tools_tier=tools_tier,
                    effective_tools=upgrade_tier,
                    routing=routing,
                    classifier_failure_threshold=routing.classifier_failure_threshold,
                    classify_prompt=routing.classify_prompt,
                    settings=settings,
                )
        else:
            result = await classify_request(
                messages=messages,
                models_config=settings.models,
                model_pool=model_pool_for_classifier,
                actual_tool_use=actual_tool_use,
                tools_tier=tools_tier,
                effective_tools=upgrade_tier,
                routing=routing,
                classifier_failure_threshold=routing.classifier_failure_threshold,
                classify_prompt=routing.classify_prompt,
                settings=settings,
            )
    else:
        result = await classify_request(
            messages=messages,
            models_config=settings.models,
            model_pool=model_pool_for_classifier,
            actual_tool_use=actual_tool_use,
            tools_tier=tools_tier,
            effective_tools=upgrade_tier,
            routing=routing,
            classifier_failure_threshold=routing.classifier_failure_threshold,
            classify_prompt=routing.classify_prompt,
            settings=settings,
        )

    # 安全网：需要 tools 但分类器推荐的模型不支持 → 升级到 tools_tier
    # upgrade_tier=True 时（always 有 tools / auto 有实际调用），推荐模型必须支持 tools
    if upgrade_tier and result.recommended_model and tools_tier:
        cfg = settings.get_model_config(result.recommended_model)
        if cfg and not cfg.supports_tools:
            logger.info(
                "Tools in body but %s doesn't support tools, upgrading to tier=%s",
                result.recommended_model,
                tools_tier,
            )
            result.tier = tools_tier
            result.recommended_model = ""
            result.signals["corrected"] = "tools_upgrade"

    # 优先使用分类器推荐的模型
    if result.recommended_model:
        cfg = settings.get_model_config(result.recommended_model)
        if cfg:
            resolved_tier = _get_routing_tier_for_model(result.recommended_model, routing)
            # 检查熔断状态：如果推荐模型可用（处于 ACTIVE 状态，非熔断冷却或禁用），直接使用
            cb = get_circuit_breaker(settings)
            ch_id = model_key_to_channel_id(result.recommended_model)
            if cb.get_channel_status(ch_id) == ChannelStatus.ACTIVE:
                return _build_result(
                    result.recommended_model,
                    resolved_tier,
                    settings,
                    result,
                    pass_tools=pass_tools,
                    upgrade_tier=upgrade_tier,
                )
            # 推荐模型被熔断禁用或处于冷却中，从该 tier 中重新选择其他可用渠道
            entry = select_channel(resolved_tier, settings, exclude_ids={ch_id})
            if entry:
                res = _build_result_from_entry(
                    entry, resolved_tier, settings, result, pass_tools=pass_tools, upgrade_tier=upgrade_tier
                )
            else:
                res = _build_result(
                    result.recommended_model,
                    resolved_tier,
                    settings,
                    result,
                    pass_tools=pass_tools,
                    upgrade_tier=upgrade_tier,
                )
            return _apply_circuit_breaker_fallback(res, settings)

    resolved_tier, resolved_key = _resolve_tier(result.tier, routing)
    entry = select_channel(resolved_tier, settings)
    if entry:
        res = _build_result_from_entry(
            entry, resolved_tier, settings, result, pass_tools=pass_tools, upgrade_tier=upgrade_tier
        )
    else:
        res = _build_result(
            resolved_key, resolved_tier, settings, result, pass_tools=pass_tools, upgrade_tier=upgrade_tier
        )

    return _apply_circuit_breaker_fallback(res, settings)


def _apply_circuit_breaker_fallback(result: RouteResult, settings: Settings) -> RouteResult:
    """检查熔断状态：若所选模型已被熔断处于冷却中，尝试主动预降级到 fallback 链中的可用模型"""
    if not settings.circuit_breaker.enabled or not settings.fallback.enabled:
        return result

    cb = get_circuit_breaker(settings)
    ch_id = result.channel_id or model_key_to_channel_id(result.model_key)
    if cb.get_channel_status(ch_id) != ChannelStatus.ACTIVE:
        from prism_router.fallback import build_fallback_route_result, get_fallback_chain

        for fb_tier in get_fallback_chain(result.tier, settings):
            fb_res = build_fallback_route_result(fb_tier, settings, result)
            if fb_res:
                fb_id = fb_res.channel_id or model_key_to_channel_id(fb_res.model_key)
                if cb.get_channel_status(fb_id) == ChannelStatus.ACTIVE:
                    logger.warning(
                        "Model %s is in circuit breaker cooldown, proactively routing to fallback %s",
                        result.model_key,
                        fb_res.model_key,
                    )
                    return fb_res

    return result


def _build_result(
    model_key: str,
    tier: str,
    settings: Settings,
    classification: ClassificationResult | None,
    pass_tools: bool = False,
    upgrade_tier: bool = False,
) -> RouteResult:
    cfg = settings.get_model_config(model_key)
    use_tools = pass_tools and (cfg.supports_tools if cfg else False)
    return RouteResult(
        model_key=model_key,
        model_id=cfg.model if cfg else model_key,
        tier=tier,
        base_url=cfg.base_url if cfg else "",
        api_key=cfg.api_key if cfg else "",
        is_local=settings.is_local(model_key),
        classification=classification,
        model_config=cfg,
        channel_id=model_key_to_channel_id(model_key),
        use_tools=use_tools,
    )


def _build_result_from_entry(
    entry: ChannelEntry,
    tier: str,
    settings: Settings,
    classification: ClassificationResult | None,
    pass_tools: bool = False,
    upgrade_tier: bool = False,
) -> RouteResult:
    """从 ChannelEntry 构建 RouteResult"""
    return _build_result(entry.model, tier, settings, classification, pass_tools, upgrade_tier)
