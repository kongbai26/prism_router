"""Fallback 降级逻辑：请求失败时按 tier 链或 model 链自动重试"""

from __future__ import annotations

import logging

from prism_router.channel import select_channel
from prism_router.routing import RouteResult
from prism_router.settings import TIER_FALLBACK, Settings, model_key_to_channel_id

logger = logging.getLogger("prism_router.fallback")


def get_fallback_tiers(current_tier: str) -> list[str]:
    """返回当前 tier 的降级候选列表（默认 tier 级）"""
    return TIER_FALLBACK.get(current_tier, [])


def get_fallback_chain(current_tier: str, settings: Settings, override: list[str] | None = None) -> list[str]:
    """返回 fallback 链：优先用 per-request override，否则用默认 tier 顺序"""
    if override:
        return override
    return get_fallback_tiers(current_tier)


def build_fallback_from_model(
    model_key: str,
    settings: Settings,
    original: RouteResult,
) -> RouteResult | None:
    """从指定 model key 构建 fallback RouteResult"""
    cfg = settings.get_model_config(model_key)
    if not cfg:
        return None
    from prism_router.routing import _get_routing_tier_for_model

    tier = _get_routing_tier_for_model(model_key, settings.routing)
    # 重新计算 use_tools：基于 fallback 模型的 supports_tools
    use_tools = original.use_tools and cfg.supports_tools
    return RouteResult(
        model_key=model_key,
        model_id=cfg.model,
        tier=tier,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        is_local=settings.is_local(model_key),
        classification=original.classification,
        model_config=cfg,
        channel_id=model_key_to_channel_id(model_key),
        use_tools=use_tools,
    )


def should_retry(status_code: int, retry_on: list[str | int]) -> bool:
    """判断是否应该重试（支持整数状态码和字符串标记如 'timeout'）"""
    for code in retry_on:
        if isinstance(code, int) and status_code == code:
            return True
        if isinstance(code, str) and code == "timeout" and status_code == 408:
            return True
        if isinstance(code, str) and str(status_code) == code:
            return True
    return False


def build_fallback_route_result(
    tier: str,
    settings: Settings,
    original: RouteResult,
) -> RouteResult | None:
    """为降级 tier 构建 RouteResult，模型不存在则返回 None"""
    # 通过 select_channel 选择渠道（考虑优先级/权重/熔断器）
    entry = select_channel(tier, settings)
    if not entry:
        # 所有渠道都被禁用，记录警告并返回 None
        logger.warning("All channels for tier %s are disabled, fallback unavailable", tier)
        return None

    model_key = entry.model
    cfg = settings.get_model_config(model_key)
    if not cfg:
        return None
    # 重新计算 use_tools：基于 fallback 模型的 supports_tools
    use_tools = original.use_tools and cfg.supports_tools
    if original.use_tools and not use_tools:
        logger.info("Fallback: tools downgraded (model %s doesn't support tools)", model_key)
    return RouteResult(
        model_key=model_key,
        model_id=cfg.model,
        tier=tier,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        is_local=settings.is_local(model_key),
        classification=original.classification,
        model_config=cfg,
        channel_id=model_key_to_channel_id(model_key),
        use_tools=use_tools,
    )
