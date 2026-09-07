"""渠道选择：优先级 + 权重 + 熔断器过滤、自动 tier 分配"""

from __future__ import annotations

import logging
import random

from prism_router.health import ChannelStatus
from prism_router.settings import ChannelEntry, Settings, model_key_to_channel_id

logger = logging.getLogger("prism_router.channel")


def weighted_random_select(entries: list[ChannelEntry]) -> ChannelEntry:
    """加权随机选择（参考 new-api）"""
    total = sum(e.weight for e in entries)
    if total == 0:
        return random.choice(entries)
    r = random.randint(0, total - 1)
    for entry in entries:
        r -= entry.weight
        if r < 0:
            return entry
    return entries[-1]


def select_channel(
    tier: str,
    settings: Settings,
    exclude_ids: set[int] | None = None,
) -> ChannelEntry | None:
    """
    从 tier 的渠道列表中选择最优渠道。

    选择逻辑（参考 one-api）：
    1. 硬性过滤：排除手动禁用、上下文不够、已排除的
    2. 按 priority 降序分组
    3. 先试最高优先级（跳过 cooldown 中的）
    4. 最高优先级全 cooldown → 放宽到所有层级
    5. 全部 cooldown → 强制选一个
    """
    from prism_router.routing import get_circuit_breaker

    if exclude_ids is None:
        exclude_ids = set()

    channels = settings.get_tier_channels(tier)
    if not channels:
        return None

    cb = get_circuit_breaker(settings)

    available = []
    for ch in channels:
        ch_id = model_key_to_channel_id(ch.model)
        if ch_id in exclude_ids:
            continue
        status = cb.get_channel_status(ch_id)
        if status == ChannelStatus.DISABLED:
            continue
        available.append((ch, ch_id, status))

    if not available:
        return None

    available.sort(key=lambda x: x[0].priority, reverse=True)

    first_priority = available[0][0].priority
    top_active = [
        ch for ch, ch_id, status in available if ch.priority == first_priority and status == ChannelStatus.ACTIVE
    ]
    if top_active:
        return weighted_random_select(top_active)

    all_active = [ch for ch, ch_id, status in available if status == ChannelStatus.ACTIVE]
    if all_active:
        return weighted_random_select(all_active)

    top_entries = [ch for ch, _, _ in available if ch.priority == first_priority]
    if top_entries:
        return top_entries[0]
    return available[0][0]


def auto_assign_tiers(settings: Settings) -> dict[str, list[ChannelEntry]]:
    """
    从 model_pool 自动分配 tier。

    策略（由 routing.auto_strategy 控制）：
    - "cost": 按成本从低到高分配
    - "tier": 按模型 config 的 tier 字段分配

    两种策略都保留 routing.simple/mid/complex 的用户手动指定。
    返回 routing.tiers 格式。
    """

    routing = settings.routing
    strategy = getattr(routing, "auto_strategy", "cost")

    manual_models = set()
    manual_tiers = {}
    for tier in ("simple", "mid", "complex"):
        key = getattr(routing, tier, "")
        if key:
            manual_models.add(key)
            manual_tiers[tier] = [ChannelEntry(model=key, priority=1, weight=100)]

    pool = []
    for key in routing.model_pool:
        if key in manual_models:
            continue
        cfg = settings.get_model_config(key)
        if cfg:
            total_cost = cfg.cost_in + cfg.cost_out * routing.output_cost_weight
            pool.append((key, cfg, total_cost))

    tiers = dict(manual_tiers)

    if strategy == "tier":
        tier_buckets: dict[str, list[tuple[str, float]]] = {"simple": [], "mid": [], "complex": []}
        rest = []
        for key, cfg, total_cost in pool:
            if cfg.tier in tier_buckets:
                tier_buckets[cfg.tier].append((key, total_cost))
            else:
                rest.append((key, total_cost))
        for tier in ("simple", "mid", "complex"):
            if tier not in tiers and tier_buckets[tier]:
                tier_buckets[tier].sort(key=lambda x: x[1])
                tiers[tier] = [ChannelEntry(model=tier_buckets[tier][0][0], priority=1, weight=100)]
        rest.sort(key=lambda x: x[1])
        for tier in ("simple", "mid", "complex"):
            if tier not in tiers and rest:
                key, _ = rest.pop(0)
                tiers[tier] = [ChannelEntry(model=key, priority=1, weight=100)]
    else:
        pool.sort(key=lambda x: x[2])
        for tier in ("simple", "mid", "complex"):
            if tier not in tiers and pool:
                key, _, _ = pool.pop(0)
                tiers[tier] = [ChannelEntry(model=key, priority=1, weight=100)]

    for tier in ("simple", "mid", "complex"):
        if tier not in tiers:
            tiers[tier] = []

    return tiers


def save_generated_routing(tiers: dict[str, list[ChannelEntry]], path: str) -> None:
    """将自动分配结果写入 routing.generated.yaml"""
    from pathlib import Path

    import yaml

    data = {}
    for tier in ("simple", "mid", "complex"):
        entries = tiers.get(tier, [])
        data[tier] = entries[0].model if entries else ""

    file_path = Path(path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# 此文件由 prism-router 自动生成，请勿手动编辑\n")
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
