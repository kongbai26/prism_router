"""渠道健康度追踪 + 熔断器

参考:
- one-api: 滑动窗口健康度、错误分类禁用
- LiteLLM: CooldownCache TTL 熔断
- LLMux: 三态熔断器 (CLOSED/OPEN/HALF_OPEN)
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("prism_router.health")


class ChannelStatus(Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


class CircuitState(Enum):
    CLOSED = "closed"  # 正常
    OPEN = "open"  # 熔断
    HALF_OPEN = "half_open"  # 半开（探测）


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True
    failure_threshold: int = 2
    local_failure_threshold: int = 3
    success_threshold: int = 1
    cooldown_base: float = 30
    cooldown_max: float = 3000
    jitter: float = 0.2
    window_size: int = 20
    # 慢响应检测
    slow_response_enabled: bool = False
    slow_response_threshold_ms: int = 15000
    slow_response_failure_threshold: int = 3


@dataclass
class ChannelHealth:
    """单个渠道的健康度追踪"""

    window_size: int = 20
    slow_response_threshold_ms: int = 15000  # 慢响应阈值（从 config 传入）
    results: deque[bool] = field(default_factory=lambda: deque())
    latencies: deque[float] = field(default_factory=lambda: deque())
    failure_count: int = 0
    slow_response_count: int = 0  # 连续慢响应计数
    cooldown_until: float = 0
    state: CircuitState = CircuitState.CLOSED
    consecutive_successes: int = 0  # HALF_OPEN 状态下的成功计数

    def __post_init__(self):
        self.results = deque(maxlen=self.window_size)
        self.latencies = deque(maxlen=self.window_size)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(self.results) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def record_success(self, latency_ms: float = 0):
        self.results.append(True)
        if latency_ms > 0:
            self.latencies.append(latency_ms)
        self.failure_count = 0
        # 慢响应检测：成功但太慢
        if latency_ms > self.slow_response_threshold_ms:
            self.slow_response_count += 1
        else:
            self.slow_response_count = 0
        # consecutive_successes 只在 HALF_OPEN 状态下累加（用于探测恢复）
        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

    def record_failure(self):
        self.results.append(False)
        self.failure_count += 1
        self.consecutive_successes = 0


# ── 致命错误码/消息 ──

FATAL_STATUS_CODES = {401, 403}
FATAL_ERROR_TYPES = {
    "invalid_api_key",
    "insufficient_quota",
    "authentication_error",
    "permission_error",
    "forbidden",
    "account_deactivated",
}
FATAL_KEYWORDS = [
    "credit",
    "balance",
    "terminated",
    "violation",
    "permission denied",
    "invalid api key",
    "quota exceeded",
]


def is_fatal_error(status_code: int, error_body: dict | str | None = None) -> bool:
    """判断是否为致命错误（应立即禁用渠道）"""
    if status_code in FATAL_STATUS_CODES:
        return True

    if error_body is None:
        return False

    if isinstance(error_body, dict):
        err_type = error_body.get("type", "")
        err_msg = error_body.get("message", "").lower()
        if err_type in FATAL_ERROR_TYPES:
            return True
        return any(kw in err_msg for kw in FATAL_KEYWORDS)

    if isinstance(error_body, str):
        lower = error_body.lower()
        return any(kw in lower for kw in FATAL_KEYWORDS)

    return False


def is_retryable_error(status_code: int) -> bool:
    """判断是否为可重试错误"""
    return status_code == 429 or status_code >= 500


# ── 熔断器 ──


class CircuitBreaker:
    """三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED"""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._health: dict[int, ChannelHealth] = {}

    def get_health(self, channel_id: int) -> ChannelHealth:
        if channel_id not in self._health:
            self._health[channel_id] = ChannelHealth(
                window_size=self.config.window_size,
                slow_response_threshold_ms=self.config.slow_response_threshold_ms,
            )
        return self._health[channel_id]

    def record_success(self, channel_id: int, latency_ms: float = 0):
        if not self.config.enabled:
            return
        health = self.get_health(channel_id)
        health.record_success(latency_ms)

        # 慢响应达到阈值 → OPEN（仅 CLOSED 状态触发）
        if (
            self.config.slow_response_enabled
            and health.slow_response_count >= self.config.slow_response_failure_threshold
            and health.state == CircuitState.CLOSED
        ):
            health.state = CircuitState.OPEN
            cooldown = self._get_cooldown_time(health.slow_response_count)
            health.cooldown_until = time.time() + cooldown
            logger.warning(
                "Channel %d circuit OPEN (slow responses=%d, threshold=%dms)",
                channel_id,
                health.slow_response_count,
                health.slow_response_threshold_ms,
            )
            self.save_state()

        # HALF_OPEN → CLOSED：探测成功，恢复
        if health.state == CircuitState.HALF_OPEN:
            if health.consecutive_successes >= self.config.success_threshold:
                health.state = CircuitState.CLOSED
                health.cooldown_until = 0
                logger.info("Channel %d recovered (HALF_OPEN → CLOSED)", channel_id)
                self.save_state()

    def record_failure(self, channel_id: int, is_local: bool = False):
        if not self.config.enabled:
            return
        health = self.get_health(channel_id)
        health.record_failure()
        health.slow_response_count = 0  # 正常失败重置慢响应计数

        # CLOSED → OPEN：连续失败达阈值（本地模型用更低阈值）
        if health.state == CircuitState.CLOSED:
            threshold = self.config.local_failure_threshold if is_local else self.config.failure_threshold
            if health.failure_count >= threshold:
                health.state = CircuitState.OPEN
                cooldown = self._get_cooldown_time(health.failure_count)
                health.cooldown_until = time.time() + cooldown
                logger.warning(
                    "Channel %d circuit OPEN (failures=%d, cooldown=%.0fs, local=%s)",
                    channel_id,
                    health.failure_count,
                    cooldown,
                    is_local,
                )
                self.save_state()

        # HALF_OPEN → OPEN：探测失败，重新熔断（使用基础冷却，不累加）
        elif health.state == CircuitState.HALF_OPEN:
            health.state = CircuitState.OPEN
            cooldown = self._get_cooldown_time(1)
            health.cooldown_until = time.time() + cooldown
            logger.warning(
                "Channel %d probe failed (HALF_OPEN → OPEN, cooldown=%.0fs)",
                channel_id,
                cooldown,
            )
            self.save_state()

    def disable_channel(self, channel_id: int):
        """立即禁用渠道（致命错误）"""
        health = self.get_health(channel_id)
        health.state = CircuitState.OPEN
        health.cooldown_until = float("inf")  # 永久冷却，需手动恢复
        logger.warning("Channel %d DISABLED (fatal error)", channel_id)
        self.save_state()

    def get_channel_status(self, channel_id: int) -> ChannelStatus:
        """获取渠道当前状态"""
        if not self.config.enabled:
            return ChannelStatus.ACTIVE

        health = self.get_health(channel_id)

        if health.cooldown_until == float("inf"):
            return ChannelStatus.DISABLED

        if health.state == CircuitState.OPEN:
            if health.is_in_cooldown:
                return ChannelStatus.COOLDOWN
            else:
                # 冷却到期 → HALF_OPEN
                health.state = CircuitState.HALF_OPEN
                health.consecutive_successes = 0
                logger.info("Channel %d entering HALF_OPEN (cooldown expired)", channel_id)
                return ChannelStatus.ACTIVE

        if health.state == CircuitState.HALF_OPEN:
            return ChannelStatus.ACTIVE

        # CLOSED
        return ChannelStatus.ACTIVE

    def _get_cooldown_time(self, failure_count: int) -> float:
        """冷却时间：指数退避 + 抖动"""
        cooldown: float = min(
            self.config.cooldown_base * (2 ** (failure_count - 1)),
            self.config.cooldown_max,
        )
        jitter: float = cooldown * random.uniform(-self.config.jitter, self.config.jitter)
        return max(0.0, cooldown + jitter)

    def reset_channel(self, channel_id: int):
        """手动重置渠道状态"""
        if channel_id in self._health:
            del self._health[channel_id]
            logger.info("Channel %d manually reset", channel_id)
            from prism_router.db import delete_channel_health

            delete_channel_health(channel_id)

    def save_state(self) -> None:
        """将所有渠道状态持久化到 DB"""
        from prism_router.db import save_channel_health

        for channel_id in list(self._health.keys()):
            health = self._health[channel_id]
            save_channel_health(
                channel_id=channel_id,
                state=health.state.value,
                cooldown_until=health.cooldown_until,
                failure_count=health.failure_count,
                consecutive_successes=health.consecutive_successes,
                slow_response_count=health.slow_response_count,
            )

    def load_state(self) -> None:
        """从 DB 恢复渠道状态"""
        from prism_router.db import load_all_channel_health

        rows = load_all_channel_health()
        for row in rows:
            channel_id = row["channel_id"]
            health = self.get_health(channel_id)
            try:
                health.state = CircuitState(row["state"])
            except ValueError:
                health.state = CircuitState.CLOSED
            health.cooldown_until = row["cooldown_until"]
            health.failure_count = row["failure_count"]
            health.consecutive_successes = row["consecutive_successes"]
            health.slow_response_count = row.get("slow_response_count", 0)
        if rows:
            logger.info("Loaded circuit breaker state for %d channels from DB", len(rows))

    def get_all_health(self) -> dict[int, ChannelHealth]:
        """返回所有渠道的健康状态（只读副本）"""
        return dict(self._health)
