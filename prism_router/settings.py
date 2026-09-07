"""配置加载：YAML + .env，环境变量覆盖"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """单个模型的完整配置"""

    model: str  # 发给 API 的实际 model ID
    base_url: str = ""  # API 地址（完整端点，不做拼接）
    api_key: str = ""  # 认证密钥
    backend: str = ""  # 本地后端类型: "ollama" / "llamacpp" / "" (自动检测)
    context: int = 4096  # 上下文窗口
    cost_in: float = 0  # 输入价格 ($/1K tokens)
    cost_out: float = 0  # 输出价格 ($/1K tokens)
    tier: str = "mid"  # 默认 tier
    description: str = ""  # 模型能力描述（带给分类器）
    tags: list[str] = Field(default_factory=list)  # 能力标签，如 ["code", "chat", "reasoning"]
    source: str = "auto"  # "auto" / "local" / "cloud"（显式标注模型来源）
    supports_tools: bool = False  # 是否支持 tool calling（用于路由决策）
    connect_timeout: float | None = None  # 单个模型专用的连接超时（秒），留空使用全局配置


class ChannelEntry(BaseModel):
    """单个渠道配置（tier 内的一个候选模型）"""

    model: str  # model_key，如 "zhipu/glm-4.5-flash"
    priority: int = 1  # 越高越优先
    weight: int = 100  # 同优先级内的权重


class RoutingConfig(BaseModel):
    mode: str = "manual"  # "auto" / "manual"
    auto_strategy: str = "cost"  # auto 模式分配策略: "cost" / "tier"
    output_cost_weight: int = 3  # 成本计算中输出价格的倍率（total = cost_in + cost_out * weight）
    simple: str = ""
    mid: str = ""
    complex: str = ""
    classifier: str = "rules"  # "rules" / "local" / "cloud"
    classifier_model: str = ""  # 分类器模型 key，引用 models 中的定义
    classifier_failure_threshold: int = 3  # 分类器连续失败多少次后禁用
    # 分类器总超时（防止深度思考模型无限挂起）
    classifier_timeout_enabled: bool = True  # 分类器总超时开关
    classifier_timeout_seconds: int = 15  # 分类器总超时（秒），应 ≥ httpx timeout
    # 分类器上下文配置
    classifier_local_rounds: int = 3  # 本地分类器传几轮对话（含所有角色）
    classifier_local_max_chars: int = 300  # 本地分类器每轮截断字符数
    classifier_local_max_tokens: int = 4096  # 本地分类器最大输出 token 数
    classifier_cloud_rounds: int = 6  # 云端分类器传几轮对话（含所有角色）
    classifier_cloud_max_chars: int = 900  # 云端分类器每轮截断字符数
    classifier_cloud_max_tokens: int = 256  # 云端分类器最大输出 token 数
    classifier_max_retries: int = 3  # 分类器瞬时错误重试次数
    classify_prompt: str = ""  # 分类器提示词路径（由 config.yaml 提供）
    # 分类器探测
    classifier_probe_enabled: bool = True  # 是否定期探测分类器后端可用性
    classifier_probe_interval: int = 30  # 探测间隔（秒）
    # 多渠道配置（新格式，优先级高于 simple/mid/complex）
    tiers: dict[str, list[ChannelEntry]] = Field(default_factory=dict)
    # 模型池 — 参与自动路由的模型列表（仅 mode: auto 时生效）
    model_pool: list[str] = Field(default_factory=list)
    # 模型别名 — 客户端可用语义化名称（如 "fast" → "zhipu/glm-4.5-flash"）
    aliases: dict[str, str] = Field(default_factory=dict)
    # Tools 策略 — 控制 body 带 tools 时是否触发 tier 升级
    #   "auto"   = 仅在 messages 中有实际 tool 调用时才升级（默认，推荐）
    #   "always" = 只要 body 带 tools 就升级（旧行为）
    #   "never"  = 忽略 tools，纯按消息内容分类
    tools_policy: str = "auto"
    tools_enabled: bool = True  # tools 策略总开关（关闭后 tools 仍透传，但不做 tier 升级和分类器提示）
    # 透传模式可用模型 key 列表（引用 models 中的定义）。空=所有 models 中的模型均可透传
    passthrough_pool: list[str] = Field(default_factory=list)


class FallbackConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 3
    connect_max_retries: int = 1  # 网络连接失败/超时的重试次数（默认 1 次，快速失败）
    connect_timeout_seconds: float = 8.0  # TCP 连接建立超时（秒），默认 8s（兼顾内网与冷启动，防止死等）
    retry_on: list[str | int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])  # type: ignore[arg-type]
    timeout_seconds: int = 120
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    first_token_timeout: float = 0  # 流式首 chunk 超时（秒），0=不启用（用 timeout_seconds）


class LoggingConfig(BaseModel):
    level: str = "INFO"
    db_path: str = "logs/requests.db"
    content_logging_enabled: bool = False  # 是否持久化请求、响应和提示词内容（默认关闭）
    artifact_storage_enabled: bool = False  # 是否将大 tool output 持久化到 SQLite（默认关闭）
    body_max_size: int = 5120
    full_body_max_size: int = 10240
    retention_days: dict = Field(
        default_factory=lambda: {
            "request_bodies": 3,
            "request_logs": 30,
            "error_logs": 90,
            "rewrite_logs": 30,
            "server_logs": 7,
            "rewrite_files": 7,
            "artifacts": 1,
        }
    )
    tool_call_cache: dict = Field(
        default_factory=lambda: {
            "ttl_seconds": 3600,
            "max_size": 1000,
            "persist": False,
        }
    )


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    api_key: str = ""
    admin_api_key: str = ""  # 管理接口独立密钥；留空时禁用管理接口
    proxy: str = ""  # 代理地址，留空=不使用代理
    verify_ssl: bool = True  # 是否验证 HTTPS 证书（云端 API 必须保持开启）
    write_codex_catalog: bool = False  # 是否写入用户 Codex 配置（默认关闭）
    dedup_window_seconds: int = 10  # 请求去重窗口（秒），0=不启用


class CacheConfig(BaseModel):
    enabled: bool = False
    ttl_seconds: int = 300
    max_entries: int = 1000


class RateLimitConfig(BaseModel):
    enabled: bool = False
    local_max_concurrent: int = 3


class CircuitBreakerSettings(BaseModel):
    enabled: bool = True
    failure_threshold: int = 2
    success_threshold: int = 1
    local_failure_threshold: int = 3
    cooldown_base: float = 30
    cooldown_max: float = 3000
    jitter: float = 0.2
    window_size: int = 20
    # 慢响应检测（成功但太慢也算失败）
    slow_response_enabled: bool = True
    slow_response_threshold_ms: int = 15000  # 慢响应阈值（毫秒）
    slow_response_failure_threshold: int = 3  # 连续慢响应多少次触发熔断


class ContextConfig(BaseModel):
    """会话上下文配置"""

    enabled: bool = True
    max_conversations: int = 200  # 最大跟踪会话数
    history_window: int = 3  # 每个会话保留的历史决策数


class RewritingTransforms(BaseModel):
    """改写变换控制"""

    clarity: bool = True  # 清晰度变换：模糊指令变具体
    structure: bool = False  # 结构化变换：多部分请求拆成编号子任务（仅大 tier 模型）
    completeness: bool = False  # 完整性变换：补充隐含需求（仅大 tier 模型）


class RewritingConfig(BaseModel):
    """智能改写配置"""

    enabled: bool = False
    mode: str = "classify_only"  # "classify_only" / "rewrite_only" / "classify_and_rewrite" / "passthrough"
    rewriter_model: str = ""  # 改写器模型 key，引用 models 中的定义
    fallback_to_original: bool = True  # 改写失败时降级用原始 prompt
    add_plan: bool = False  # 为复杂任务生成执行计划（仅大 tier 模型）
    rewrite_small_prompt: str = ""  # 小模型改写提示词路径（由 config.yaml 提供）
    rewrite_large_prompt: str = ""  # 大模型改写提示词路径（由 config.yaml 提供）
    tier_template_map: dict[str, str] = Field(
        default_factory=lambda: {"simple": "small", "mid": "large", "complex": "large"}
    )  # tier → 模板类型映射（"small" / "large"）
    transforms: RewritingTransforms = Field(default_factory=RewritingTransforms)
    max_rewriter_tokens: int = 2048  # 改写器最大输出 token
    timeout_seconds: int = 30  # 改写器超时（httpx per-read timeout）
    # 改写器总超时（防止深度思考模型无限挂起）
    rewriter_timeout_enabled: bool = True  # 改写器总超时开关
    rewriter_timeout_seconds: int = 20  # 改写器总超时（秒）
    rewrite_log_dir: str = "logs/rewrites"  # 改写日志目录
    write_rewrite_logs: bool = False  # 是否将改写前后文本写入文件（默认关闭）
    cli_timeout: int = 5  # CLI 启动时交互选择超时（秒），0=不超时
    cli_default_choice: int = 0  # 超时默认选项（1-5），0=自动根据当前模式推断
    rewrite_when_model_specified: bool = True  # 用户指定模型时是否改写（classify_and_rewrite 模式）
    skip_rewrite_keywords: list[str] = Field(
        default_factory=lambda: [
            "title",
            "summarize",
            "summarise",
        ]
    )  # 包含这些关键词的请求跳过改写（忽略大小写，简单匹配）
    skip_rewrite_patterns: list[str] = Field(default_factory=list)  # 正则匹配（高级，忽略大小写）
    # 上下文配置（与分类器对齐）
    context_local_rounds: int = 2  # 本地改写器传几轮对话（0=不传上下文）
    context_local_max_chars: int = 200  # 本地改写器每轮截断字符数
    context_cloud_rounds: int = 4  # 云端改写器传几轮对话
    context_cloud_max_chars: int = 400  # 云端改写器每轮截断字符数
    # 改写器生成参数
    temperature: float = 0.3  # 改写器温度（低=稳定，高=多样）
    top_p: float = 0.9  # top_p 采样
    thinking_enabled: bool = True  # 是否启用 thinking（如 API 支持）
    thinking_params: dict[str, Any] = Field(default_factory=lambda: {"thinking": {"type": "enabled"}})
    # 改写器熔断配置
    breaker_enabled: bool = True
    breaker_failure_threshold: int = 3  # 连续失败几次后熔断
    breaker_cooldown_base: float = 30  # 冷却基准时间（秒）
    breaker_cooldown_max: float = 600  # 冷却最大时间（秒）
    # 改写器探测
    probe_enabled: bool = True  # 是否定期探测改写器后端可用性
    probe_interval: int = 30  # 探测间隔（秒）


# Tier fallback 顺序（所有模块共享）
TIER_FALLBACK = {
    "simple": ["mid", "complex"],
    "mid": ["complex", "simple"],
    "complex": ["mid", "simple"],
}


def model_key_to_channel_id(model_key: str) -> int:
    """model_key 转换为稳定的整数 ID（用于健康度追踪，大小写不敏感）"""
    return int(hashlib.md5(model_key.lower().encode()).hexdigest()[:8], 16)


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    context: ContextConfig = Field(default_factory=ContextConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rewriting: RewritingConfig = Field(default_factory=RewritingConfig)

    def get_model_config(self, key: str) -> ModelConfig | None:
        """根据 provider/model 格式的 key 获取模型配置（大小写不敏感）"""
        cfg = self.models.get(key)
        if cfg is not None:
            return cfg
        # 大小写不敏感匹配
        key_lower = key.lower()
        for model_key, model_cfg in self.models.items():
            if model_key.lower() == key_lower:
                return model_cfg
        return None

    def is_local(self, key: str) -> bool:
        """判断模型是否为本地/内网模型"""
        cfg = self.get_model_config(key)
        if cfg is None:
            return False
        # source 字段优先
        if cfg.source == "local":
            return True
        if cfg.source == "cloud":
            return False
        # auto 推断：没有 api_key 或 base_url 是本地/内网地址
        if cfg.api_key:
            return False
        url = cfg.base_url.lower()
        # localhost / 127.0.0.1
        if "localhost" in url or "127.0.0.1" in url:
            return True
        # 内网 IP: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
        import re

        if re.search(r"://10\.\d+\.\d+\.\d+", url):
            return True
        if re.search(r"://172\.(1[6-9]|2\d|3[01])\.\d+\.\d+", url):
            return True
        if re.search(r"://192\.168\.\d+\.\d+", url):
            return True
        return False

    def get_tier_channels(self, tier: str) -> list[ChannelEntry]:
        """获取某个 tier 的渠道列表，兼容旧格式"""
        # 新格式：tiers 里有配置
        if self.routing.tiers and tier in self.routing.tiers:
            return self.routing.tiers[tier]

        # 旧格式：simple/mid/complex 字符串
        model_key = getattr(self.routing, tier, "")
        if model_key:
            return [ChannelEntry(model=model_key, priority=1, weight=100)]
        return []

    def find_tools_tier(self) -> str:
        """找到最便宜的支持 tool calling 的 tier，找不到返回空字符串"""
        tier_order = ["simple", "mid", "complex"]
        for tier in tier_order:
            channels = self.get_tier_channels(tier)
            for ch in channels:
                cfg = self.get_model_config(ch.model)
                if cfg and cfg.supports_tools:
                    return tier
        logging.getLogger("prism_router.settings").warning(
            "No model with supports_tools=True found in any tier. Tool calling requests may not route correctly."
        )
        return ""


def _resolve_env_vars(obj: Any) -> Any:
    """递归解析 ${VAR} 格式的环境变量引用"""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        value = os.environ.get(var_name, "")
        if value == "":
            logging.getLogger("prism_router.settings").warning(
                "Environment variable %s is not set, using empty string", var_name
            )
        return value
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def load_settings(config_path: str | Path | None = None) -> Settings:
    """加载配置：先读 YAML，再加载 .env，环境变量覆盖"""
    if config_path is None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
        if config_path is None:
            config_path = Path("config.yaml")

    config_path = Path(config_path)

    # 加载 .env
    env_path = config_path.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # 读取 YAML
    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # 解析 ${VAR} 引用
    raw = _resolve_env_vars(raw)

    return Settings(**raw)
