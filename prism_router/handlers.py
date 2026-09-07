"""请求编排：chat_completions 主函数、缓存、并发限制、共用工具函数"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from prism_router.context import ConversationContext
from prism_router.dedup import get_dedup
from prism_router.fallback import (
    build_fallback_from_model,
    build_fallback_route_result,
    get_fallback_chain,
    should_retry,
)
from prism_router.health import is_fatal_error
from prism_router.logger import RequestLogger
from prism_router.rewriter import rewrite_prompt
from prism_router.routing import RouteResult, check_context_window, get_circuit_breaker, route
from prism_router.server import VIRTUAL_MODELS, _error, get_settings
from prism_router.settings import model_key_to_channel_id

if TYPE_CHECKING:
    from prism_router.settings import FallbackConfig, Settings

logger = logging.getLogger("prism_router.handlers")

# Responses API 工具转换函数（延迟导入避免循环）

# ── 共用工具函数 ──


def has_tool_evidence_in_messages(messages: list[dict]) -> bool:
    """检测 messages 中是否有 tool-calling 证据（tool_calls / tool result）"""
    for msg in messages:
        role = msg.get("role", "")
        if role in ("tool", "function"):
            return True
        if role == "assistant" and (msg.get("tool_calls") or msg.get("function_call")):
            return True
    return False


def strip_tool_calls_from_messages(messages: list[dict]) -> list[dict]:
    """从 messages 中移除 tool_calls/function_call 字段和 tool/function role 消息。"""
    cleaned = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ("tool", "function"):
            continue
        if role == "assistant":
            msg = dict(msg)
            msg.pop("tool_calls", None)
            msg.pop("function_call", None)
            if not msg.get("content") and not msg.get("reasoning_content"):
                continue
        cleaned.append(msg)
    return cleaned


# ── 内存缓存 ──


class ResponseCache:
    def __init__(self, max_entries: int, ttl_seconds: int):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, dict]] = {}

    def _make_key(self, model: str, messages: list[dict], tools: list | None) -> str:
        raw = json.dumps({"model": model, "messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], tools: list | None) -> dict | None:
        key = self._make_key(model, messages, tools)
        if key in self._store:
            ts, data = self._store[key]
            if time.time() - ts < self.ttl_seconds:
                return data
            del self._store[key]
        return None

    def put(self, model: str, messages: list[dict], tools: list | None, data: dict) -> None:
        key = self._make_key(model, messages, tools)
        if len(self._store) >= self.max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.time(), data)

    def clear(self) -> None:
        self._store.clear()


_cache: list[ResponseCache | None] = [None]


class _BoundedRewriteCache:
    """有上限的改写缓存，超出时淘汰最旧条目"""

    def __init__(self, max_entries: int = 200):
        from collections import OrderedDict

        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max = max_entries

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def put(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)


_rewrite_cache = _BoundedRewriteCache()


def _get_cache() -> ResponseCache | None:
    settings = get_settings()
    if settings.cache.enabled and _cache[0] is None:
        _cache[0] = ResponseCache(settings.cache.max_entries, settings.cache.ttl_seconds)
    return _cache[0] if settings.cache.enabled else None


# ── 本地模型并发限制 ──

_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(model_key: str) -> asyncio.Semaphore | None:
    settings = get_settings()
    if not settings.rate_limit.enabled:
        return None
    if model_key not in _semaphores:
        _semaphores[model_key] = asyncio.Semaphore(settings.rate_limit.local_max_concurrent)
    return _semaphores[model_key]


# ── 会话上下文 ──

_ctx: list[ConversationContext | None] = [None]


def _get_ctx() -> ConversationContext:
    if _ctx[0] is None:
        settings = get_settings()
        ctx_cfg = settings.context
        _ctx[0] = ConversationContext(
            max_conversations=ctx_cfg.max_conversations,
            history_window=ctx_cfg.history_window,
        )
    ctx = _ctx[0]
    assert ctx is not None
    return ctx


def reset_handlers() -> None:
    """重置所有处理器状态（用于测试 / 配置重载）"""
    _cache[0] = None
    _ctx[0] = None
    _semaphores.clear()


async def _prompt_choice_with_timeout(
    ctx: ConversationContext,
    messages: list[dict],
    choice_name: str,
    model: str,
    req_log: RequestLogger,
    timeout: float = 5.0,
) -> str:
    """会话内首次遇到未知模型时，给用户超时选择。

    选项：
      1. route        — 只路由（分类 → 选模型），不改写
      2. route_rewrite — 先路由再改写

    超时默认 route。选择结果缓存到会话上下文。
    """
    import sys

    existing = ctx.get_choice(messages, choice_name)
    if existing is not None:
        return existing

    option_map = {"1": "route", "2": "route_rewrite"}
    default = "route"

    if sys.stdin.isatty():
        try:
            import threading

            result: list[str] = []

            def _read():
                try:
                    ans = input(
                        f"\n  模型 '{model}' 未在配置中找到。\n"
                        f"  [1] 路由模型（默认，{timeout:.0f}s 后自动选择）\n"
                        f"  [2] 先路由后改写\n"
                        f"  请选择: "
                    ).strip()
                    result.append(ans)
                except EOFError:
                    pass

            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if result:
                choice = option_map.get(result[0], default)
            else:
                choice = default
                req_log.info("Choice timeout (%.0fs), defaulting to '%s'", timeout, default)
        except Exception:
            choice = default
    else:
        choice = default
        req_log.warning("%s: model '%s' not found, defaulting to '%s'", choice_name, model, default)

    ctx.set_choice(messages, choice_name, choice)
    req_log.info("%s: model '%s' → choice='%s'", choice_name, model, choice)
    return choice


# ── 请求执行分发 ──


async def execute_request(
    route_result: RouteResult,
    body: dict,
    stream: bool,
    fallback_cfg: FallbackConfig | None = None,
    proxy: str = "",
    settings: Settings | None = None,
    tried_tiers: list[str] | None = None,
    fallback_override: list[str] | None = None,
) -> JSONResponse | StreamingResponse:
    """根据模型类型分发到本地或云端处理"""
    from prism_router.cloud import handle_cloud
    from prism_router.local import handle_local

    if route_result.is_local:
        return await handle_local(
            route_result,
            body,
            stream,
            settings,
            tried_tiers,
            fallback_override,
            fallback_cfg,
            proxy,
            has_tool_evidence_fn=has_tool_evidence_in_messages,
            strip_tool_calls_fn=strip_tool_calls_from_messages,
            execute_request_fn=execute_request,
        )
    else:
        return await handle_cloud(
            route_result,
            body,
            stream,
            fallback_cfg,
            proxy,
            has_tool_evidence_fn=has_tool_evidence_in_messages,
            strip_tool_calls_fn=strip_tool_calls_from_messages,
            settings=settings,
            tried_tiers=tried_tiers,
            fallback_override=fallback_override,
        )


# ── chat_completions 端点 ──


async def chat_completions(request: Request):
    """HTTP 端点包装：解析请求体，委托给 process_chat_request"""
    try:
        body = await request.json()
    except Exception as e:
        logger.debug("JSON parse error: %s", e)
        return _error(400, "Invalid JSON body")
    return await process_chat_request(body)


async def process_chat_request(body: dict, settings: Settings | None = None):
    """Chat Completions 核心处理逻辑（纯函数，不需要 Request 对象）"""
    if settings is None:
        settings = get_settings()

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return _error(400, "messages is required and must be a non-empty array")
    if not all(isinstance(m, dict) for m in messages):
        return _error(400, "each message must be an object")

    model = body.get("model")
    stream = body.get("stream", False)
    content_logging_enabled = settings.logging.content_logging_enabled

    _msg_count = len(messages)
    _last_user_msg = ""
    if content_logging_enabled:
        for _m in reversed(messages):
            if _m.get("role") == "user":
                _content = _m.get("content", "")
                if isinstance(_content, str):
                    _last_user_msg = _content[:80]
                elif isinstance(_content, list):
                    for _p in _content:
                        if isinstance(_p, dict) and _p.get("type") == "text":
                            _last_user_msg = _p.get("text", "")[:80]
                            break
                break
    # 提前生成 request_id，所有日志统一携带
    start_time = time.time()
    from prism_router.db import gen_ulid

    request_id = gen_ulid()
    req_log = RequestLogger(logger, {"request_id": request_id})

    req_log.info("Request in: model=%s, stream=%s, msgs=%d", model, stream, _msg_count)

    # ── 请求去重 ──
    dedup_window = settings.server.dedup_window_seconds
    dedup_key_str: str | None = None
    if dedup_window > 0:
        dedup = get_dedup(dedup_window)
        dup_key = dedup.check_and_register(model or "prism-auto", messages, body.get("tools"))
        if dup_key is None:
            dedup_key_str = dedup._make_key(model or "prism-auto", messages, body.get("tools"))
        if dup_key is not None:
            req_log.warning("Dedup: 429 → %s (key=%s, retry after %ds)", model or "auto", dup_key[:12], dedup_window)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(dedup_window)},
                content={"error": {"message": "Duplicate request in progress", "type": "rate_limit_error"}},
            )
        logger.debug(
            "Dedup registered: key=%s, model=%s",
            dedup._make_key(model or "prism-auto", messages, body.get("tools"))[:12],
            model,
        )
    fallback_override = body.get("prism_fallback")

    # ── 获取会话上下文 ──
    ctx = _get_ctx()
    conversation_history = ctx.get_history(messages) if settings.context.enabled else None

    # ── passthrough 透传模式 ──
    _force_classify_rewrite = False
    if settings.rewriting.mode == "passthrough":
        if not model or model in ("prism-auto", "auto"):
            fb_model = (
                settings.routing.mid
                or getattr(settings.routing, "simple", "")
                or getattr(settings.routing, "complex", "")
            )
            if fb_model and settings.get_model_config(fb_model):
                model = fb_model
            else:
                can_route = bool(
                    settings.routing.mid
                    or settings.routing.tiers
                    or (settings.routing.mode == "auto" and settings.routing.model_pool)
                )
                if not can_route:
                    if dedup_window > 0 and dedup_key_str:
                        dedup.complete_by_key(dedup_key_str)
                    return _error(
                        400,
                        "当前处于纯透传模式（未配置默认目标模型），请在客户端请求中明确指定具体的 model 名称（如 deepseek-chat）",
                        "invalid_request_error",
                    )

    if settings.rewriting.mode == "passthrough" and model:
        cfg = settings.get_model_config(model)
        pool = settings.routing.passthrough_pool
        model_available = cfg is not None and (not pool or model.lower() in [p.lower() for p in pool])

        if not model_available:
            # 模型不在透传池 → 会话内首次给用户选择
            choice = await _prompt_choice_with_timeout(
                ctx,
                messages,
                "passthrough_fallback",
                model,
                req_log,
            )
            if choice == "route_rewrite":
                _force_classify_rewrite = True
        else:
            # 模型在透传池 → 直接转发
            assert cfg is not None
            route_result = RouteResult(
                model_key=model,
                model_id=cfg.model,
                tier=cfg.tier,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                is_local=settings.is_local(model),
                classification=None,
                model_config=cfg,
                channel_id=model_key_to_channel_id(model),
                use_tools=bool(body.get("tools")),
            )
            req_log.info("Passthrough: %s → %s", model, model)

            # 转换工具格式（Responses API → Chat Completions）
            # 防止客户端发 flat 格式 tools 导致上游报 tools[0] is missing function.name
            from prism_router.responses_api import _convert_tool_choice, _convert_tools

            if body.get("tools") or "tool_choice" in body:
                body = dict(body)
                if body.get("tools"):
                    body["tools"] = _convert_tools(body["tools"])
                if "tool_choice" in body:
                    body["tool_choice"] = _convert_tool_choice(body["tool_choice"])

            tools = body.get("tools")
            if tools:
                tool_names = [t.get("function", {}).get("name") or t.get("name", "?") for t in tools]
                req_log.info("  tools: %d [%s]", len(tools), ", ".join(tool_names))
            tool_choice = body.get("tool_choice")
            if tool_choice:
                req_log.info("  tool_choice: %s", tool_choice)

            # 主动上下文压缩：当消息数超过阈值时自动触发 compact
            # 参考 codex-app-transfer: auto_compact_token_limit = context_window × 80%
            # 我们用消息数近似判断（每条约 800 tokens，含工具定义）
            msg_count = len(body.get("messages", []))
            context_window = getattr(cfg, "context", 128000) or 128000
            # 工具定义也占 token，13 个工具约 8000 tokens
            tool_count = len(body.get("tools", []))
            tool_tokens = tool_count * 600
            # 内部触发阈值 70%（比报告给客户端的 80% 更保守，留余量）
            auto_compact_limit = context_window * 70 // 100 - tool_tokens
            estimated_tokens = msg_count * 800  # 每条约 800 tokens
            if estimated_tokens > auto_compact_limit and msg_count > 15:
                req_log.info(
                    "Auto-compact triggered: %d msgs (~%d tokens > %d limit)",
                    msg_count,
                    estimated_tokens,
                    auto_compact_limit,
                )
                # 调用 compact 逻辑压缩消息
                from prism_router.responses_api import _auto_compact_messages

                body = await _auto_compact_messages(body, model, settings, req_log)

            # 展开 artifact 引用为完整内容
            if settings.logging.artifact_storage_enabled:
                from prism_router.artifact_store import get_artifact_store

                body["messages"] = get_artifact_store().expand_artifacts(body.get("messages", []))

            sem = _get_semaphore(route_result.model_key)
            try:
                if sem:
                    async with sem:
                        response = await execute_request(route_result, body, stream, settings=settings)
                else:
                    response = await execute_request(route_result, body, stream, settings=settings)
            except Exception as e:
                req_log.error("Passthrough failed: %s", e, exc_info=True)
                response = _error(502, str(e), "upstream_error")

            elapsed = time.time() - start_time
            is_streaming_response = isinstance(response, StreamingResponse)
            status = 0 if is_streaming_response else (response.status_code if hasattr(response, "status_code") else 0)
            req_log.debug("Response: %s | %d | %.2fs", route_result.model_key, status, elapsed)

            # 熔断器记录
            if route_result.channel_id:
                cb = get_circuit_breaker()
                stream_ok = is_streaming_response and not getattr(response, "_had_error", False)
                json_ok = isinstance(response, JSONResponse) and response.status_code == 200
                if stream_ok or json_ok:
                    cb.record_success(route_result.channel_id, latency_ms=elapsed * 1000)
                else:
                    cb.record_failure(route_result.channel_id, is_local=route_result.is_local)

            # DB 日志
            from prism_router.db import log_request, log_request_bodies, log_request_detail

            resp_status = (
                0 if is_streaming_response else (response.status_code if hasattr(response, "status_code") else 0)
            )
            resp_usage: dict[str, Any] = {}
            response_body = None
            if is_streaming_response:
                resp_usage = getattr(response, "_usage", {})
            elif isinstance(response, JSONResponse):
                try:
                    response_body = json.loads(bytes(response.body))
                    resp_usage = response_body.get("usage", {})
                except Exception:
                    pass

            prompt_tokens = resp_usage.get("prompt_tokens", 0)
            completion_tokens = resp_usage.get("completion_tokens", 0)
            final_model_key = model
            if hasattr(response, "_final_model_key"):
                final_model_key = response._final_model_key

            final_cfg = settings.get_model_config(final_model_key) or route_result.model_config
            cost = 0
            if final_cfg:
                cost = (prompt_tokens * final_cfg.cost_in + completion_tokens * final_cfg.cost_out) / 1000

            time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            log_request(
                client_model=model,
                final_model=final_model_key,
                tier=route_result.tier,
                classifier_method="passthrough",
                classifier_model="",
                status_code=resp_status,
                latency_ms=elapsed * 1000,
                upstream_latency_ms=0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                is_stream=stream,
                is_fallback=False,
                tried_tiers=None,
                is_rewritten=False,
                rewriter_latency_ms=0,
                request_id=request_id,
                last_user_msg=_last_user_msg,
                msg_count=_msg_count,
                has_tools=bool(body.get("tools")),
                tool_count=len(body.get("tools") or []),
                tools_upgraded=False,
            )

            body_max_size = getattr(settings.logging, "body_max_size", 10240)
            full_body_max_size = getattr(settings.logging, "full_body_max_size", 5120)
            if content_logging_enabled:
                log_request_bodies(
                    request_id=request_id,
                    request_body=body,
                    response_body=response_body,
                    body_max_size=body_max_size,
                    full_body_max_size=full_body_max_size,
                )

            log_request_detail(
                request_id=request_id,
                time_str=time_str,
                client_model=model,
                final_model=final_model_key,
                tier=route_result.tier,
                classifier_method="passthrough",
                classifier_model="",
                latency_ms=elapsed * 1000,
                upstream_latency_ms=0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status_code=resp_status,
                is_stream=stream,
                is_fallback=False,
                tried_tiers=None,
                tools_upgraded=False,
                is_rewritten=False,
                rewriter_latency_ms=0,
                rewriter_model="",
                target_model="",
                last_user_msg=_last_user_msg,
            )

            # 错误详情写入 DB（passthrough 路径）
            _is_stream_error = is_streaming_response and getattr(response, "_had_error", False)
            if _is_stream_error or (resp_status != 0 and resp_status != 200):
                from prism_router.db import log_error

                error_type = "server_error" if resp_status == 500 else "upstream_error"
                log_error(
                    request_id=request_id,
                    status_code=resp_status,
                    error_type=error_type,
                    error_message=f"HTTP {resp_status}" if not _is_stream_error else "Stream error",
                    client_model=model or "",
                    final_model=final_model_key,
                )

            if dedup_window > 0 and dedup_key_str:
                dedup.complete_by_key(dedup_key_str)
            return response

    # ── 路由 ──
    try:
        route_result = await route(
            model=model,
            messages=messages,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            max_tokens=body.get("max_tokens"),
            settings=settings,
            conversation_history=conversation_history,
        )
    except ValueError as e:
        req_log.error("Routing error: %s", e)
        if dedup_window > 0 and dedup_key_str:
            dedup.complete_by_key(dedup_key_str)
        return _error(400, f"路由失败: {e}", "invalid_request_error")

    cls_method = ""
    if route_result.classification:
        cls_method = route_result.classification.signals.get("method", "")
    req_log.info(
        "Route: %s → %s (tier=%s, cls=%s)",
        model or "prism-auto",
        route_result.model_key,
        route_result.tier,
        cls_method or "passthrough",
    )

    # 保存原始 body（rewrite 前），用于 DB 记录
    _original_body = copy.deepcopy(body)

    # ── 智能改写 ──
    rewrite_meta = None
    if settings.rewriting.enabled or _force_classify_rewrite:
        mode = "classify_and_rewrite" if _force_classify_rewrite else settings.rewriting.mode

        _is_title_request = False
        _skip_keywords = settings.rewriting.skip_rewrite_keywords
        _skip_patterns = settings.rewriting.skip_rewrite_patterns
        for _m in messages:
            if _m.get("role") == "user":
                _content = (_m.get("content") or "")[:500]
                _content_lower = _content.lower()
                for _kw in _skip_keywords:
                    if _kw.lower() in _content_lower:
                        _is_title_request = True
                        req_log.info("Skip rewrite: matched keyword %r", _kw)
                        break
                if not _is_title_request and _skip_patterns:
                    for _pat in _skip_patterns:
                        if re.search(_pat, _content):
                            _is_title_request = True
                            req_log.info("Skip rewrite: matched pattern %r", _pat)
                            break
                if _is_title_request:
                    break

        if not _is_title_request:
            if mode == "rewrite_only":
                if model and model in settings.models:
                    # 已知模型 → 正常改写
                    body["messages"], rewrite_meta = await rewrite_prompt(
                        messages,
                        target_model_key=route_result.model_key,
                        settings=settings,
                    )
                    messages = body["messages"]
                elif model:
                    # 指定了模型但不在配置中 → 会话内首次给用户选择
                    choice = await _prompt_choice_with_timeout(
                        ctx,
                        messages,
                        "rewrite_only_model",
                        model,
                        req_log,
                    )
                    if choice == "route_rewrite":
                        body["messages"], rewrite_meta = await rewrite_prompt(
                            messages,
                            target_model_key=route_result.model_key,
                            settings=settings,
                        )
                        messages = body["messages"]
                else:
                    # 未指定 model → 自动分类 + 改写
                    req_log.info("rewrite_only: no model specified, classifying first")
                    body["messages"], rewrite_meta = await rewrite_prompt(
                        messages,
                        target_model_key=route_result.model_key,
                        settings=settings,
                    )
                    messages = body["messages"]

            elif mode == "classify_and_rewrite":
                # 用户指定了真实模型（非 prism-auto 等虚拟模型）且配置为跳过
                if model and model not in VIRTUAL_MODELS and not settings.rewriting.rewrite_when_model_specified:
                    req_log.info("Skip rewrite: model specified by user (%s)", model)
                else:
                    body["messages"], rewrite_meta = await rewrite_prompt(
                        messages,
                        target_model_key=route_result.model_key,
                        settings=settings,
                    )
                    messages = body["messages"]

        if rewrite_meta and rewrite_meta.rewritten:
            _rewrite_cache.put(rewrite_meta.original_text, rewrite_meta.rewritten_text)
            req_log.info(
                "Rewrite OK: %s → %s | latency: %.0fms",
                rewrite_meta.rewriter_key,
                rewrite_meta.target_key,
                rewrite_meta.latency_ms,
            )
            if content_logging_enabled:
                req_log.info("  Original:  %s", rewrite_meta.original_text[:100])
                req_log.info("  Rewritten: %s", rewrite_meta.rewritten_text[:100])
        elif rewrite_meta and not rewrite_meta.rewritten and rewrite_meta.rewriter_key:
            for _m in reversed(body.get("messages", [])):
                if _m.get("role") == "user":
                    _orig = _m.get("content", "")
                    _cached = _rewrite_cache.get(_orig) if isinstance(_orig, str) else None
                    if _cached:
                        _m["content"] = _cached
                    break
            req_log.info(
                "Rewrite skip: agent loop (rewriter=%s → target=%s)",
                rewrite_meta.rewriter_key,
                rewrite_meta.target_key,
            )

    # 改写后重新提取 last_user_msg（日志应反映实际发送内容）
    if content_logging_enabled and rewrite_meta and rewrite_meta.rewritten:
        for _m in reversed(messages):
            if _m.get("role") == "user":
                _content = _m.get("content", "")
                if isinstance(_content, str):
                    _last_user_msg = _content[:80]
                elif isinstance(_content, list):
                    for _p in _content:
                        if isinstance(_p, dict) and _p.get("type") == "text":
                            _last_user_msg = _p.get("text", "")[:80]
                            break
                break

    # ── 执行配置 ──
    fallback_cfg = settings.fallback
    proxy = settings.server.proxy
    tried_tiers = [route_result.tier]
    cb = get_circuit_breaker(settings)
    final_model_key = route_result.model_key

    # ── Context Window 预检查 ──
    if not check_context_window(messages, body.get("tools"), route_result.model_config):
        req_log.warning(
            "Context window exceeded for %s (estimated > %d tokens)",
            route_result.model_key,
            route_result.model_config.context if route_result.model_config else 0,
        )
        if fallback_cfg.enabled:
            for fallback_target in get_fallback_chain(route_result.tier, settings, fallback_override):
                if not fallback_target:
                    continue
                if "/" in fallback_target:
                    fb_result = build_fallback_from_model(fallback_target, settings, route_result)
                else:
                    fb_result = build_fallback_route_result(fallback_target, settings, route_result)
                if fb_result is None:
                    continue
                if check_context_window(messages, body.get("tools"), fb_result.model_config):
                    req_log.info("Context fallback: %s → %s", route_result.model_key, fb_result.model_key)
                    route_result = fb_result
                    tried_tiers = [route_result.tier]
                    final_model_key = route_result.model_key
                    break

    # ── 缓存检查（仅非流式） ──
    cache = _get_cache()
    if cache and not stream:
        cached = cache.get(model or "prism-auto", messages, body.get("tools"))
        if cached:
            req_log.info("Cache hit for model=%s", model or "prism-auto")
            if dedup_window > 0 and dedup_key_str:
                dedup.complete_by_key(dedup_key_str)
            resp = JSONResponse(content=cached)
            resp.headers["X-Cache"] = "HIT"
            return resp

    # ── 本地模型并发限制 ──
    sem = _get_semaphore(route_result.model_key)

    # ── 执行请求（带 Fallback） ──
    try:
        if sem:
            async with sem:
                response = await execute_request(
                    route_result,
                    body,
                    stream,
                    fallback_cfg,
                    proxy,
                    settings=settings,
                    tried_tiers=tried_tiers,
                    fallback_override=fallback_override,
                )
        else:
            response = await execute_request(
                route_result,
                body,
                stream,
                fallback_cfg,
                proxy,
                settings=settings,
                tried_tiers=tried_tiers,
                fallback_override=fallback_override,
            )
    except Exception as e:
        req_log.error("Request failed: %s", e, exc_info=True)
        response = _error(502, str(e), "upstream_error")
        if dedup_window > 0 and dedup_key_str:
            dedup.complete_by_key(dedup_key_str)

    elapsed = time.time() - start_time
    is_streaming_response = isinstance(response, StreamingResponse)
    status = 0 if is_streaming_response else (response.status_code if hasattr(response, "status_code") else 0)
    req_log.debug("Response: %s | %d | %.2fs", route_result.model_key, status, elapsed)

    # ── 记录成功/失败到熔断器 ──
    _local_fb = getattr(response, "_local_fallback_done", False)
    _cloud_fb = getattr(response, "_cloud_fallback_done", False)
    _fallback_occurred = _local_fb or _cloud_fb
    final_model_key = getattr(response, "_final_model_key", "")
    stream_ok = is_streaming_response and not getattr(response, "_had_error", False)
    json_ok = isinstance(response, JSONResponse) and response.status_code == 200
    is_success = stream_ok or json_ok

    if _fallback_occurred:
        # 发生降级：原始模型记录失败（防止被降级模型的成功抹除）
        if route_result.channel_id:
            cb.record_failure(route_result.channel_id, is_local=route_result.is_local)
        # 实际回答的降级模型记录其成功/失败
        if final_model_key:
            final_ch_id = model_key_to_channel_id(final_model_key)
            if is_success:
                cb.record_success(final_ch_id, latency_ms=elapsed * 1000)
            else:
                is_loc = settings.is_local(final_model_key) if settings else False
                cb.record_failure(final_ch_id, is_local=is_loc)
    elif route_result.channel_id:
        if is_success:
            cb.record_success(route_result.channel_id, latency_ms=elapsed * 1000)
        else:
            cb.record_failure(route_result.channel_id, is_local=route_result.is_local)
            if isinstance(response, JSONResponse):
                # 解析响应体以检测致命错误（如 quota 不足、账户禁用等）
                error_body = None
                raw_body = bytes(response.body)
                try:
                    error_body = json.loads(raw_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    error_body = raw_body.decode("utf-8", errors="replace")
                if is_fatal_error(response.status_code, error_body):
                    cb.disable_channel(route_result.channel_id)

    # Fallback：仅在启用、非流式、且返回可重试错误时触发
    # 检查：local.py / cloud.py 是否已经完成了 fallback（避免重复 fallback）
    if (
        fallback_cfg.enabled
        and not _fallback_occurred
        and not stream
        and isinstance(response, JSONResponse)
        and should_retry(response.status_code, fallback_cfg.retry_on)
    ):
        for fallback_target in get_fallback_chain(route_result.tier, settings, fallback_override):
            if not fallback_target:
                continue
            if tried_tiers and fallback_target in tried_tiers:
                continue
            if "/" in fallback_target:
                fb_result = build_fallback_from_model(fallback_target, settings, route_result)
            else:
                fb_result = build_fallback_route_result(fallback_target, settings, route_result)
            if fb_result is None:
                continue
            req_log.warning(
                "⚠ FALLBACK: %s → %s (reason: HTTP %s)",
                route_result.model_key,
                fb_result.model_key,
                response.status_code,
            )
            tried_tiers.append(fb_result.tier)
            try:
                fb_start = time.time()
                response = await execute_request(
                    fb_result,
                    body,
                    stream,
                    fallback_cfg,
                    proxy,
                    settings=settings,
                    tried_tiers=tried_tiers,
                    fallback_override=fallback_override,
                )
                fb_elapsed = time.time() - fb_start
            except Exception as e:
                req_log.error("Fallback request failed: %s", e, exc_info=True)
                # 内部错误（代码 bug）返回 500，网络/上游错误返回 502
                response = _error(502, str(e), "upstream_error")
                if fb_result.channel_id:
                    cb.record_failure(fb_result.channel_id, is_local=fb_result.is_local)
                continue
            if fb_result.channel_id:
                if isinstance(response, JSONResponse) and response.status_code == 200:
                    cb.record_success(fb_result.channel_id, latency_ms=fb_elapsed * 1000)
                else:
                    cb.record_failure(fb_result.channel_id, is_local=fb_result.is_local)
            final_model_key = fb_result.model_key
            if not (isinstance(response, JSONResponse) and should_retry(response.status_code, fallback_cfg.retry_on)):
                break

    if isinstance(response, JSONResponse) and response.status_code != 200 and len(tried_tiers) > 1:
        req_log.warning(
            "All fallbacks exhausted for tier '%s', tried: %s, last status: %s",
            route_result.tier,
            " → ".join(tried_tiers),
            response.status_code,
        )

    # ── 写入缓存（仅非流式成功响应） ──
    if cache and not stream and isinstance(response, JSONResponse) and response.status_code == 200:
        try:
            cache.put(model or "prism-auto", messages, body.get("tools"), json.loads(bytes(response.body)))
        except Exception as e:
            req_log.debug("Cache put failed: %s", e)

    # ── 记录请求日志 ──
    from prism_router.db import log_request, log_request_bodies, log_request_detail

    resp_status = 0 if is_streaming_response else (response.status_code if hasattr(response, "status_code") else 0)
    resp_usage = {}
    response_body = None
    if is_streaming_response:
        resp_usage = getattr(response, "_usage", {})
    elif isinstance(response, JSONResponse):
        try:
            response_body = json.loads(bytes(response.body))
            resp_usage = response_body.get("usage", {})
        except Exception as e:
            req_log.debug("Failed to parse response body: %s", e)

    prompt_tokens = resp_usage.get("prompt_tokens", 0)
    completion_tokens = resp_usage.get("completion_tokens", 0)
    if hasattr(response, "_final_model_key"):
        final_model_key = response._final_model_key

    final_cfg = settings.get_model_config(final_model_key) or route_result.model_config
    cost = 0
    if final_cfg:
        cost = (prompt_tokens * final_cfg.cost_in + completion_tokens * final_cfg.cost_out) / 1000

    classifier_method = "passthrough"
    classifier_model = ""
    if route_result.classification:
        cls = route_result.classification
        classifier_method = cls.signals.get("method", "unknown")
        if classifier_method == "model":
            classifier_model = settings.routing.classifier_model
            classifier_method = f"model ({settings.routing.classifier})"
    elif model and not model.startswith("prism-"):
        classifier_method = "passthrough"

    upstream_latency_ms = 0

    time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    tools_upgraded = (
        route_result.classification is not None
        and route_result.classification.signals.get("corrected") == "tools_upgrade"
    )

    request_id = log_request(
        client_model=model,
        final_model=final_model_key,
        tier=route_result.tier,
        classifier_method=classifier_method,
        classifier_model=classifier_model,
        status_code=resp_status,
        latency_ms=elapsed * 1000,
        upstream_latency_ms=upstream_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
        is_stream=stream,
        is_fallback=len(tried_tiers) > 1,
        tried_tiers=tried_tiers if len(tried_tiers) > 1 else None,
        is_rewritten=rewrite_meta is not None and rewrite_meta.rewritten,
        rewriter_latency_ms=rewrite_meta.latency_ms if rewrite_meta else 0,
        request_id=request_id,
        last_user_msg=_last_user_msg,
        msg_count=_msg_count,
        has_tools=bool(body.get("tools")),
        tool_count=len(body.get("tools") or []),
        tools_upgraded=tools_upgraded,
    )

    # 请求体/响应体写入 DB（用 rewrite 前的原始 body）
    body_max_size = getattr(settings.logging, "body_max_size", 10240)
    full_body_max_size = getattr(settings.logging, "full_body_max_size", 5120)
    if content_logging_enabled:
        log_request_bodies(
            request_id=request_id,
            request_body=_original_body,
            response_body=response_body,
            body_max_size=body_max_size,
            full_body_max_size=full_body_max_size,
        )

    # 改写详情写入 DB
    if content_logging_enabled and rewrite_meta and rewrite_meta.rewritten:
        from prism_router.db import log_rewrite

        log_rewrite(
            request_id=request_id,
            rewriter_key=rewrite_meta.rewriter_key,
            target_key=rewrite_meta.target_key,
            latency_ms=rewrite_meta.latency_ms,
            original_text=rewrite_meta.original_text,
            rewritten_text=rewrite_meta.rewritten_text,
        )

    # 错误详情写入 DB
    _is_stream_error = is_streaming_response and getattr(response, "_had_error", False)
    if _is_stream_error or (resp_status != 0 and resp_status != 200):
        from prism_router.db import log_error

        error_type = "server_error" if resp_status == 500 else "upstream_error"
        log_error(
            request_id=request_id,
            status_code=resp_status,
            error_type=error_type,
            error_message=f"HTTP {resp_status}" if not _is_stream_error else "Stream error",
            tried_tiers=tried_tiers if len(tried_tiers) > 1 else None,
            client_model=model or "",
            final_model=final_model_key,
        )
    log_request_detail(
        request_id=request_id,
        time_str=time_str,
        client_model=model,
        final_model=final_model_key,
        tier=route_result.tier,
        classifier_method=classifier_method,
        classifier_model=classifier_model,
        latency_ms=elapsed * 1000,
        upstream_latency_ms=upstream_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        status_code=resp_status,
        is_stream=stream,
        is_fallback=len(tried_tiers) > 1,
        tried_tiers=tried_tiers if len(tried_tiers) > 1 else None,
        tools_upgraded=tools_upgraded,
        is_rewritten=rewrite_meta is not None and rewrite_meta.rewritten,
        rewriter_latency_ms=rewrite_meta.latency_ms if rewrite_meta else 0,
        rewriter_model=rewrite_meta.rewriter_key if rewrite_meta else "",
        target_model=rewrite_meta.target_key if rewrite_meta else "",
        last_user_msg=_last_user_msg,
    )

    if settings.context.enabled:
        ctx.record(
            messages=messages,
            tier=route_result.tier,
            model_key=route_result.model_key,
            success=resp_status == 200,
            has_tools=bool(body.get("tools")),
        )

    if dedup_window > 0 and dedup_key_str:
        dedup.complete_by_key(dedup_key_str)

    return response
