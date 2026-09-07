"""本地模型处理：通过适配器转发到 base_url"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from prism_router.fallback import (
    build_fallback_from_model,
    build_fallback_route_result,
    get_fallback_chain,
    should_retry,
)
from prism_router.local_backends import get_backend
from prism_router.routing import RouteResult
from prism_router.server import _error

logger = logging.getLogger("prism_router.local")


async def wrap_stream_with_usage(gen: Any, usage_container: dict):
    """包装流式 generator，拦截 SSE 中的 usage 数据供日志记录。"""
    try:
        async for chunk in gen:
            yield chunk
            if "usage" in chunk:
                try:
                    for line in chunk.split("\n"):
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            data = json.loads(line[6:])
                            if "usage" in data:
                                usage_container.update(data["usage"])
                except Exception:
                    pass
    except (asyncio.CancelledError, GeneratorExit):
        pass
    except Exception as e:
        logger.error("Stream wrapper error: %s", e)
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"


async def _build_local_stream_response(sse_gen, is_retry: bool = False) -> JSONResponse | StreamingResponse:
    """构建本地模型流式响应。

    is_retry=True 时，TransportError/HTTPStatusError 会 re-raise（让重试循环检测失败）。
    is_retry=False 时，这些错误被捕获并 yield SSE 错误帧。
    """
    if not is_retry:
        try:
            first_chunk = await sse_gen.__anext__()
        except StopAsyncIteration:
            return _error(502, "Upstream returned empty response", "upstream_error")
    else:
        first_chunk = await sse_gen.__anext__()

    _local_error_state = [False]
    _done_sent = [False]
    _log_suffix = "retry " if is_retry else ""

    async def _gen():
        nonlocal first_chunk
        yield first_chunk
        try:
            async for chunk in sse_gen:
                yield chunk
                if "data: [DONE]" in chunk:
                    _done_sent[0] = True
        except (asyncio.CancelledError, GeneratorExit):
            _local_error_state[0] = True
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            _local_error_state[0] = True
            if is_retry:
                raise
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"
            if not _done_sent[0]:
                yield "data: [DONE]\n\n"
        except Exception as e:
            _local_error_state[0] = True
            logger.error("Local stream %serror: %s", _log_suffix, e, exc_info=True)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"
        if not _done_sent[0]:
            yield "data: [DONE]\n\n"

    usage_container: dict = {}
    wrapped_gen = wrap_stream_with_usage(_gen(), usage_container)
    resp = StreamingResponse(
        wrapped_gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
    resp._usage = usage_container  # type: ignore[attr-defined]
    resp._had_error = False  # type: ignore[attr-defined]

    async def _tracked_gen():
        try:
            async for chunk in wrapped_gen:
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            resp._had_error = True  # type: ignore[attr-defined]
        except Exception:
            resp._had_error = True  # type: ignore[attr-defined]
        if _local_error_state[0]:
            resp._had_error = True  # type: ignore[attr-defined]

    resp.body_iterator = _tracked_gen()
    return resp


async def handle_local(
    route_result: RouteResult,
    body: dict,
    stream: bool,
    settings=None,
    tried_tiers: list[str] | None = None,
    fallback_override: list[str] | None = None,
    fallback_cfg=None,
    proxy: str = "",
    has_tool_evidence_fn=None,
    strip_tool_calls_fn=None,
    execute_request_fn=None,
) -> JSONResponse | StreamingResponse:
    """本地模型：通过适配器转发到 base_url

    流式模式下，连接/超时/HTTP 错误会自动 fallback 到其他模型。
    """
    from prism_router.handlers import execute_request as _default_exec

    _exec = execute_request_fn or _default_exec

    base_url = route_result.base_url
    if not base_url:
        return _error(503, f"No base_url configured for {route_result.model_key}", "server_error")

    logger.info("Local → %s (%s) tier=%s", route_result.model_key, base_url, route_result.tier)

    backend_name = ""
    if route_result.model_config and route_result.model_config.backend:
        backend_name = route_result.model_config.backend
    backend = get_backend(backend_name, base_url)

    try:
        kwargs: dict[str, Any] = {}
        # 转发 body 中所有参数（排除内部字段和已单独处理的字段）
        _skip_keys = {"messages", "model", "stream", "tools", "tool_choice"}
        for key, value in body.items():
            if key.startswith("prism_") or key in _skip_keys:
                continue
            kwargs[key] = value

        should_pass_tools = route_result.use_tools
        if not should_pass_tools:
            messages = body.get("messages", [])
            if has_tool_evidence_fn and has_tool_evidence_fn(messages):
                if body.get("tools"):
                    should_pass_tools = True
                    logger.info("Keep tools for local model: messages contain tool evidence")
                elif strip_tool_calls_fn:
                    logger.warning(
                        "Messages contain tool_calls/function_call but body has no tools definitions. "
                        "Stripping tool_calls from messages."
                    )
                    body["messages"] = strip_tool_calls_fn(messages)
        if should_pass_tools and body.get("tools"):
            kwargs["tools"] = body["tools"]
            if body.get("tool_choice"):
                kwargs["tool_choice"] = body["tool_choice"]

        # 传递配置的超时时间给 backend（覆盖 backend 默认的 120s）
        connect_timeout: float = fallback_cfg.connect_timeout_seconds if fallback_cfg else 8.0
        if route_result.model_config and route_result.model_config.connect_timeout is not None:
            connect_timeout = route_result.model_config.connect_timeout
        kwargs["connect_timeout"] = connect_timeout
        if fallback_cfg:
            kwargs["timeout"] = fallback_cfg.timeout_seconds

        if stream:
            sse_gen = await backend.chat_completions(
                model=route_result.model_id,
                messages=body["messages"],
                stream=True,
                **kwargs,
            )
            return await _build_local_stream_response(sse_gen, is_retry=False)
        else:
            result = await backend.chat_completions(
                model=route_result.model_id,
                messages=body["messages"],
                stream=False,
                **kwargs,
            )
            return JSONResponse(content=result)
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        last_error: Exception = e
        error_type = type(e).__name__
        error_detail = str(e) or repr(e)
        logger.warning("Local request failed: %s [%s] %s", route_result.model_key, error_type, error_detail)

        # 区分连接层错误（连不上/握手超时）与业务读取错误
        is_connect_err = isinstance(e, (httpx.ConnectTimeout, httpx.ConnectError))
        if is_connect_err:
            max_retries = fallback_cfg.connect_max_retries if fallback_cfg else 1
        else:
            max_retries = fallback_cfg.max_retries if fallback_cfg else 3

        # ── 重试指定模型 ──
        for retry_i in range(max_retries):
            wait = 0.5 * (retry_i + 1)
            retry_tag = "connect retry" if is_connect_err else "retry"
            logger.info(
                "Retrying %s (%d/%d) in %.1fs (%s)...",
                route_result.model_key,
                retry_i + 1,
                max_retries,
                wait,
                retry_tag,
            )
            await asyncio.sleep(wait)
            try:
                if stream:
                    sse_gen = await backend.chat_completions(
                        model=route_result.model_id,
                        messages=body["messages"],
                        stream=True,
                        **kwargs,
                    )
                    return await _build_local_stream_response(sse_gen, is_retry=True)
                else:
                    result = await backend.chat_completions(
                        model=route_result.model_id,
                        messages=body["messages"],
                        stream=False,
                        **kwargs,
                    )
                    return JSONResponse(content=result)
            except Exception as retry_e:
                last_error = retry_e
                logger.warning("Retry %d/%d failed: [%s] %s", retry_i + 1, max_retries, type(retry_e).__name__, retry_e)

        # ── 重试都失败 → 降级（如果允许）──
        if settings and settings.fallback.enabled:
            logger.warning("⚠ All retries exhausted for %s, trying fallback chain", route_result.model_key)
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
                if fb_result.model_key == route_result.model_key:
                    continue
                logger.warning(
                    "⚠ FALLBACK: %s → %s (after %d retries)", route_result.model_key, fb_result.model_key, max_retries
                )
                if tried_tiers is not None:
                    tried_tiers.append(fb_result.tier)
                resp = await _exec(
                    fb_result, body, stream, fallback_cfg, proxy, settings, tried_tiers, fallback_override
                )
                resp._final_model_key = fb_result.model_key  # type: ignore[union-attr]
                resp._local_fallback_done = True  # type: ignore[union-attr]  # 标记：local.py 已完成 fallback，handlers.py 不要重复
                if hasattr(resp, "headers"):
                    resp.headers["X-Prism-Fallback"] = "true"
                    resp.headers["X-Prism-Fallback-From"] = route_result.model_key
                    resp.headers["X-Prism-Fallback-To"] = fb_result.model_key
                if isinstance(resp, JSONResponse) and should_retry(resp.status_code, settings.fallback.retry_on):
                    continue
                return resp
        return _error(502, str(last_error), "server_error")
    except Exception as e:
        logger.error("Local backend unexpected error: %s", e, exc_info=True)
        # 内部编程错误返回 500，其他未预期异常返回 502
        return _error(502, str(e), "upstream_error")
