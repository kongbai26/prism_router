"""云端模型处理：httpx 透传（流式/非流式）"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse
from tenacity import retry, retry_if_exception, wait_exponential

from prism_router.fallback import should_retry
from prism_router.local_backends.pool import get_cloud_client
from prism_router.routing import RouteResult
from prism_router.server import _error
from prism_router.settings import ModelConfig

logger = logging.getLogger("prism_router.cloud")


def build_cloud_payload(
    route_result: RouteResult, body: dict, has_tool_evidence_fn, strip_tool_calls_fn
) -> dict[str, Any]:
    """透传 body，覆盖 model，根据 use_tools 决定是否保留 tools。
    如果 messages 中有 tool 证据但 body 无 tools，移除孤立的 tool_calls。
    """
    payload = dict(body)
    # 过滤内部扩展字段，防止泄露给上游
    for key in list(payload.keys()):
        if key.startswith("prism_"):
            del payload[key]
    payload["model"] = route_result.model_id
    if not route_result.use_tools:
        messages = body.get("messages", [])
        if has_tool_evidence_fn(messages):
            if payload.get("tools"):
                logger.info("Keep tools in payload: messages contain tool evidence")
            else:
                logger.warning(
                    "Messages contain tool_calls/function_call but body has no tools definitions. "
                    "Stripping tool_calls from messages. Client should include tools field."
                )
                payload["messages"] = strip_tool_calls_fn(messages)
        else:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            payload.pop("function_call", None)
    return payload


def cloud_headers(route_result: RouteResult) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if route_result.api_key:
        headers["Authorization"] = f"Bearer {route_result.api_key}"
    return headers


def build_cloud_url(base_url: str) -> str:
    """构建云端 API URL — 自动补全端点路径（兼容 OpenAI SDK 行为）"""
    url = base_url.rstrip("/")
    scheme_end = url.find("://")
    if scheme_end != -1:
        scheme = url[: scheme_end + 3]
        path = url[scheme_end + 3 :]
        while "//" in path:
            path = path.replace("//", "/")
        url = scheme + path
    else:
        while "//" in url:
            url = url.replace("//", "/")
    known_endpoints = ("/chat/completions", "/embeddings", "/completions", "/models")
    if not any(url.endswith(ep) for ep in known_endpoints):
        url = url + "/chat/completions"
    return url


async def test_stream_connection(url: str, payload: dict, headers: dict, client_kw: dict, timeout: int) -> None:
    """尝试建立流式连接，成功则立即关闭。用于在返回 StreamingResponse 前预检连接。"""
    payload_copy = {**payload, "stream": True}
    read_timeout = timeout
    proxy = client_kw.get("proxy", "")
    verify = client_kw.get("verify", False)
    connect_timeout = client_kw.get("connect_timeout")
    client = get_cloud_client(timeout=read_timeout, proxy=proxy, verify=verify, connect_timeout=connect_timeout)
    async with client.stream("POST", url, json=payload_copy, headers=headers) as resp:
        if resp.status_code >= 400:
            error_body = await resp.aread()
            error_text = error_body.decode(errors="replace")[:500]
            logger.error("Upstream precheck %d: %s | url=%s", resp.status_code, error_text, url)
            await resp.aclose()
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}: {error_text}",
                request=resp.request,
                response=resp,
            )
        await resp.aclose()


async def _safe_stream(gen: AsyncIterator) -> AsyncIterator:
    """包装流式 generator，防止客户端断连时 CancelledError 冒泡到 uvicorn"""
    try:
        async for chunk in gen:
            yield chunk
    except (asyncio.CancelledError, GeneratorExit):
        pass


async def passthrough_stream(
    url: str,
    payload: dict,
    headers: dict,
    client_kw: dict,
    timeout: int,
    _usage_container: dict | None = None,
    first_token_timeout: float = 0,
) -> StreamingResponse:
    """流式透传：上游 SSE 原样 pipe 给客户端。

    first_token_timeout > 0 时：预触发模式，阻塞等首 chunk，超时抛异常（供上层回退）。
    first_token_timeout <= 0 时：原有行为，立即返回 StreamingResponse。
    """
    payload = dict(payload)
    payload["stream"] = True
    read_timeout = timeout
    proxy = client_kw.get("proxy", "")
    verify = client_kw.get("verify", False)
    connect_timeout = client_kw.get("connect_timeout")
    _error_state = [False]

    # ── 预触发模式：阻塞等首 chunk，超时抛异常 ──
    if first_token_timeout > 0:
        client = get_cloud_client(timeout=read_timeout, proxy=proxy, verify=verify, connect_timeout=connect_timeout)
        connect_ctx = client.stream("POST", url, json=payload, headers=headers)
        resp = await connect_ctx.__aenter__()
        try:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                error_text = error_body.decode(errors="replace")[:500]
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {error_text}",
                    request=resp.request,
                    response=resp,
                )
            first_chunk = await asyncio.wait_for(resp.aiter_bytes().__anext__(), timeout=first_token_timeout)
        except Exception:
            await connect_ctx.__aexit__(None, None, None)
            raise

        async def generate():
            done_sent = False
            try:
                text = first_chunk.decode("utf-8", errors="replace")
                if "data: [DONE]" in text:
                    done_sent = True
                yield text
                if _usage_container is not None and "usage" in text:
                    try:
                        for line in text.split("\n"):
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                data = json.loads(line[6:])
                                if "usage" in data:
                                    _usage_container.update(data["usage"])
                    except Exception:
                        pass
                async for chunk in resp.aiter_bytes():
                    text = chunk.decode("utf-8", errors="replace")
                    if "data: [DONE]" in text:
                        done_sent = True
                    yield text
                    if _usage_container is not None and "usage" in text:
                        try:
                            for line in text.split("\n"):
                                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                    data = json.loads(line[6:])
                                    if "usage" in data:
                                        _usage_container.update(data["usage"])
                        except Exception:
                            pass
            except httpx.TimeoutException:
                _error_state[0] = True
                logger.warning("Stream timeout for %s", url)
                yield f"data: {json.dumps({'error': {'message': 'Upstream timeout', 'type': 'timeout_error'}})}\n\n"
            except (httpx.ConnectError, httpx.NetworkError, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
                _error_state[0] = True
                logger.warning("Stream connection error for %s: %s", url, e)
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"
            except httpx.HTTPStatusError as e:
                _error_state[0] = True
                error_msg = f"Upstream {e.response.status_code}"
                try:
                    error_body = e.response.text
                    if error_body:
                        err_json = json.loads(error_body)
                        if "error" in err_json:
                            upstream_err = err_json["error"]
                            if isinstance(upstream_err, dict):
                                error_msg = upstream_err.get("message", error_msg)
                            elif isinstance(upstream_err, str):
                                error_msg = upstream_err
                except Exception:
                    pass
                yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'upstream_error'}})}\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception as e:
                _error_state[0] = True
                logger.error("Stream error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            finally:
                await connect_ctx.__aexit__(None, None, None)
            if not done_sent:
                yield "data: [DONE]\n\n"

    # ── 传统模式：立即返回 StreamingResponse ──
    else:

        async def generate():
            done_sent = False
            try:
                client = get_cloud_client(
                    timeout=read_timeout, proxy=proxy, verify=verify, connect_timeout=connect_timeout
                )
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code == 401:
                        _error_state[0] = True
                        yield f"data: {json.dumps({'error': {'message': 'Authentication failed', 'type': 'authentication_error'}})}\n\n"
                        yield "data: [DONE]\n\n"
                        done_sent = True
                        return
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        text = chunk.decode("utf-8", errors="replace")
                        if "data: [DONE]" in text:
                            done_sent = True
                        yield text
                        if _usage_container is not None and "usage" in text:
                            try:
                                for line in text.split("\n"):
                                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                        data = json.loads(line[6:])
                                        if "usage" in data:
                                            _usage_container.update(data["usage"])
                            except Exception:
                                pass
            except httpx.TimeoutException:
                _error_state[0] = True
                logger.warning("Stream timeout for %s", url)
                yield f"data: {json.dumps({'error': {'message': 'Upstream timeout', 'type': 'timeout_error'}})}\n\n"
            except (httpx.ConnectError, httpx.NetworkError, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
                _error_state[0] = True
                logger.warning("Stream connection error for %s: %s", url, e)
                raise
            except httpx.HTTPStatusError as e:
                _error_state[0] = True
                error_msg = f"Upstream {e.response.status_code}"
                error_body = ""
                try:
                    error_body = e.response.text
                    if error_body:
                        err_json = json.loads(error_body)
                        if "error" in err_json:
                            upstream_err = err_json["error"]
                            if isinstance(upstream_err, dict):
                                error_msg = upstream_err.get("message", error_msg)
                            elif isinstance(upstream_err, str):
                                error_msg = upstream_err
                except Exception:
                    error_body = getattr(e.response, "text", "")
                logger.error(
                    "Upstream %d: %s | body=%s",
                    e.response.status_code,
                    error_msg,
                    error_body[:300] if error_body else "",
                )
                yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'upstream_error', 'code': None, 'param': None}})}\n\n"
            except httpx.TransportError as e:
                _error_state[0] = True
                logger.warning("Stream transport error for %s: %s", url, e)
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"
            except Exception as e:
                _error_state[0] = True
                logger.error("Stream error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            if not done_sent:
                yield "data: [DONE]\n\n"

    original_gen = generate()

    async def _tracked_generate():
        try:
            async for chunk in original_gen:
                yield chunk
        except Exception:
            response._had_error = True  # type: ignore[attr-defined]
        if _error_state[0]:
            response._had_error = True  # type: ignore[attr-defined]

    response = StreamingResponse(
        _safe_stream(_tracked_generate()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
    response._had_error = False  # type: ignore[attr-defined]
    return response


async def passthrough_non_stream(url: str, payload: dict, headers: dict, client_kw: dict, timeout: int) -> JSONResponse:
    """非流式透传：上游 JSON 原样返回"""
    payload = dict(payload)  # 浅拷贝，避免污染调用方的 dict
    payload["stream"] = False
    proxy = client_kw.get("proxy", "")
    verify = client_kw.get("verify", False)
    connect_timeout = client_kw.get("connect_timeout")
    client = get_cloud_client(timeout=timeout, proxy=proxy, verify=verify, connect_timeout=connect_timeout)
    resp = await client.post(url, json=payload, headers=headers)
    try:
        if resp.status_code == 401:
            return _error(401, "Authentication failed", "authentication_error")
        resp.raise_for_status()
        try:
            return JSONResponse(content=resp.json())
        except Exception:
            text = resp.text[:500] if resp.text else "(empty body)"
            logger.error("Upstream returned non-JSON response: %s", text)
            return _error(502, f"Upstream returned non-JSON: {text}", "upstream_error")
    finally:
        await resp.aclose()


async def handle_cloud(
    route_result: RouteResult,
    body: dict,
    stream: bool,
    fallback_cfg=None,
    proxy: str = "",
    has_tool_evidence_fn=None,
    strip_tool_calls_fn=None,
    settings=None,
    tried_tiers: list[str] | None = None,
    fallback_override: list[str] | None = None,
) -> JSONResponse | StreamingResponse:
    """云端透传：客户端要流式就流式转发，要非流式就非流式转发，不做任何解析/重建。

    流式模式下，首 chunk 前超时会自动切模型（需配置 first_token_timeout > 0）。
    """
    from prism_router.fallback import build_fallback_from_model, build_fallback_route_result, get_fallback_chain
    from prism_router.server import get_settings as _get_settings

    _settings = settings or _get_settings()
    _verify = _settings.server.verify_ssl

    timeout = fallback_cfg.timeout_seconds if fallback_cfg else 120
    max_retries = fallback_cfg.max_retries if fallback_cfg else 3
    backoff_base = fallback_cfg.backoff_base if fallback_cfg else 1.0
    backoff_max = fallback_cfg.backoff_max if fallback_cfg else 30.0
    retry_codes = fallback_cfg.retry_on if fallback_cfg else [429, 500, 502, 503, 504]
    first_token_timeout = getattr(fallback_cfg, "first_token_timeout", 0) if fallback_cfg else 0
    _allow_fallback = fallback_cfg and fallback_cfg.enabled

    def _should_retry(exc: BaseException) -> bool:
        if isinstance(
            exc,
            (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError, httpx.PoolTimeout, httpx.ConnectTimeout),
        ):
            return True
        if isinstance(exc, httpx.HTTPStatusError) and should_retry(exc.response.status_code, retry_codes):
            return True
        return False

    connect_max_retries = fallback_cfg.connect_max_retries if fallback_cfg else 1

    def _build_client_kw(m_cfg: ModelConfig | None = None) -> dict:
        c_timeout = fallback_cfg.connect_timeout_seconds if fallback_cfg else 8.0
        if m_cfg and m_cfg.connect_timeout is not None:
            c_timeout = m_cfg.connect_timeout
        elif route_result.model_config and route_result.model_config.connect_timeout is not None:
            c_timeout = route_result.model_config.connect_timeout
        kw: dict[str, Any] = {"verify": _verify, "connect_timeout": c_timeout}
        if proxy:
            kw["proxy"] = proxy
        else:
            kw["trust_env"] = False
        return kw

    # ── 回退链：遍历 fallback targets ──
    fallback_targets: list[str] = []
    if _allow_fallback and _settings:
        fallback_targets = get_fallback_chain(route_result.tier, _settings, fallback_override)

    last_error: BaseException | None = None
    _cloud_fallback_done = False

    for fallback_idx, fallback_target in enumerate([None, *fallback_targets]):
        # 第一次用原始 route_result，后续用 fallback target
        if fallback_idx == 0:
            current_result = route_result
        else:
            if not _allow_fallback or not _settings:
                break
            if not fallback_target:
                continue
            if tried_tiers and fallback_target in tried_tiers:
                continue
            if "/" in fallback_target:
                fb_result = build_fallback_from_model(fallback_target, _settings, route_result)
            else:
                fb_result = build_fallback_route_result(fallback_target, _settings, route_result)
            if fb_result is None or fb_result.model_key == route_result.model_key:
                continue
            logger.warning(
                "⚠ FALLBACK: %s → %s (reason: timeout)",
                route_result.model_key,
                fb_result.model_key,
            )
            if tried_tiers is not None:
                tried_tiers.append(fb_result.tier)
            current_result = fb_result
            _cloud_fallback_done = True

        logger.info(
            "Cloud %s: %s (tier=%s, model=%s) via %s",
            "stream" if stream else "call",
            current_result.model_key,
            current_result.tier,
            current_result.model_id,
            current_result.base_url,
        )
        payload = build_cloud_payload(current_result, body, has_tool_evidence_fn, strip_tool_calls_fn)
        url = current_result.base_url
        if not url:
            return _error(
                502, f"Invalid base_url for {current_result.model_key}: {current_result.base_url!r}", "server_error"
            )
        headers = cloud_headers(current_result)
        client_kw = _build_client_kw(current_result.model_config)

        def _stop_policy(retry_state: Any) -> bool:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError)):
                return bool(retry_state.attempt_number >= (connect_max_retries + 1))
            return bool(retry_state.attempt_number >= max_retries)

        # 绑定循环变量为默认参数（避免闭包延迟绑定）
        _url, _payload, _headers, _client_kw = url, payload, headers, client_kw

        @retry(
            stop=_stop_policy,
            wait=wait_exponential(multiplier=backoff_base, max=backoff_max),
            retry=retry_if_exception(_should_retry),
            reraise=True,
        )
        async def _do_request(
            _url=_url, _payload=_payload, _headers=_headers, _client_kw=_client_kw
        ) -> JSONResponse | StreamingResponse:
            if stream:
                usage_container: dict = {}
                resp = await passthrough_stream(
                    _url,
                    _payload,
                    _headers,
                    _client_kw,
                    timeout,
                    usage_container,
                    first_token_timeout=first_token_timeout,
                )
                resp._usage = usage_container  # type: ignore[attr-defined]
                return resp
            else:
                return await passthrough_non_stream(_url, _payload, _headers, _client_kw, timeout)

        try:
            resp = await _do_request()
            if _cloud_fallback_done:
                resp._cloud_fallback_done = True  # type: ignore[union-attr]
                resp._final_model_key = current_result.model_key  # type: ignore[union-attr]
                if hasattr(resp, "headers"):
                    resp.headers["X-Prism-Fallback"] = "true"
                    resp.headers["X-Prism-Fallback-From"] = route_result.model_key
                    resp.headers["X-Prism-Fallback-To"] = current_result.model_key
            return resp
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning("Cloud timeout for %s after %d retries", current_result.model_key, max_retries)
            # 有 fallback targets 且是流式 → 继续循环尝试下一个
            if stream and fallback_targets and fallback_idx < len(fallback_targets):
                continue
            return _error(504, "Upstream timeout", "server_error")
        except httpx.HTTPStatusError as e:
            last_error = e
            error_body = ""
            upstream_msg = f"Upstream {e.response.status_code}"
            upstream_type = "upstream_error"
            try:
                error_body = e.response.text
                if error_body:
                    err_json = json.loads(error_body)
                    if "error" in err_json:
                        upstream_err = err_json["error"]
                        if isinstance(upstream_err, dict):
                            upstream_msg = upstream_err.get("message", upstream_msg)
                            upstream_type = upstream_err.get("type", upstream_type)
                        elif isinstance(upstream_err, str):
                            upstream_msg = upstream_err
            except Exception:
                pass
            logger.error("Upstream %d from %s: %s", e.response.status_code, current_result.model_key, upstream_msg)
            # 流式 + 可重试状态码 → 继续回退链
            if (
                stream
                and should_retry(e.response.status_code, retry_codes)
                and fallback_targets
                and fallback_idx < len(fallback_targets)
            ):
                continue
            if upstream_msg != f"Upstream {e.response.status_code}":
                return JSONResponse(
                    status_code=e.response.status_code,
                    content={"error": {"message": upstream_msg, "type": upstream_type, "code": None, "param": None}},
                )
            msg = (
                f"Upstream {e.response.status_code}: {error_body[:500]}"
                if error_body
                else f"Upstream {e.response.status_code}"
            )
            return _error(e.response.status_code, msg, "upstream_error")
        except (httpx.ConnectError, httpx.NetworkError, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
            last_error = e
            # 有 fallback targets 且是流式 → 继续循环尝试下一个
            if stream and fallback_targets and fallback_idx < len(fallback_targets):
                continue
            return _error(502, f"Upstream connection error: {e}", "server_error")
        except Exception as e:
            last_error = e
            logger.error("Cloud request error: %s", e, exc_info=True)
            return _error(502, str(e), "server_error")

    # 回退链耗尽
    return _error(502, str(last_error) if last_error else "All fallback targets exhausted", "server_error")
