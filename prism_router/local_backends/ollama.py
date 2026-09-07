"""Ollama 适配器

Ollama 0.1.20+ 支持 /v1/chat/completions（OpenAI 兼容）。
回退到 /api/chat 时需要格式转换。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from prism_router.local_backends.base import LocalBackend
from prism_router.local_backends.pool import get_client

logger = logging.getLogger("prism_router.ollama")


class OllamaBackend(LocalBackend):
    def __init__(self, config=None, *, base_url: str = ""):
        if config is not None:
            self.base_url = config.base_url.rstrip("/")
        else:
            self.base_url = base_url.rstrip("/")
        self.timeout = 120

    async def chat_completions(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict | AsyncIterator[str]:
        timeout = kwargs.pop("timeout", self.timeout)
        connect_timeout = kwargs.pop("connect_timeout", None)
        url = self.base_url
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        for key in ("temperature", "top_p", "max_tokens", "stop", "n", "seed", "tools", "tool_choice"):
            if key in kwargs:
                payload[key] = kwargs[key]
        # 转发其他参数（如 thinking_params 等）
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        if stream:
            return self._stream(url, payload, timeout, connect_timeout)
        else:
            return await self._non_stream(url, payload, timeout, connect_timeout)

    @staticmethod
    def _to_ollama_native(payload: dict) -> dict:
        """将 OpenAI 格式载荷转换为 Ollama /api/chat 原生格式"""
        native = {
            "model": payload.get("model", ""),
            "messages": payload.get("messages", []),
            "stream": payload.get("stream", False),
        }
        options = {}
        if "temperature" in payload:
            options["temperature"] = payload["temperature"]
        if "top_p" in payload:
            options["top_p"] = payload["top_p"]
        if "max_tokens" in payload:
            options["num_predict"] = payload["max_tokens"]
        if "stop" in payload:
            options["stop"] = payload["stop"]
        if "seed" in payload:
            options["seed"] = payload["seed"]
        if options:
            native["options"] = options
        if "tools" in payload:
            native["tools"] = payload["tools"]
        if "tool_choice" in payload:
            native["tool_choice"] = payload["tool_choice"]
        return native

    async def _non_stream(
        self, url: str, payload: dict, timeout: float | None = None, connect_timeout: float | None = None
    ) -> dict:
        client = get_client(timeout=timeout or self.timeout, connect_timeout=connect_timeout)
        resp = await client.post(url, json=payload)
        try:
            if resp.status_code == 404:
                await resp.aclose()
                alt_url = url.replace("/v1/chat/completions", "/api/chat")
                native_payload = self._to_ollama_native(payload)
                resp = await client.post(alt_url, json=native_payload)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                text = resp.text[:500] if resp.text else "(empty body)"
                raise RuntimeError(f"Upstream returned non-JSON: {text}") from None
            return self._normalize_response(data)
        finally:
            await resp.aclose()

    async def _stream(
        self, url: str, payload: dict, timeout: float | None = None, connect_timeout: float | None = None
    ) -> AsyncIterator[str]:
        done_sent = False
        stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        try:
            client = get_client(timeout=timeout or self.timeout, connect_timeout=connect_timeout)
            # 先尝试 /v1 端点
            resp = await client.send(
                client.build_request("POST", url, json=payload),
                stream=True,
            )
            if resp.status_code == 404:
                await resp.aclose()
                alt_url = url.replace("/v1/chat/completions", "/api/chat")
                native_payload = self._to_ollama_native(payload)
                resp = await client.send(
                    client.build_request("POST", alt_url, json=native_payload),
                    stream=True,
                )
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Upstream {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            try:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        if "data: [DONE]" in line:
                            done_sent = True
                        yield line + "\n\n"
                    elif line.strip():
                        try:
                            chunk = json.loads(line)
                            yield f"data: {json.dumps(self._normalize_chunk(chunk, stream_id))}\n\n"
                        except json.JSONDecodeError:
                            continue
            finally:
                await resp.aclose()
        except (httpx.TransportError, httpx.HTTPStatusError):
            raise  # 抛出，让调用方 fallback
        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
        if not done_sent:
            yield "data: [DONE]\n\n"

    def _normalize_response(self, data: dict) -> dict:
        """确保响应是 OpenAI 格式"""
        if "choices" in data:
            return data
        # Ollama /api/chat 格式转换
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": data.get("message", {}),
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }

    def _normalize_chunk(self, data: dict, stream_id: str = "") -> dict:
        """将 Ollama chunk 转为 OpenAI streaming chunk 格式"""
        if "choices" in data:
            return data
        message = data.get("message", {})
        return {
            "id": stream_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": message.get("content", "")},
                    "finish_reason": None if not data.get("done") else "stop",
                }
            ],
        }

    async def health_check(self) -> bool:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(self.base_url)
            host_root = f"{parsed.scheme}://{parsed.netloc}"
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                resp = await client.get(f"{host_root}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
