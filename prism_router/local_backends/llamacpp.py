"""llama.cpp HTTP Server 适配器

llama.cpp 暴露 OpenAI 兼容 API，直接转发请求即可。
典型端点：http://localhost:8080/v1/chat/completions
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from prism_router.local_backends.base import LocalBackend
from prism_router.local_backends.pool import get_client

logger = logging.getLogger("prism_router.llamacpp")


class LlamaCppBackend(LocalBackend):
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        # 透传其他参数
        for key in ("temperature", "top_p", "max_tokens", "stop", "n", "seed", "tools", "tool_choice"):
            if key in kwargs:
                payload[key] = kwargs[key]
        # 转发其他参数（如 thinking_params 等）
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        url = self.base_url

        if stream:
            return self._stream(url, payload, timeout, connect_timeout)
        else:
            return await self._non_stream(url, payload, timeout, connect_timeout)

    async def _non_stream(
        self, url: str, payload: dict, timeout: float | None = None, connect_timeout: float | None = None
    ) -> dict:
        client = get_client(timeout=timeout or self.timeout, connect_timeout=connect_timeout)
        resp = await client.post(url, json=payload)
        try:
            resp.raise_for_status()
            try:
                data = resp.json()
                return data if isinstance(data, dict) else {}
            except Exception:
                text = resp.text[:500] if resp.text else "(empty body)"
                raise RuntimeError(f"Upstream returned non-JSON: {text}") from None
        finally:
            await resp.aclose()

    async def _stream(
        self, url: str, payload: dict, timeout: float | None = None, connect_timeout: float | None = None
    ) -> AsyncIterator[str]:
        """SSE 透传：逐行读取上游响应，yield 给调用方

        连接/超时/HTTP 错误会抛异常（调用方可 catch 后 fallback），
        数据传输中的错误返回 error SSE event。
        """
        done_sent = False
        try:
            client = get_client(timeout=timeout or self.timeout, connect_timeout=connect_timeout)
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"Upstream {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        if "data: [DONE]" in line:
                            done_sent = True
                        yield line + "\n\n"
                    elif line.strip() == "":
                        continue
        except (httpx.TransportError, httpx.HTTPStatusError):
            raise  # 抛出，让调用方 fallback
        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
        if not done_sent:
            yield "data: [DONE]\n\n"

    async def health_check(self) -> bool:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(self.base_url)
            health_url = f"{parsed.scheme}://{parsed.netloc}/health"
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                resp = await client.get(health_url)
                return resp.status_code == 200
        except Exception:
            return False
