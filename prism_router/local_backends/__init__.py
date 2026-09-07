"""本地推理后端适配器"""

from __future__ import annotations

import logging

from prism_router.local_backends.base import LocalBackend

logger = logging.getLogger("prism_router")

# 按 (name, base_url) 缓存 backend 实例，避免每次请求重建
_backend_cache: dict[str, LocalBackend] = {}


def get_backend(name: str, base_url: str) -> LocalBackend:
    """根据名称和 base_url 获取（或创建）本地后端适配器"""
    key = f"{name}:{base_url}"
    backend = _backend_cache.get(key)
    if backend is not None:
        return backend
    if name == "ollama":
        from prism_router.local_backends.ollama import OllamaBackend

        backend = OllamaBackend(base_url=base_url)
    else:
        if name != "llamacpp":
            logger.warning("Unknown backend '%s', falling back to llamacpp adapter", name)
        from prism_router.local_backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend(base_url=base_url)
    _backend_cache[key] = backend
    return backend
