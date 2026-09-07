"""全局 httpx 连接池 — 按 proxy 键复用 TCP 连接"""

from __future__ import annotations

import httpx

_clients: dict[str, httpx.AsyncClient] = {}


def get_client(
    timeout: float | int = 120,
    proxy: str = "",
    verify: bool = True,
    connect_timeout: float | int | None = None,
) -> httpx.AsyncClient:
    """获取或创建 httpx.AsyncClient，按 proxy+verify+timeout+connect_timeout 键复用"""
    c_timeout = 8.0 if connect_timeout is None else float(connect_timeout)
    key = f"{proxy or '__default__'}:{verify}:{timeout}:{c_timeout}"
    client = _clients.get(key)
    if client is None or client.is_closed:
        if client is not None and client.is_closed:
            del _clients[key]
        kw: dict = {
            "timeout": httpx.Timeout(connect=c_timeout, read=timeout, write=c_timeout, pool=c_timeout),
            "verify": verify,
        }
        if proxy:
            kw["proxy"] = proxy
        else:
            kw["trust_env"] = False
        client = httpx.AsyncClient(**kw)
        _clients[key] = client
    return client


def get_cloud_client(
    timeout: float | int = 60,
    proxy: str = "",
    verify: bool = True,
    connect_timeout: float | int | None = None,
) -> httpx.AsyncClient:
    """获取云端 API 客户端（默认验证 TLS 证书，按 proxy+verify 键复用）"""
    return get_client(timeout=timeout, proxy=proxy, verify=verify, connect_timeout=connect_timeout)


async def close_all() -> None:
    """关闭所有连接（用于优雅退出）"""
    for client in _clients.values():
        if not client.is_closed:
            await client.aclose()
    _clients.clear()
