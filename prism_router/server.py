"""FastAPI 服务入口：app 初始化、中间件、端点定义"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from prism_router.settings import Settings, model_key_to_channel_id

logger = logging.getLogger("prism_router")

_settings: list[Settings | None] = [None]


def reset_settings():
    """重置全局设置实例（用于测试）"""
    _settings[0] = None
    from prism_router.handlers import reset_handlers

    reset_handlers()


def get_settings() -> Settings:
    if _settings[0] is None:
        from prism_router.settings import load_settings

        _settings[0] = load_settings()
    settings = _settings[0]
    assert settings is not None
    return settings


def _error(status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": None, "param": None}},
    )


# ── 探活 ──


async def _probe_classifier(settings: Settings) -> None:
    from prism_router.status import set_classifier_ok

    if settings.routing.classifier != "local" or not settings.routing.classifier_model:
        set_classifier_ok(True)
        return
    cfg = settings.get_model_config(settings.routing.classifier_model)
    if not cfg:
        set_classifier_ok(True)
        return
    from prism_router.local_backends import get_backend

    backend = get_backend(cfg.backend or "", cfg.base_url)
    ok = await backend.health_check()
    set_classifier_ok(ok)
    log = logging.getLogger("prism_router")
    if ok:
        log.info("Classifier backend: OK (%s)", settings.routing.classifier_model)
    else:
        log.warning("Classifier backend: UNAVAILABLE (%s) — falling back to rules", settings.routing.classifier_model)


async def _probe_rewriter(settings: Settings) -> None:
    from prism_router.status import set_rewriter_ok

    if not settings.rewriting.enabled or not settings.rewriting.rewriter_model:
        set_rewriter_ok(True)
        return
    cfg = settings.get_model_config(settings.rewriting.rewriter_model)
    if not cfg:
        set_rewriter_ok(False)
        logger.warning("Rewriter backend: UNAVAILABLE (%s) — rewriting disabled", settings.rewriting.rewriter_model)
        return
    from prism_router.local_backends import get_backend

    backend = get_backend(cfg.backend or "", cfg.base_url)
    ok = await backend.health_check()
    set_rewriter_ok(ok)
    if ok:
        logger.info("Rewriter backend: OK (%s)", settings.rewriting.rewriter_model)
    else:
        logger.warning("Rewriter backend: UNAVAILABLE (%s) — rewriting disabled", settings.rewriting.rewriter_model)


async def _periodic_probe(settings: Settings) -> None:
    from prism_router.status import is_classifier_ok, set_classifier_ok

    if not settings.routing.classifier_probe_enabled:
        return
    interval = settings.routing.classifier_probe_interval
    while True:
        await asyncio.sleep(interval)
        if settings.routing.classifier != "local" or not settings.routing.classifier_model:
            continue
        cfg = settings.get_model_config(settings.routing.classifier_model)
        if not cfg:
            continue
        from prism_router.local_backends import get_backend

        backend = get_backend(cfg.backend or "", cfg.base_url)
        ok = await backend.health_check()
        prev = is_classifier_ok()
        if ok and not prev:
            logger.info("Classifier backend recovered: %s", settings.routing.classifier_model)
        elif not ok and prev:
            logger.warning("Classifier backend went down: %s", settings.routing.classifier_model)
        set_classifier_ok(ok)


async def _periodic_probe_rewriter(settings: Settings) -> None:
    from prism_router.status import is_rewriter_ok, set_rewriter_ok

    if not settings.rewriting.probe_enabled:
        return
    interval = settings.rewriting.probe_interval
    while True:
        await asyncio.sleep(interval)
        if not settings.rewriting.enabled or not settings.rewriting.rewriter_model:
            continue
        cfg = settings.get_model_config(settings.rewriting.rewriter_model)
        if not cfg:
            continue
        from prism_router.local_backends import get_backend

        backend = get_backend(cfg.backend or "", cfg.base_url)
        ok = await backend.health_check()
        prev = is_rewriter_ok()
        if ok and not prev:
            logger.info("Rewriter backend recovered: %s", settings.rewriting.rewriter_model)
        elif not ok and prev:
            logger.warning("Rewriter backend went down: %s", settings.rewriting.rewriter_model)
        set_rewriter_ok(ok)


# ── model_catalog_json 写入 ──
# 参考 codex-app-transfer model_catalog.rs
# 写入 ~/.codex-app-transfer/config.json，Codex CLI 读取此文件获取 auto_compact_token_limit

# AUTO_COMPACT_TRIGGER_PERCENT 定义在 responses_api.py 中，通过延迟导入使用


def _write_model_catalog(settings: Settings) -> None:
    """写入 model_catalog_json 到 ~/.codex-app-transfer/config.json，
    并更新 ~/.codex/config.toml 指向此文件。

    参考 codex-app-transfer apply.rs:166-186
    """
    import json

    try:
        config_path = Path.home() / ".prism-router" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取现有配置（如果有）
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 构建 models 列表
        models = []
        for model_key, cfg in settings.models.items():
            context_window = getattr(cfg, "context", 128000) or 128000
            from prism_router.responses_api import AUTO_COMPACT_TRIGGER_PERCENT

            auto_compact = context_window * AUTO_COMPACT_TRIGGER_PERCENT // 100
            models.append(
                {
                    "slug": model_key,
                    "display_name": model_key,
                    "description": f"Routed through Prism Router as {model_key}.",
                    "context_window": context_window,
                    "max_context_window": context_window,
                    "effective_context_window_percent": 95,
                    "auto_compact_token_limit": auto_compact,
                }
            )

        existing["models"] = models
        config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote model_catalog_json to %s (%d models)", config_path, len(models))

        # 更新 ~/.codex/config.toml，写入 model_catalog_json 指向我们的文件
        codex_config = Path.home() / ".codex" / "config.toml"
        if codex_config.exists():
            content = codex_config.read_text(encoding="utf-8")
            catalog_line = f'model_catalog_json = "{config_path}"'
            if "model_catalog_json" in content:
                # 替换现有行
                import re

                content = re.sub(
                    r'model_catalog_json\s*=\s*"[^"]*"',
                    catalog_line,
                    content,
                )
            else:
                # 追加新行
                content = content.rstrip() + f"\n{catalog_line}\n"
            codex_config.write_text(content, encoding="utf-8")
            logger.info("Updated %s with model_catalog_json", codex_config)
        else:
            # 创建 config.toml
            codex_config.parent.mkdir(parents=True, exist_ok=True)
            codex_config.write_text(
                f'model_catalog_json = "{config_path}"\n',
                encoding="utf-8",
            )
            logger.info("Created %s with model_catalog_json", codex_config)
    except Exception as e:
        logger.warning("Failed to write model_catalog_json: %s", e)


# ── Lifespan ──

_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app):
    # 延迟注册 chat_completions 端点（避免 handlers↔server 循环导入）
    from prism_router.handlers import chat_completions
    from prism_router.responses_api import compact_endpoint, models_endpoint, responses_endpoint

    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/v1/responses", responses_endpoint, methods=["POST"])
    app.add_api_route("/responses", responses_endpoint, methods=["POST"])
    app.add_api_route("/v1/responses/compact", compact_endpoint, methods=["POST"])
    app.add_api_route("/responses/compact", compact_endpoint, methods=["POST"])
    app.add_api_route("/v1/models", models_endpoint, methods=["GET"])
    app.add_api_route("/models", models_endpoint, methods=["GET"])

    settings = get_settings()

    # 仅在用户明确启用时写入 Codex 配置，避免修改宿主机个人配置。
    if settings.server.write_codex_catalog:
        _write_model_catalog(settings)

    from prism_router.routing import get_circuit_breaker

    cb = get_circuit_breaker(settings)
    cb.load_state()

    # 启动后台任务并追踪
    if settings.routing.classifier == "local" and settings.routing.classifier_model:
        t1 = asyncio.create_task(_probe_classifier(settings))
        t2 = asyncio.create_task(_periodic_probe(settings))
        _background_tasks.add(t1)
        _background_tasks.add(t2)
        t1.add_done_callback(_background_tasks.discard)
        t2.add_done_callback(_background_tasks.discard)

    # 改写器探测
    if settings.rewriting.enabled and settings.rewriting.rewriter_model:
        t3 = asyncio.create_task(_probe_rewriter(settings))
        t4 = asyncio.create_task(_periodic_probe_rewriter(settings))
        _background_tasks.add(t3)
        _background_tasks.add(t4)
        t3.add_done_callback(_background_tasks.discard)
        t4.add_done_callback(_background_tasks.discard)

    _check_models: set[str] = set()
    routing = settings.routing
    rewriting = settings.rewriting

    if routing.mode == "manual":
        if routing.tiers:
            for tier_entries in routing.tiers.values():
                for entry in tier_entries:
                    if entry.model:
                        _check_models.add(entry.model)
        for tier_key in ("simple", "mid", "complex"):
            k = getattr(routing, tier_key, "")
            if k:
                _check_models.add(k)
    else:
        _check_models.update(routing.model_pool)

    if rewriting.enabled and rewriting.rewriter_model:
        _check_models.add(rewriting.rewriter_model)

    for model_key in _check_models:
        cfg = settings.get_model_config(model_key)
        if cfg is None:
            logger.warning("Health check SKIP: %s (not in models config)", model_key)
            continue
        base_url = cfg.base_url.rstrip("/")
        _test_url = base_url
        if _test_url.endswith("/chat/completions"):
            _test_url = _test_url.rsplit("/chat/completions", 1)[0]
        _test_url = _test_url.rstrip("/")
        if not _test_url.endswith("/v1"):
            _test_url += "/v1"
        _test_url += "/models"
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        try:
            async with httpx.AsyncClient(timeout=5) as _client:
                resp = await _client.get(_test_url, headers=headers)
                if resp.status_code < 400:
                    logger.info("Health check OK: %s (%d)", model_key, resp.status_code)
                else:
                    logger.warning("Health check FAIL: %s (%d)", model_key, resp.status_code)
        except Exception as _e:
            logger.warning("Health check FAIL: %s (%s)", model_key, type(_e).__name__)

    yield

    # 优雅关闭：取消后台任务 → 保存状态 → 关闭连接池
    for task in _background_tasks:
        task.cancel()
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)
    _background_tasks.clear()

    try:
        cb.save_state()
    except Exception:
        pass
    from prism_router.local_backends.pool import close_all

    try:
        await close_all()
    except Exception:
        pass
    from prism_router.db import close_db

    try:
        close_db()
    except Exception:
        pass


# ── App ──

app = FastAPI(title="Prism Router", version="0.1.0", lifespan=lifespan)

VIRTUAL_MODELS = ["prism-auto", "prism-simple", "prism-mid", "prism-complex"]


# ── 鉴权中间件 ──


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    settings = get_settings()
    path = request.url.path.rstrip("/")
    if path == "/health":
        return await call_next(request)
    if path.startswith("/admin/"):
        admin_key = settings.server.admin_api_key
        supplied_key = request.headers.get("X-Prism-Admin-Key", "")
        if not admin_key:
            return _error(404, "Admin endpoints are disabled")
        if not supplied_key or not hmac.compare_digest(supplied_key, admin_key):
            return _error(401, "Invalid admin API key", "authentication_error")
    elif settings.server.api_key:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], settings.server.api_key):
            return _error(401, "Invalid API key", "authentication_error")
    try:
        return await call_next(request)
    except asyncio.CancelledError:
        return JSONResponse(
            status_code=503, content={"error": {"message": "Server shutting down", "type": "server_error"}}
        )


# ── 端点 ──


@app.get("/health")
async def health():
    from prism_router import __version__

    return {"status": "ok", "version": __version__}


@app.get("/v1/models")
@app.get("/models")
async def list_available_models():
    settings = get_settings()
    data = []
    for key, cfg in settings.models.items():
        data.append(
            {
                "id": key,
                "object": "model",
                "created": int(time.time()),
                "owned_by": key.split("/")[0],
                "permission": [],
                "root": key,
                "parent": None,
            }
        )
    for virtual in VIRTUAL_MODELS:
        data.append(
            {
                "id": virtual,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "prism-router",
                "permission": [],
                "root": virtual,
                "parent": None,
            }
        )
    return {"object": "list", "data": data}


@app.get("/admin/channels")
async def admin_channels():
    settings = get_settings()
    from prism_router.routing import get_circuit_breaker

    cb = get_circuit_breaker(settings)
    from prism_router.status import is_classifier_ok

    channels = []
    for model_key, cfg in settings.models.items():
        ch_id = model_key_to_channel_id(model_key)
        health = cb.get_health(ch_id)
        status = cb.get_channel_status(ch_id)
        channels.append(
            {
                "model_key": model_key,
                "channel_id": ch_id,
                "status": status.value,
                "state": health.state.value,
                "success_rate": round(health.success_rate, 3),
                "avg_latency_ms": round(health.avg_latency, 1),
                "failure_count": health.failure_count,
                "consecutive_successes": health.consecutive_successes,
                "cooldown_remaining_s": max(0, round(health.cooldown_until - time.time(), 1))
                if health.cooldown_until != float("inf")
                else -1,
                "tier": cfg.tier,
                "source": cfg.source,
                "base_url": cfg.base_url,
            }
        )

    return {
        "channels": channels,
        "classifier_ok": is_classifier_ok(),
        "config": {
            "circuit_breaker_enabled": settings.circuit_breaker.enabled,
            "fallback_enabled": settings.fallback.enabled,
        },
    }


@app.post("/admin/channels/{model_key:path}/reset")
async def admin_reset_channel(model_key: str):
    settings = get_settings()
    from prism_router.routing import get_circuit_breaker

    cb = get_circuit_breaker(settings)
    cfg = settings.get_model_config(model_key)
    if cfg is None:
        return JSONResponse(status_code=404, content={"error": {"message": f"Unknown model: {model_key}"}})
    canonical_key = next((k for k in settings.models if k.lower() == model_key.lower()), model_key)
    ch_id = model_key_to_channel_id(canonical_key)
    cb.reset_channel(ch_id)
    return {"status": "ok", "model_key": canonical_key}


# ── init_app ──


def init_app(settings: Settings) -> None:
    _settings[0] = settings
    level = getattr(logging, settings.logging.level, logging.INFO)

    prism_logger = logging.getLogger("prism_router")
    prism_logger.setLevel(level)
    if not prism_logger.handlers:
        from prism_router.logger import CompactFormatter, make_console_handler

        stream_handler = make_console_handler()
        prism_logger.addHandler(stream_handler)

    # 文件 handler（独立去重，避免重复调用 init_app 时产生重复日志）
    if not any(isinstance(h, logging.FileHandler) for h in prism_logger.handlers):
        import time as _time

        from prism_router.logger import CompactFormatter

        log_dir = Path("logs/server")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"server.{_time.strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(CompactFormatter("%(asctime)s [%(levelname)s] %(message)s"))
        prism_logger.addHandler(file_handler)

    prism_logger.propagate = False

    if settings.logging.db_path:
        from prism_router.db import init_logger

        init_logger(settings.logging.db_path, retention_days=settings.logging.retention_days)
