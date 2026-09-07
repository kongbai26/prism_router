# Prism Router

[English](README.md) | [简体中文](README_zh.md)

面向 OpenAI 生态的智能 LLM 网关：支持任务复杂度路由、零开销直连透传与针对性 Prompt 改写，兼顾本地/云端调度与 API 大幅降本。

## 特性

- OpenAI Chat Completions 与 Responses API 兼容
- 复杂度路由、模型别名、工具调用感知与透传模式
- Ollama、llama.cpp 及其他 OpenAI 兼容本地/云端端点
- 多模型降级、渠道熔断、上下文压缩与可选 Prompt 改写
- 默认安全：仅监听本机、验证上游 TLS、不开启内容日志或宿主机配置写入

## 快速开始

要求：Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .

cp config.example.yaml config.yaml
cp .env.example .env
```

编辑 `.env`，至少填写一个上游模型的密钥；如需对外网或局域网非本机地址提供服务，还必须设置路由服务密钥：

```bash
PRISM_ROUTER_API_KEY="请替换为高强度随机密钥"
OPENAI_API_KEY="请替换为上游密钥"
```

然后检查 `config.yaml` 中的模型、路由 tier 和端点地址，再启动服务：

```bash
prism-router server
```

本机客户端示例：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4671/v1
export OPENAI_API_KEY="$PRISM_ROUTER_API_KEY"
```

```bash
curl http://127.0.0.1:4671/v1/chat/completions \
  -H "Authorization: Bearer $PRISM_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"prism-auto","messages":[{"role":"user","content":"你好"}]}'
```

## 安全与隐私

- 示例配置默认监听 `127.0.0.1`。CLI 会拒绝在非本机地址启动未配置 `server.api_key` 的服务。
- 上游 HTTPS 默认验证证书。仅在受信任的自签名内网服务场景下，才可将 `server.verify_ssl` 改为 `false`。
- `/health` 可匿名访问；模型和请求接口需要普通 API Key。`/admin/*` 默认禁用，启用后使用独立的 `X-Prism-Admin-Key`。
- 默认只记录路由、耗时、token 与状态等元数据，不保存用户 Prompt、模型响应、工具参数或大工具输出。
- 需要排障时，可显式启用 `logging.content_logging_enabled`、`logging.artifact_storage_enabled`、`logging.tool_call_cache.persist` 或 `rewriting.write_rewrite_logs`。这些选项会写入敏感内容，请限定运行目录权限并设置保留期。
- `server.write_codex_catalog` 默认关闭；只有显式开启时，服务才会修改宿主机的 Codex 模型目录配置。

请勿提交 `.env`、`config.yaml`、`logs/` 或任何真实密钥。它们已经列入 `.gitignore`。

## 配置概览

`config.example.yaml` 是唯一的配置模板。模型键使用 `<provider>/<model>` 格式，例如：

```yaml
models:
  openai/gpt-4o-mini:
    model: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1/chat/completions"
    api_key: "${OPENAI_API_KEY}"
    source: "cloud"
    tier: mid
    supports_tools: true
```

可用的虚拟模型：

- `prism-auto`：按复杂度自动选择 tier。
- `prism-simple`、`prism-mid`、`prism-complex`：强制指定 tier。

完整路由策略、回退链、缓存、熔断、改写与日志参数均在模板中有中文注释。

## 管理接口

设置 `PRISM_ROUTER_ADMIN_API_KEY` 后，才会启用管理接口：

```bash
curl http://127.0.0.1:4671/admin/channels \
  -H "X-Prism-Admin-Key: $PRISM_ROUTER_ADMIN_API_KEY"
```

未配置该密钥时，`/admin/*` 返回 404。

## 开发与发布验证

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy prism_router/settings.py prism_router/routing.py prism_router/health.py
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip check
```

## 致谢与灵感

Prism Router 在设计与架构过程中，借鉴了 LLM 与网关生态中许多优秀开源项目的宝贵经验与设计思想，特别致谢：

- [RouteLLM](https://github.com/lm-sys/RouteLLM) (LMSYS) — 任务复杂度评估与动态模型成本-质量权衡体系的先驱探索。
- [LiteLLM](https://github.com/BerriAI/litellm) — 多上游统一代理规范与渠道容灾调度的优秀参考。
- [Ollama](https://github.com/ollama/ollama) 与 [llama.cpp](https://github.com/ggerganov/llama.cpp) — 为本地轻量模型接入与私有化推理提供了坚实易用的底座。
- [FastAPI](https://fastapi.tiangolo.com/) 与 [HTTPX](https://www.python-httpx.org/) — 为本网关高并发、低延迟的异步流式转发提供了卓越性能基石。

## 许可证与贡献

本项目采用 [Apache License 2.0](LICENSE)。提交贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING_zh.md)；安全问题请遵循 [SECURITY.md](SECURITY_zh.md) 的私密披露流程。
