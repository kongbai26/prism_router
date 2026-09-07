# Prism Router

[English](README.md) | [简体中文](README_zh.md)

An intelligent LLM gateway for the OpenAI ecosystem: featuring task-complexity routing, zero-overhead direct passthrough, and targeted prompt rewriting, seamlessly orchestrating local and cloud models to slash API costs.

## Features

- **OpenAI Compatible**: Drop-in proxy replacement for OpenAI Chat Completions (`/v1/chat/completions`) and Responses API.
- **Smart Complexity Routing**: Classifies incoming requests (via heuristic rules, local SLM, or cloud models) and dispatches them to the most cost-effective tier (`simple`, `mid`, `complex`).
- **Local & Cloud Backend Support**: Seamless integration with Ollama, llama.cpp, vLLM, and any OpenAI-compatible cloud provider.
- **High Availability & Fault Tolerance**:
  - Multi-tier fallback chains across models.
  - Proactive three-state circuit breaking (CLOSED, OPEN, HALF_OPEN) to isolate failing backends.
  - Configurable connection timeout and fast-fail retries to eliminate blocking hangs.
- **Advanced Request Processing**:
  - Tool-calling awareness: automatically detects tools and upgrades tiers or strips orphaned tool calls when necessary.
  - Prompt rewriting engine: optimizes prompts for target models with session context awareness.
  - In-memory request deduplication and response caching.
- **Security-First by Default**: Binds to `127.0.0.1`, verifies upstream TLS certificates, and never logs prompts or responses unless explicitly enabled.

## Quick Start

### Requirements

Python 3.10 or higher.

### Installation

```bash
# Clone the repository
git clone https://github.com/kongbai26/prism_router.git
cd prism_router

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install .

# Create configuration files from templates
cp config.example.yaml config.yaml
cp .env.example .env
```

### Configuration

Edit `.env` to set your upstream provider keys and router access key:

```bash
PRISM_ROUTER_API_KEY="your-secure-random-token"
OPENAI_API_KEY="your-upstream-api-key"
```

Configure your models, routing tiers, and endpoints in `config.yaml`, then start the gateway:

```bash
prism-router server
```

### Usage Example

Point your OpenAI client or SDK to Prism Router:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4671/v1
export OPENAI_API_KEY="$PRISM_ROUTER_API_KEY"
```

Send a test request:

```bash
curl http://127.0.0.1:4671/v1/chat/completions \
  -H "Authorization: Bearer $PRISM_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prism-auto",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}]
  }'
```

## Security & Privacy

- **Localhost Binding**: Default configuration binds strictly to `127.0.0.1`. The CLI will refuse to start on non-loopback addresses without a configured `server.api_key`.
- **TLS Verification**: Upstream TLS certificate verification is enforced by default (`verify_ssl: true`).
- **Zero Content Logging**: By default, only metadata (route, latency, token count, status code) is recorded. User prompts, model responses, and tool arguments are never persisted.
- **Sensitive Debugging Controls**: Detailed payload logging (`logging.content_logging_enabled`, `logging.artifact_storage_enabled`) must be explicitly opted into.
- **Secrets Protection**: `.env`, `config.yaml`, and `logs/` are strictly ignored by `.gitignore`. Never commit credentials to version control.

## Configuration Overview

Configuration is managed via `config.yaml` (see [config.example.yaml](config.example.yaml) for full commented reference).

```yaml
models:
  openai/gpt-4o-mini:
    model: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1/chat/completions"
    api_key: "${OPENAI_API_KEY}"
    source: "cloud"
    tier: mid
    supports_tools: true
    connect_timeout: 8.0

routing:
  mode: "manual"
  simple: "ollama/qwen2.5:1.5b"
  mid: "openai/gpt-4o-mini"
  complex: "openai/gpt-4o"
  classifier: "local"
  classifier_model: "ollama/qwen2.5:1.5b"

fallback:
  enabled: true
  connect_timeout_seconds: 8.0
  connect_max_retries: 1
  max_retries: 3
  timeout_seconds: 120
```

### Virtual Routing Models

- `prism-auto`: Automatically classifies the request complexity and dispatches to `simple`, `mid`, or `complex`.
- `prism-simple`: Forces routing to the `simple` tier.
- `prism-mid`: Forces routing to the `mid` tier.
- `prism-complex`: Forces routing to the `complex` tier.

## Admin API

When `PRISM_ROUTER_ADMIN_API_KEY` is configured in `.env`, management endpoints become available:

```bash
# Query channel health and circuit breaker status
curl http://127.0.0.1:4671/admin/channels \
  -H "X-Prism-Admin-Key: $PRISM_ROUTER_ADMIN_API_KEY"
```

If the admin key is not configured, all `/admin/*` routes return `404 Not Found`.

## Development & Verification

```bash
# Install development dependencies
pip install -e ".[dev]"

# Lint and format checks
ruff check .
ruff format --check .

# Type checking
mypy prism_router/settings.py prism_router/routing.py prism_router/health.py

# Package build verification
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip check
```

## Acknowledgements

Prism Router is inspired by and builds upon ideas from outstanding open-source projects in the LLM and gateway ecosystem:

- [RouteLLM](https://github.com/lm-sys/RouteLLM) (LMSYS) — Pioneering concepts in task-complexity classification and cost-quality trade-offs.
- [LiteLLM](https://github.com/BerriAI/litellm) — Multi-provider unified proxying patterns and resilient channel fallback designs.
- [Ollama](https://github.com/ollama/ollama) & [llama.cpp](https://github.com/ggerganov/llama.cpp) — High-performance local inference engines enabling seamless private SLM execution.
- [FastAPI](https://fastapi.tiangolo.com/) & [HTTPX](https://www.python-httpx.org/) — Asynchronous foundations powering our high-concurrency, low-latency streaming pipeline.

## License & Contributing

Licensed under the [Apache License 2.0](LICENSE).  
Please review [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests, and consult [SECURITY.md](SECURITY.md) for security vulnerability reporting.
