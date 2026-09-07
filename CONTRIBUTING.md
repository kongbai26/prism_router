# Contributing to Prism Router

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING_zh.md)

Thank you for your interest in contributing to Prism Router! Prism Router is an open-source project dedicated to reducing LLM API costs while providing intelligent multi-tier routing, fault tolerance, and prompt optimization.

---

## 🛠 Development Setup

### 1. Clone the Repository & Install Dependencies

Prism Router requires Python >= 3.10. We recommend developing inside a virtual environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Local Configuration

Copy the example configuration files and adjust them to your local environment:

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

> **Important**: `config.yaml` and `.env` are ignored by `.gitignore`. Never commit your real API keys or internal network addresses to version control.

---

## 🧪 Code Standards & Quality Checks

Before submitting a Pull Request, make sure all linting, formatting, type checking, and packaging checks pass.

### 1. Linting and Code Formatting (Ruff)

This project uses [Ruff](https://docs.astral.sh/ruff/) for high-performance linting and formatting:

```bash
# Run lint check
ruff check .

# Check code formatting
ruff format --check .

# Automatically fix lint issues and format files
ruff check --fix .
ruff format .
```

### 2. Type Checking (Mypy)

Core modules must pass Mypy static type checking:

```bash
mypy prism_router/settings.py prism_router/routing.py prism_router/health.py
```

### 3. Package Verification

```bash
# Build and verify the wheel distribution
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip check
```

---

## 📝 Coding Conventions

1. **Comments and Documentation**: Internal code comments, docstrings, and config template annotations are primarily maintained in Chinese for consistency with the original codebase.
2. **Configuration Isolation**: Always use `config.example.yaml` and `.env.example` as templates for local development. Never check in production secrets or internal endpoints.
3. **Model Key Resolution**: Model key lookups must be case-insensitive (standardized via `settings.get_model_config`).

---

## 🚀 Submitting a Pull Request

1. Fork this repository to your personal GitHub account.
2. Create a feature branch: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-fix-name`.
3. Commit your changes with clear and descriptive commit messages.
4. Ensure `ruff check`, `ruff format --check`, `mypy`, and package build pass locally.
5. Push your branch to GitHub and open a Pull Request against the `main` branch.
