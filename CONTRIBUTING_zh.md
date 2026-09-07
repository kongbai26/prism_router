# Contributing to Prism Router

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING_zh.md)

感谢你对 Prism Router 的关注与贡献！Prism Router 是一个致力于降低大模型 API 成本、提供智能多层级路由与提示词优化的开源项目。

---

## 🛠 开发环境搭建

### 1. 克隆代码与安装依赖

项目要求 Python >= 3.10，建议使用虚拟环境进行开发：

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 以可编辑模式安装，包含开发工具
pip install -e ".[dev]"
```

### 2. 本地配置

从模版复制配置文件并根据本地环境调整：

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

> **注意**：`config.yaml` 和 `.env` 已被 `.gitignore` 忽略，请切勿将个人的 API Key 或内网服务地址提交到 Git。

---

## 🧪 代码规范与质量检查

在提交 Pull Request 前，请确保代码风格、类型检查和打包验证全部通过。

### 1. 代码风格检查与格式化 (Ruff)

项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行极速 Lint 与代码格式化：

```bash
# 检查 Lint
ruff check .

# 检查格式
ruff format --check .

# 自动修复 Lint 与格式化
ruff check --fix .
ruff format .
```

### 2. 类型检查 (Mypy)

核心模块需通过 Mypy 静态类型检查：

```bash
mypy prism_router/settings.py prism_router/routing.py prism_router/health.py
```

### 3. 打包验证

```bash
# 构建并验证 wheel
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip check
```

---

## 📝 编码约定

1. **文档与注释语言**：代码内注释、Docstring 以及配置注释统一采用**中文**。
2. **配置隔离**：请使用 `config.example.yaml` 与 `.env.example` 创建本地配置，切勿提交真实 API Key 或内网地址。
3. **模型查找**：模型标识符查询必须保持大小写不敏感（统一使用 `settings.get_model_config`）。

---

## 🚀 提交 Pull Request

1. Fork 本仓库到个人 GitHub 账号。
2. 创建特性分支：`git checkout -b feature/your-feature-name` 或 `git checkout -b fix/your-fix-name`。
3. 提交变更并编写清晰的 Commit Message。
4. 确保本地 `ruff check`、`ruff format --check`、`mypy` 和 wheel 构建均通过。
5. 推送分支并向 `main` 分支发起 Pull Request。
