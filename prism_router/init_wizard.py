"""Prism Router 交互式初始化配置向导

支持通过终端问答完成基础模型配置、生成 .env 与 config.yaml，并具备自适应降级与防呆能力。
"""

from __future__ import annotations

import getpass
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console

console = Console()


@dataclass
class ConfiguredModel:
    """向导中收集到的单个模型配置信息"""

    key: str  # 模型在 config 中的唯一标识，如 "ollama/qwen2.5:7b"
    model_id: str  # 传递给上游的真实 model 名称，如 "qwen2.5:7b"
    source: str  # "local" 或 "cloud"
    base_url: str  # API 端点完整地址
    api_key_ref: str  # 在 yaml 中引用的环境变量占位符，如 "${DEEPSEEK_API_KEY}"
    env_var_name: str  # 环境变量名称，如 "DEEPSEEK_API_KEY"
    env_var_val: str  # 真实的 API Key 值
    backend: str  # 本地后端的名称 ("ollama" / "llamacpp" / "")
    tier: str  # 默认分级 ("simple" / "mid" / "complex")
    description: str  # 中文描述
    supports_tools: bool = True


def mask_key(key: str) -> str:
    """将 API Key 进行脱敏回显，如 sk-1234****abcd"""
    k = key.strip()
    if len(k) <= 8:
        return "********"
    return f"{k[:4]}****{k[-4:]}"


def normalize_local_url(raw_url: str) -> str:
    """规范化本地模型端点 URL，确保包含 /v1/chat/completions"""
    url = raw_url.strip().rstrip("/")
    if not url:
        url = "http://localhost:11434"

    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def test_local_connection(base_url: str) -> bool:
    """快速探测本地端点是否可用（非阻塞轻量检测）"""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host_root = f"{parsed.scheme}://{parsed.netloc}"
        with httpx.Client(timeout=1.5, trust_env=False) as client:
            resp = client.get(f"{host_root}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def configure_model(role_title: str) -> ConfiguredModel | None:
    """
    通用模型配置交互模块：
    1. 选择：本地模型 / 云端模型 / 跳过
    2. 本地：选择服务类型 -> 输入端点与模型名 -> 连通性测试
    3. 云端：选择厂商 -> 输入 API Key -> 设定官方推荐模型名
    """
    console.print(f"\n[bold cyan]── {role_title} ──[/bold cyan]")
    console.print("请选择模型来源：")
    console.print("  [bold cyan]1)[/bold cyan] 本地模型 (如 Ollama / llama.cpp / vLLM / 内网服务)")
    console.print("  [bold cyan]2)[/bold cyan] 云端大模型 (如 DeepSeek / 智谱 AI / 阿里通义 / OpenAI 等)")
    console.print("  [bold cyan]0)[/bold cyan] 跳过此项 [dim][直接回车跳过][/dim]")

    choice = input("请输入选项 [0-2] (默认 0): ").strip()
    if choice in ("", "0"):
        console.print("[dim]已跳过该项配置。[/dim]")
        return None

    # ── 1. 本地模型分支 ──
    if choice == "1":
        console.print("\n  [bold]请选择本地推理框架：[/bold]")
        console.print("    1) Ollama (默认端口 http://localhost:11434)")
        console.print("    2) llama.cpp / vLLM / 自定义端点")
        sub_c = input("    请输入选项 [1-2] (默认 1): ").strip()

        if sub_c == "2":
            endpoint = input("    请输入端点地址 (如 http://127.0.0.1:8080): ").strip()
            endpoint = normalize_local_url(endpoint or "http://127.0.0.1:8080")
            model_name = input("    请输入模型名称 (例如 qwen3.6-35b): ").strip() or "local-model"
            backend = "llamacpp"
        else:
            endpoint = normalize_local_url("http://localhost:11434")
            model_name = input("    请输入模型名称 (默认 qwen2.5:7b): ").strip() or "qwen2.5:7b"
            backend = "ollama"

        # 连通性探测
        console.print("    [dim]正在探测本地连通性...[/dim]", end=" ")
        if test_local_connection(endpoint):
            console.print("[bold green][✓ 连通成功][/bold green]")
        else:
            console.print("[yellow][⚠ 本地暂未响应，服务启动后可启动 Ollama][/yellow]")

        safe_key_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)
        model_key = f"{backend}/{safe_key_name}"

        return ConfiguredModel(
            key=model_key,
            model_id=model_name,
            source="local",
            base_url=endpoint,
            api_key_ref="",
            env_var_name="",
            env_var_val="",
            backend=backend,
            tier="mid",
            description=f"本地 {backend} 模型 ({model_name})",
            supports_tools=True,
        )

    # ── 2. 云端模型分支 ──
    if choice == "2":
        console.print("\n  [bold]请选择云端平台：[/bold]")
        console.print("    1) DeepSeek (deepseek-chat / 极高性价比与推理能力)")
        console.print("    2) 智谱 AI (glm-4.5-flash / 快速高性价比)")
        console.print("    3) 阿里通义千问 (qwen-plus / 中文通用强)")
        console.print("    4) OpenAI (gpt-4o-mini / 国际通用)")
        console.print("    5) 自定义兼容 OpenAI 协议的端点")
        cloud_c = input("    请输入选项 [1-5] (默认 1): ").strip() or "1"

        if cloud_c == "2":
            provider = "zhipu"
            default_model = "glm-4.5-flash"
            base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            env_var = "ZHIPU_API_KEY"
            desc = "智谱 AI 云端主力模型"
        elif cloud_c == "3":
            provider = "aliyun"
            default_model = "qwen-plus"
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            env_var = "DASHSCOPE_API_KEY"
            desc = "阿里通义千问云端大模型"
        elif cloud_c == "4":
            provider = "openai"
            default_model = "gpt-4o-mini"
            base_url = "https://api.openai.com/v1/chat/completions"
            env_var = "OPENAI_API_KEY"
            desc = "OpenAI 官方通用大模型"
        elif cloud_c == "5":
            provider = "custom"
            default_model = input("    请输入模型名称 (例如 my-model): ").strip() or "custom-model"
            base_url = input("    请输入完整 API 端点地址: ").strip() or "https://api.openai.com/v1/chat/completions"
            env_var = "CUSTOM_API_KEY"
            desc = "自定义云端兼容模型"
        else:
            provider = "deepseek"
            default_model = "deepseek-chat"
            base_url = "https://api.deepseek.com/v1/chat/completions"
            env_var = "DEEPSEEK_API_KEY"
            desc = "DeepSeek 官方云端大模型"

        # 输入 API Key
        raw_key = ""
        try:
            raw_key = getpass.getpass(f"    请输入您的 {env_var} (输入已隐藏): ").strip()
        except Exception:
            pass
        if not raw_key:
            raw_key = input(f"    请输入您的 {env_var}: ").strip()

        if raw_key:
            console.print(f"    [bold green]✔ 已识别 API Key:[/bold green] [dim]{mask_key(raw_key)}[/dim]")
        else:
            console.print("    [yellow]⚠ 未输入 API Key，后续请务必在 .env 中补充。[/yellow]")

        model_key = f"{provider}/{default_model}"

        return ConfiguredModel(
            key=model_key,
            model_id=default_model,
            source="cloud",
            base_url=base_url,
            api_key_ref=f"${{{env_var}}}",
            env_var_name=env_var,
            env_var_val=raw_key,
            backend="",
            tier="complex",
            description=desc,
            supports_tools=True,
        )

    console.print("[dim]无效选项，已跳过。[/dim]")
    return None


def _format_model_yaml_block(model: ConfiguredModel) -> str:
    """把单个 ConfiguredModel 格式化为符合 config.yaml 规范的缩进 YAML 块"""
    lines = [
        f'  "{model.key}":',
        f'    model: "{model.model_id}"',
        f'    base_url: "{model.base_url}"',
    ]
    if model.api_key_ref:
        lines.append(f'    api_key: "{model.api_key_ref}"')
    if model.backend:
        lines.append(f'    backend: "{model.backend}"')
    lines.extend(
        [
            f'    source: "{model.source}"',
            f"    context: {128000 if model.source == 'cloud' else 32000}",
            f"    cost_in: {0.001 if model.source == 'cloud' else 0}",
            f"    cost_out: {0.002 if model.source == 'cloud' else 0}",
            f'    tier: "{model.tier}"',
            f'    description: "{model.description}"',
            "    supports_tools: true",
        ]
    )
    return "\n".join(lines)


def run_init_wizard(
    config_path: Path | None = None,
    env_path: Path | None = None,
    example_path: Path | None = None,
) -> bool:
    """
    运行完整的 3 层初始化向导。
    返回 True 表示用户完成了有效配置且确认立即启动；
    返回 False 表示取消、全跳过、或选择暂不启动。
    """
    root_dir = Path(".").resolve()
    target_config = config_path or (root_dir / "config.yaml")
    target_env = env_path or (root_dir / ".env")
    source_example = example_path or (root_dir / "config.example.yaml")

    # ── 1. 防误触覆盖检查 ──
    if target_config.exists():
        console.print(f"\n[bold yellow]⚠ 检测到当前目录已存在配置文件: {target_config.name}[/bold yellow]")
        confirm = input("重新初始化将覆盖现有配置，是否继续？(y/N, 默认 N): ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]已取消初始化向导。[/dim]")
            return False

    console.print("\n" + "═" * 60)
    console.print("[bold cyan]       Prism Router 极简初始化配置向导[/bold cyan]")
    console.print("   （三层清晰配置，每一层均可输入 0 或按回车跳过）")
    console.print("═" * 60)

    # ── 2. 第 1 层：路由/分类模型 (Router) ──
    router_model = configure_model("第 1 层：路由/分类模型 (负责评估提问难度并指路)")
    if router_model:
        console.print(f"[bold green]✔ 已设定路由模型:[/bold green] [bold]{router_model.key}[/bold]")

    # ── 3. 第 2 层：提示词改写模型 (Rewriter) ──
    console.print("\n[bold cyan]── 第 2 层：提示词改写模型 (负责在请求发出前对 Prompt 润色结构化) ──[/bold cyan]")
    rewriter_model: ConfiguredModel | None = None
    if router_model:
        console.print("请选择改写模型来源：")
        console.print(f"  [bold cyan]1)[/bold cyan] 复用第 1 层的路由模型 [{router_model.key}] [dim](推荐)[/dim]")
        console.print("  [bold cyan]2)[/bold cyan] 另加一个新模型来配置")
        console.print("  [bold cyan]0)[/bold cyan] 跳过，暂不开启改写 [dim][直接回车跳过][/dim]")
        c2 = input("请输入选项 [0-2] (默认 1): ").strip()
        if c2 in ("", "1"):
            rewriter_model = router_model
            console.print(f"[bold green]✔ 已复用路由模型作为改写器:[/bold green] [bold]{router_model.key}[/bold]")
        elif c2 == "2":
            rewriter_model = configure_model("配置单独的改写模型")
        else:
            console.print("[dim]已跳过改写模型配置。[/dim]")
    else:
        rewriter_model = configure_model("第 2 层：提示词改写模型")
        if rewriter_model:
            console.print(f"[bold green]✔ 已设定改写模型:[/bold green] [bold]{rewriter_model.key}[/bold]")

    # ── 4. 第 3 层：回答提问的主力模型 (Target Model) ──
    console.print("\n[bold cyan]── 第 3 层：主力回答模型与模型池 (Model Pool) ──[/bold cyan]")
    console.print("[dim]💡 说明：模型池支持同时配置多个本地与云端模型按复杂度分流。[/dim]")
    console.print("[dim]   向导在此仅为您绑定一个默认主力模型，稍后可直接在 config.yaml 中扩充完整模型池！[/dim]\n")
    target_model: ConfiguredModel | None = None

    # 汇总前面已登记的可用模型候选
    candidates: list[ConfiguredModel] = []
    if router_model:
        candidates.append(router_model)
    if rewriter_model and rewriter_model.key not in [m.key for m in candidates]:
        candidates.append(rewriter_model)

    if candidates:
        console.print("请选择真正回答用户提问的主力模型：")
        idx_map: dict[str, ConfiguredModel] = {}
        cur_idx = 1
        for m in candidates:
            label = "路由模型" if router_model and m.key == router_model.key else "改写模型"
            console.print(f"  [bold cyan]{cur_idx})[/bold cyan] 使用【{label}】[{m.key}] 来回答")
            idx_map[str(cur_idx)] = m
            cur_idx += 1

        add_idx = str(cur_idx)
        console.print(f"  [bold cyan]{add_idx})[/bold cyan] 另加一个模型来配置")
        console.print("  [bold cyan]0)[/bold cyan] 跳过，稍后在配置文件中手动配置模型池 [dim][直接回车跳过][/dim]")

        c3 = input(f"请输入选项 [0-{add_idx}] (默认 1): ").strip()
        if c3 in ("", "1") and "1" in idx_map:
            target_model = idx_map["1"]
            console.print(f"[bold green]✔ 已指定主力回答模型:[/bold green] [bold]{target_model.key}[/bold]")
        elif c3 in idx_map:
            target_model = idx_map[c3]
            console.print(f"[bold green]✔ 已指定主力回答模型:[/bold green] [bold]{target_model.key}[/bold]")
        elif c3 == add_idx:
            target_model = configure_model("配置单独的主力回答模型")
        else:
            console.print("[dim]已跳过主力回答模型配置，将引导去 config.yaml 配置模型池。[/dim]")
    else:
        # 前面全跳过了
        console.print("前面尚未登记任何模型，是否现在配置一个主力模型？")
        console.print("  [bold cyan]1)[/bold cyan] 配置一个主力模型")
        console.print("  [bold cyan]0)[/bold cyan] 全部跳过，稍后在 config.yaml 中手动配置 [dim][直接回车跳过][/dim]")
        c3 = input("请输入选项 [0-1] (默认 0): ").strip()
        if c3 == "1":
            target_model = configure_model("主力回答模型")
        else:
            console.print("[dim]已跳过，稍后请在 config.yaml 中配置模型池。[/dim]")

    # ── 5. 汇总检查：是否全部跳过 ──
    all_models: list[ConfiguredModel] = []
    for model_item in (router_model, rewriter_model, target_model):
        if model_item and model_item.key not in [x.key for x in all_models]:
            all_models.append(model_item)

    # 规则 1：如果一个模型都没配（全部跳过）
    if not all_models:
        if source_example.exists():
            shutil.copy(source_example, target_config)
        env_example = root_dir / ".env.example"
        if env_example.exists() and not target_env.exists():
            shutil.copy(env_example, target_env)

        console.print("\n" + "─" * 60)
        console.print("[bold yellow]ℹ 您未在向导中配置任何模型，服务暂无法工作。[/bold yellow]")
        console.print(f"[bold green]✔ 已为您生成官方标准配置文件:[/bold green] {target_config.name}（含完整中文注释）")
        if not target_env.exists():
            target_env.write_text("# Prism Router 环境变量\n", encoding="utf-8")
        console.print(f"[bold green]✔ 已为您创建环境变量文件:[/bold green] {target_env.name}")
        console.print("\n[bold]👉 下一步指引（配置模型池）：[/bold]")
        console.print(f"请使用文本编辑器打开 [bold]{target_config.name}[/bold]：")
        console.print("  1) 在 models: 节点配置您的模型端点与 API Key")
        console.print("  2) 在 routing: 节点设置 simple / mid / complex 路由模型")
        console.print("填好后运行以下命令即可直接启动服务：")
        console.print("  [bold cyan]prism-router server[/bold cyan]")
        console.print("─" * 60 + "\n")
        return False

    # ── 6. 生成配置文件 ──
    # 读取模板
    raw_template = ""
    if source_example.exists():
        raw_template = source_example.read_text(encoding="utf-8")
    else:
        raw_template = (root_dir / "config.yaml").read_text(encoding="utf-8")

    # 确定自适应模式
    if router_model and rewriter_model and target_model:
        mode_val = "classify_and_rewrite"
    elif router_model and target_model and not rewriter_model:
        mode_val = "classify_only"
    elif rewriter_model and not router_model:
        mode_val = "rewrite_only"
    else:
        mode_val = "passthrough"

    # 路由字段替换
    cls_mode = router_model.source if router_model else "rules"
    cls_key = router_model.key if router_model else ""

    # 替换 routing 关键配置
    raw_template = re.sub(
        r'classifier:\s*".*?"',
        f'classifier: "{cls_mode}"',
        raw_template,
        count=1,
    )
    raw_template = re.sub(
        r'classifier_model:\s*".*?"',
        f'classifier_model: "{cls_key}"',
        raw_template,
        count=1,
    )

    # 替换 simple / mid / complex（若未指定 target_model 且无 router_model 则清空，避免残留示例模型）
    ans_key = target_model.key if target_model else (router_model.key if router_model else "")
    raw_template = re.sub(r'simple:\s*".*?"', f'simple: "{ans_key}"', raw_template, count=1)
    raw_template = re.sub(r'mid:\s*".*?"', f'mid: "{ans_key}"', raw_template, count=1)
    raw_template = re.sub(r'complex:\s*".*?"', f'complex: "{ans_key}"', raw_template, count=1)

    # 替换 rewriting 关键配置
    rewriter_key_str = rewriter_model.key if rewriter_model else ""
    rewriter_enabled_str = "true" if rewriter_model else "false"

    raw_template = re.sub(
        r"rewriting:\s*\n\s*enabled:\s*(?:true|false)",
        f"rewriting:\n  enabled: {rewriter_enabled_str}",
        raw_template,
        count=1,
    )
    raw_template = re.sub(
        r'mode:\s*"(?:classify_only|rewrite_only|classify_and_rewrite|passthrough)"',
        f'mode: "{mode_val}"',
        raw_template,
        count=1,
    )
    raw_template = re.sub(
        r'rewriter_model:\s*".*?"',
        f'rewriter_model: "{rewriter_key_str}"',
        raw_template,
        count=1,
    )

    # 在 models: 节点插入新收集的模型
    model_blocks = [_format_model_yaml_block(m) for m in all_models]
    merged_models_text = "\n" + "\n".join(model_blocks) + "\n"
    raw_template = re.sub(
        r"(models:\s*\n)",
        r"\g<1>" + merged_models_text,
        raw_template,
        count=1,
    )

    # 写入 config.yaml
    target_config.write_text(raw_template, encoding="utf-8")

    # 写入 .env 文件
    env_lines: list[str] = []
    if target_env.exists():
        env_lines = target_env.read_text(encoding="utf-8").splitlines()

    for m in all_models:
        if m.env_var_name and m.env_var_val:
            # 检查是否已存在
            found = False
            new_lines = []
            for line in env_lines:
                if line.startswith(f"{m.env_var_name}="):
                    new_lines.append(f'{m.env_var_name}="{m.env_var_val}"')
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f'{m.env_var_name}="{m.env_var_val}"')
            env_lines = new_lines

    target_env.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # ── 7. 打印配置结果与启动确认 ──
    console.print("\n" + "─" * 60)
    console.print(f"[bold green]✔ 已成功为您生成配置文件:[/bold green] [bold]{target_config.name}[/bold]")
    console.print(f"[bold green]✔ 已安全同步更新环境变量:[/bold green] [bold]{target_env.name}[/bold]")
    console.print(
        f"  • [bold cyan]路由模型:[/bold cyan]  {router_model.key if router_model else '[dim]未配置 (走启发式规则)[/dim]'}"
    )
    console.print(
        f"  • [bold cyan]改写模型:[/bold cyan]  {rewriter_model.key if rewriter_model else '[dim]未配置 (关闭改写)[/dim]'}"
    )
    console.print(
        f"  • [bold cyan]回答模型:[/bold cyan]  {target_model.key if target_model else '[dim]未配置 (透传给客户端指定模型)[/dim]'}"
    )
    console.print(f"  • [bold green]自适应模式:[/bold green] [bold]{mode_val}[/bold]")

    console.print("\n[bold]💡 【模型池配置与进阶扩充指引】：[/bold]")
    console.print("向导已将您刚才配置的基础模型登记在 config.yaml 中。")
    console.print("如需接入多模型协同分流（如本地模型处理简单任务、云端大模型处理复杂任务）：")
    console.print(f"👉 请直接使用文本编辑器打开 [bold]{target_config.name}[/bold]：")
    console.print("   1) 在 `models:` 节点下仿照已有示例添加新模型（如通义千问、Claude 等）")
    console.print("   2) 在 `routing:` 节点下设置 simple / mid / complex 对应的模型名称")
    console.print("   3) 在 `.env` 中填入对应云端模型的 API Key")
    console.print("─" * 60 + "\n")

    start_now = input("配置就绪，是否现在立即启动 Prism Router 服务？(Y/n, 默认 Y): ").strip().lower()
    if start_now in ("", "y", "yes"):
        return True

    console.print("[dim]配置已保存。后续可直接运行 prism-router server 启动服务。[/dim]\n")
    return False
