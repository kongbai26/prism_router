"""CLI 入口：prism-router server"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from prism_router.logger import console

if TYPE_CHECKING:
    from prism_router.settings import Settings


def _is_loopback_host(host: str) -> bool:
    """判断监听地址是否仅限本机访问。"""
    normalized = host.strip().strip("[]").lower()
    return normalized in {"127.0.0.1", "::1", "localhost"}


def cmd_serve(args: argparse.Namespace) -> None:
    """启动 Prism Router 服务"""
    import shutil
    from pathlib import Path

    import uvicorn

    from prism_router import __version__
    from prism_router.channel import auto_assign_tiers, save_generated_routing
    from prism_router.server import app, init_app
    from prism_router.settings import load_settings

    # ── 检查 config.yaml 是否存在 ──
    cfg_file = Path(args.config) if args.config else Path("config.yaml")
    if not cfg_file.exists():
        example_cfg = cfg_file.parent / "config.example.yaml"
        # 交互式终端且未带 -y，提示运行向导
        if sys.stdin.isatty() and not args.yes:
            console.print("\n[bold yellow]⚠ 未检测到配置文件 config.yaml[/bold yellow]")
            run_w = input("是否立即启动快速配置向导？(Y/n, 默认 Y): ").strip().lower()
            if run_w in ("", "y", "yes"):
                from prism_router.init_wizard import run_init_wizard

                started = run_init_wizard(cfg_file, cfg_file.parent / ".env", example_cfg)
                if not started:
                    sys.exit(0)
                # 向导配置完成后，直接以 -y 模式启动，避免二次弹窗
                args.yes = True
            else:
                if example_cfg.exists():
                    shutil.copy(example_cfg, cfg_file)
                    console.print(f"[dim]已从 {example_cfg.name} 创建默认 {cfg_file.name}[/dim]")
        else:
            # 自动化环境或显式 -y：静默复制
            if example_cfg.exists():
                shutil.copy(example_cfg, cfg_file)

    settings = load_settings(args.config)

    # CLI 参数覆盖配置文件（仅显式传入时覆盖）
    host = args.host if args.host is not None else settings.server.host
    port = args.port if args.port is not None else settings.server.port

    if not _is_loopback_host(host) and not settings.server.api_key:
        console.print("[bold red]ERROR:[/bold red] 对外监听必须配置 server.api_key。")
        console.print("[dim]请在 .env 设置 PRISM_ROUTER_API_KEY，并在 config.yaml 中引用它。[/dim]")
        sys.exit(2)

    init_app(settings)

    console.print(
        f"\n[bold cyan]Prism Router[/bold cyan] [dim]v{__version__}[/dim] starting on [bold]{host}:{port}[/bold]\n"
    )

    if settings.routing.mode == "auto":
        if not settings.routing.model_pool:
            console.print("[bold red]ERROR:[/bold red] mode=auto but model_pool is empty")
            sys.exit(1)

        # 自动分配
        settings.routing.tiers = auto_assign_tiers(settings)

        # 持久化
        import os

        config_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else "."
        gen_path = os.path.join(config_dir, "routing.generated.yaml")
        save_generated_routing(settings.routing.tiers, gen_path)

        # 打印结果
        console.print("  [bold green]mode:[/bold green]        [bold]auto[/bold]")
        console.print("  [bold cyan]Auto-assigned tiers:[/bold cyan]")
        for tier, entries in settings.routing.tiers.items():
            models = [e.model for e in entries]
            console.print(f"    [bold cyan]{tier}[/bold cyan]:  {models}")
    else:
        # manual 模式
        _load_manual_tiers(settings)
        console.print("  [bold green]mode:[/bold green]        [bold]manual[/bold]")
        console.print(f"  [bold cyan]simple:[/bold cyan]      {settings.routing.simple or '[dim]未配置[/dim]'}")
        console.print(f"  [bold cyan]mid:[/bold cyan]         {settings.routing.mid or '[dim]未配置[/dim]'}")
        console.print(f"  [bold cyan]complex:[/bold cyan]     {settings.routing.complex or '[dim]未配置[/dim]'}")

    cls_model = settings.routing.classifier_model or "[dim]未配置[/dim]"
    console.print(f"  [bold cyan]classifier:[/bold cyan]  {settings.routing.classifier} → {cls_model}")
    if settings.server.api_key:
        console.print("  [bold green]auth:[/bold green]        [bold green]enabled[/bold green]")
    if settings.cache.enabled:
        console.print(
            f"  [bold green]cache:[/bold green]       [bold green]enabled[/bold green] (TTL={settings.cache.ttl_seconds}s)"
        )
    if settings.rate_limit.enabled:
        console.print(
            f"  [bold green]rate_limit:[/bold green]  [bold green]enabled[/bold green] (local_max={settings.rate_limit.local_max_concurrent})"
        )
    console.print()

    # ── 交互式选择工作模式 ──
    if not args.yes:
        _prompt_rewriting_mode(settings)

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=settings.server.workers,
            log_level=settings.logging.level.lower(),
            timeout_graceful_shutdown=5,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Prism Router 已安全停止。[/dim]")


def _prompt_rewriting_mode(settings: Settings) -> None:
    """启动时交互式选择改写工作模式（跳过条件：非交互终端）"""
    # 非交互终端（如管道、后台运行）→ 跳过
    if not sys.stdin.isatty():
        # 非交互终端下仍需打印改写状态
        if settings.rewriting.mode == "passthrough":
            pool = settings.routing.passthrough_pool
            if pool:
                console.print(
                    f"  [bold green]rewriting:[/bold green]   [bold]passthrough[/bold]（透传池: {len(pool)} 个模型）"
                )
            else:
                console.print(
                    "  [bold green]rewriting:[/bold green]   [bold]passthrough[/bold]（透传池: 所有已配置模型）"
                )
            console.print()
        elif settings.rewriting.mode in ("rewrite_only", "classify_and_rewrite"):
            _print_rewriting_status(settings)
        return

    cli_timeout = settings.rewriting.cli_timeout

    # 检查目标模型（用于路由）与改写模型可用性
    can_route = bool(
        settings.routing.simple
        or settings.routing.mid
        or settings.routing.complex
        or settings.routing.tiers
        or (settings.routing.mode == "auto" and settings.routing.model_pool)
    )
    rewriter_key = settings.rewriting.rewriter_model
    can_rewrite = bool(rewriter_key and settings.get_model_config(rewriter_key))

    route_tag = "" if can_route else " [bold red][不可用: 未配置目标模型][/bold red]"
    rewrite_tag = "" if can_rewrite else " [bold red][不可用: 未配置改写模型][/bold red]"
    cr_tag = ""
    if not can_route and not can_rewrite:
        cr_tag = " [bold red][不可用: 未配置改写与目标模型][/bold red]"
    elif not can_route:
        cr_tag = " [bold red][不可用: 未配置目标模型][/bold red]"
    elif not can_rewrite:
        cr_tag = " [bold red][不可用: 未配置改写模型][/bold red]"

    # 当前已配置的模式 → 对应菜单编号，作为超时默认值
    _MODE_TO_CHOICE = {
        "classify_only": "1",
        "rewrite_only": "2",
        "classify_and_rewrite": "3",
        "passthrough": "4",
    }
    if 1 <= settings.rewriting.cli_default_choice <= 5:
        default_choice = str(settings.rewriting.cli_default_choice)
    else:
        default_choice = _MODE_TO_CHOICE.get(settings.rewriting.mode, "5")

    console.print("[dim]─[/dim]" * 50)
    if cli_timeout > 0:
        console.print(f"[bold]工作模式选择[/bold]（[yellow]{cli_timeout}[/yellow] 秒后使用当前配置）")
    else:
        console.print("[bold]工作模式选择[/bold]")
    console.print("[dim]─[/dim]" * 50)
    console.print(f"  [bold cyan]1)[/bold cyan] 仅分类路由（classify_only）{route_tag}")
    console.print("     [dim]请求 → 分类 → 选择模型 → 转发[/dim]")
    console.print(f"  [bold cyan]2)[/bold cyan] 仅改写（rewrite_only）{rewrite_tag}")
    console.print("     [dim]请求 → 改写器改写 prompt → 转发到目标模型[/dim]")
    console.print(f"  [bold cyan]3)[/bold cyan] 先分类后改写（classify_and_rewrite）{cr_tag}")
    console.print("     [dim]未指定模型: 请求 → 分类 → 选择模型 → 改写 prompt → 转发[/dim]")
    console.print("     [dim]已指定模型: 请求 → 改写 prompt → 转发（跳过分类）[/dim]")
    console.print("  [bold cyan]4)[/bold cyan] 纯透传（passthrough）")
    console.print("     [dim]请求 → 直接转发到指定模型（不做分类、改写）[/dim]")
    console.print("     [dim]模型不在透传池中时自动降级为路由模式[/dim]")
    console.print("  [bold cyan]5)[/bold cyan] 使用 config.yaml 默认配置")
    console.print()

    # 显示当前配置的改写器模型
    if rewriter_key:
        cfg = settings.get_model_config(rewriter_key)
        if cfg:
            console.print(f"  当前改写器: [bold]{rewriter_key}[/bold] [bold green]✓[/bold green]")
        else:
            console.print(f"  当前改写器: [bold]{rewriter_key}[/bold] [bold red]✗ 未找到[/bold red]")
    else:
        console.print("  当前改写器: [dim]未配置（可在 config.yaml 中设置 rewriting.rewriter_model）[/dim]")
    console.print()

    import threading

    result = [None]

    def _read_input():
        try:
            result[0] = input(f"选择工作模式 (1-5, 默认 {default_choice}): ").strip()
        except (EOFError, KeyboardInterrupt):
            pass

    t = threading.Thread(target=_read_input, daemon=True)
    t.start()
    t.join(timeout=cli_timeout if cli_timeout > 0 else None)

    choice = result[0]

    if choice == "" or choice is None:
        choice = default_choice

    # 模式前置能力检测与强制切换
    if choice == "1" and not can_route:
        console.print("[bold yellow]⚠ 提示：未配置路由目标模型，无法使用分类路由！[/bold yellow]")
        if can_rewrite:
            console.print("[bold cyan]➜ 已自动强行切换为【2) 仅改写模式】[/bold cyan]\n")
            choice = "2"
        else:
            console.print("[bold cyan]➜ 已自动强行切换为【4) 纯透传模式】[/bold cyan]\n")
            choice = "4"
    elif choice == "2" and not can_rewrite:
        console.print("[bold yellow]⚠ 提示：未配置改写模型，无法使用改写模式！[/bold yellow]")
        if can_route:
            console.print("[bold cyan]➜ 已自动强行切换为【1) 仅分类路由模式】[/bold cyan]\n")
            choice = "1"
        else:
            console.print("[bold cyan]➜ 已自动强行切换为【4) 纯透传模式】[/bold cyan]\n")
            choice = "4"
    elif choice == "3":
        if not can_route and not can_rewrite:
            console.print("[bold yellow]⚠ 提示：改写模型与路由目标模型均未配置！[/bold yellow]")
            console.print("[bold cyan]➜ 已自动强行切换为【4) 纯透传模式】[/bold cyan]\n")
            choice = "4"
        elif not can_rewrite:
            console.print("[bold yellow]⚠ 提示：未配置改写模型，无法使用改写！[/bold yellow]")
            console.print("[bold cyan]➜ 已自动强行切换为【1) 仅分类路由模式】[/bold cyan]\n")
            choice = "1"
        elif not can_route:
            console.print("[bold yellow]⚠ 提示：未配置路由目标模型，无法使用路由！[/bold yellow]")
            console.print("[bold cyan]➜ 已自动强行切换为【2) 仅改写模式】[/bold cyan]\n")
            choice = "2"
    elif choice == "5":
        # 验证默认配置是否可行
        cur_mode = settings.rewriting.mode
        if cur_mode in ("classify_and_rewrite", "rewrite_only") and not can_rewrite:
            console.print("[bold yellow]⚠ 提示：config.yaml 中启用了改写，但未检测到有效改写模型！[/bold yellow]")
            if can_route:
                console.print("[bold cyan]➜ 已自动强行切换为【1) 仅分类路由模式】[/bold cyan]\n")
                choice = "1"
            else:
                console.print("[bold cyan]➜ 已自动强行切换为【4) 纯透传模式】[/bold cyan]\n")
                choice = "4"
        elif cur_mode in ("classify_only", "classify_and_rewrite") and not can_route:
            console.print("[bold yellow]⚠ 提示：config.yaml 中启用了路由，但未配置路由目标模型！[/bold yellow]")
            if can_rewrite:
                console.print("[bold cyan]➜ 已自动强行切换为【2) 仅改写模式】[/bold cyan]\n")
                choice = "2"
            else:
                console.print("[bold cyan]➜ 已自动强行切换为【4) 纯透传模式】[/bold cyan]\n")
                choice = "4"
        else:
            console.print("[dim]使用 config.yaml 默认配置[/dim]\n")
            return

    if choice == "1":
        settings.rewriting.enabled = False
        settings.rewriting.mode = "classify_only"
        console.print("[bold green]已选择:[/bold green] 仅分类路由\n")
        return

    if choice == "2":
        settings.rewriting.enabled = True
        settings.rewriting.mode = "rewrite_only"
        _print_rewriting_status(settings)
        return

    if choice == "3":
        settings.rewriting.enabled = True
        settings.rewriting.mode = "classify_and_rewrite"
        _print_rewriting_status(settings)
        return

    if choice == "4":
        settings.rewriting.enabled = False
        settings.rewriting.mode = "passthrough"
        pool = settings.routing.passthrough_pool
        if pool:
            console.print(f"[bold green]已选择:[/bold green] 纯透传（透传池: {len(pool)} 个模型）\n")
        else:
            console.print("[bold green]已选择:[/bold green] 纯透传（透传池: 所有已配置模型）\n")
        return

    console.print("[yellow]无效选择，使用 config.yaml 默认配置[/yellow]\n")


def _validate_rewriter_config(settings: Settings) -> bool:
    """验证改写器配置是否有效，无效时给用户降级选项"""
    import threading

    rewriter_key = settings.rewriting.rewriter_model

    # 未配置改写器模型
    if not rewriter_key:
        console.print("\n  [bold yellow]⚠ 未配置改写器模型[/bold yellow]")
        console.print("  [dim]请在 config.yaml 中设置 rewriting.rewriter_model[/dim]")
        console.print()
        console.print("  [bold cyan]1)[/bold cyan] 仅走路由（classify_only）— 放弃改写")
        console.print("  [bold cyan]2)[/bold cyan] 先走路由，后走改写（classify_and_rewrite）[延后]")
        console.print()
        console.print("  [dim]5 秒后自动选择 1 ...[/dim]")

        result = [None]

        def _read():
            try:
                result[0] = input("  选择 (1/2): ").strip()
            except (EOFError, KeyboardInterrupt):
                pass

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=5)

        if result[0] == "2":
            console.print("  [bold green]已选择:[/bold green] 先分类后改写\n")
            settings.rewriting.enabled = True
            settings.rewriting.mode = "classify_and_rewrite"
        else:
            console.print("  [bold green]已选择:[/bold green] 仅分类路由\n")
            settings.rewriting.enabled = False
            settings.rewriting.mode = "classify_only"
        return False

    # 改写器模型不存在
    cfg = settings.get_model_config(rewriter_key)
    if cfg is None:
        console.print(f"\n  [bold yellow]⚠ 改写器模型 '{rewriter_key}' 不存在于 config.yaml 的 models 中[/bold yellow]")
        console.print()
        console.print("  [bold cyan]1)[/bold cyan] 仅走路由（classify_only）— 放弃改写")
        console.print("  [bold cyan]2)[/bold cyan] 先走路由，后走改写（classify_and_rewrite）")
        console.print()
        console.print("  [dim]5 秒后自动选择 1 ...[/dim]")

        result = [None]

        def _read():
            try:
                result[0] = input("  选择 (1/2): ").strip()
            except (EOFError, KeyboardInterrupt):
                pass

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=5)

        if result[0] == "2":
            console.print("  [bold green]已选择:[/bold green] 先分类后改写\n")
            settings.rewriting.enabled = True
            settings.rewriting.mode = "classify_and_rewrite"
        else:
            console.print("  [bold green]已选择:[/bold green] 仅分类路由\n")
            settings.rewriting.enabled = False
            settings.rewriting.mode = "classify_only"
        return False

    return True


def _print_rewriting_status(settings: Settings) -> None:
    """打印已配置的改写状态"""
    cfg = settings.rewriting
    if not cfg.enabled:
        return

    rewriter_key = cfg.rewriter_model
    rewriter_cfg = settings.get_model_config(rewriter_key) if rewriter_key else None

    console.print(f"  [bold green]rewriting:[/bold green]   [bold green]enabled[/bold green] ({cfg.mode})")
    if rewriter_key and rewriter_cfg:
        source = rewriter_cfg.source
        if source == "auto":
            source = "local" if settings.is_local(rewriter_key) else "cloud"
        desc = rewriter_cfg.description[:40]
        console.print(
            f"  [bold cyan]rewriter:[/bold cyan]    [bold]{rewriter_key}[/bold] [{rewriter_cfg.tier}] ([dim]{source}[/dim]) — [dim]{desc}[/dim]"
        )
    elif rewriter_key:
        console.print(
            f"  [bold cyan]rewriter:[/bold cyan]    [bold]{rewriter_key}[/bold] [bold red]✗ 未找到[/bold red]"
        )
    else:
        console.print("  [bold cyan]rewriter:[/bold cyan]    [dim]未配置[/dim]")

    if cfg.mode == "rewrite_only":
        console.print("  [bold cyan]target:[/bold cyan]      客户端请求中指定的 model")
    elif cfg.mode == "classify_and_rewrite":
        console.print("  [bold cyan]target:[/bold cyan]      已指定模型 → 直接改写后转发（模型名见请求日志）")
        console.print("               [dim]未指定模型 → 分类器自动选出模型[/dim]")
    console.print()


def _load_manual_tiers(settings: Settings) -> None:
    """manual 模式：将 config.yaml 的 simple/mid/complex 转为 routing.tiers"""
    from prism_router.settings import ChannelEntry

    routing = settings.routing
    for tier in ("simple", "mid", "complex"):
        key = getattr(routing, tier, "")
        if key:
            routing.tiers[tier] = [ChannelEntry(model=key, priority=1, weight=100)]


def cmd_init(args: argparse.Namespace) -> None:
    """交互式初始化向导"""
    from pathlib import Path

    from prism_router.init_wizard import run_init_wizard

    config_path = Path(args.config) if getattr(args, "config", None) else None
    started = run_init_wizard(config_path=config_path)
    if started:
        cmd_serve(argparse.Namespace(config=args.config, host=None, port=None, yes=True))


def cmd_version(_args: argparse.Namespace) -> None:
    from prism_router import __version__

    console.print(f"[bold cyan]prism-router[/bold cyan] [dim]{__version__}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prism-router",
        description="Prism Router — 智能 LLM 路由器",
    )
    subparsers = parser.add_subparsers(dest="command")

    # server
    serve_parser = subparsers.add_parser("server", help="启动路由服务")
    serve_parser.add_argument("-c", "--config", default=None, help="配置文件路径 (默认: config.yaml)")
    serve_parser.add_argument("--host", default=None, help="监听地址 (默认: 配置文件值)")
    serve_parser.add_argument("--port", type=int, default=None, help="监听端口 (默认: 配置文件值)")
    serve_parser.add_argument("-y", "--yes", action="store_true", help="跳过交互式选择，使用 config.yaml 默认配置")
    serve_parser.set_defaults(func=cmd_serve)

    # init
    init_parser = subparsers.add_parser("init", help="交互式初始化向导，配置并生成 config.yaml 与 .env")
    init_parser.add_argument("-c", "--config", default=None, help="目标配置文件路径 (默认: config.yaml)")
    init_parser.set_defaults(func=cmd_init)

    # version
    version_parser = subparsers.add_parser("version", help="显示版本号")
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)
