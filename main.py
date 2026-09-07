"""Prism Router 便捷启动脚本

支持直接执行:
    python main.py                # 默认前台启动服务（等价于 python -m prism_router server -y）
    python main.py server         # 交互式模式选择启动
    python main.py server -y      # 显式非交互启动
    python main.py --port 8000    # 指定端口启动
"""

import sys

from prism_router.cli import main

if __name__ == "__main__":
    # 若直接执行 python main.py 且未传任何子命令，默认以 server -y 启动
    if len(sys.argv) == 1:
        sys.argv.extend(["server", "-y"])
    elif len(sys.argv) > 1 and sys.argv[1] not in ("server", "init", "version", "-h", "--help"):
        # 支持直接传参，如 python main.py --port 8000
        sys.argv.insert(1, "server")
        if "-y" not in sys.argv and "--yes" not in sys.argv:
            sys.argv.append("-y")
    main()
