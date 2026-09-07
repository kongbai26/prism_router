"""Prompt template utilities — shared loader with caching."""

from __future__ import annotations

from pathlib import Path

_template_cache: dict[str, str] = {}


def load_prompt(path: str, purpose: str = "prompt") -> str:
    """加载 prompt 模板（带缓存）。文件不存在则报错。"""
    if not path:
        raise ValueError(f"{purpose} 路径未配置")
    if path not in _template_cache:
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"{purpose} 文件不存在: {path}，请检查 config.yaml 中的配置")
        _template_cache[path] = p.read_text(encoding="utf-8")
    return _template_cache[path]
