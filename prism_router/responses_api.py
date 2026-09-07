"""OpenAI Responses API 格式转换与端点处理。

将 OpenAI Responses API 请求转换为 Chat Completions 格式，
转发给 handlers.process_chat_request 管线处理，再将响应转回 Responses API 格式。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from prism_router.server import _error, get_settings
from prism_router.tool_call_cache import get_tool_call_cache

logger = logging.getLogger("prism_router.responses_api")


# ── reasoning_content 缓存（DeepSeek thinking 模式） ──
# Claude Code 不认识 reasoning_content，不会在后续请求中传回。
# 服务端缓存：content_hash → (reasoning_content, inserted_at)，自动注入缺失的字段。
_REASONING_CACHE: dict[str, tuple[str, float]] = {}
_REASONING_CACHE_MAX = 500
_REASONING_CACHE_TTL = 1800  # 30 分钟


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]


def _cache_reasoning(content: str, reasoning: str) -> None:
    """缓存 assistant 消息的 reasoning_content"""
    if not content or not reasoning:
        return
    now = time.time()
    key = _content_hash(content)
    # 仅在超限时做全量淘汰（惰性策略，读取时 TTL 检查已过滤过期条目）
    if len(_REASONING_CACHE) >= _REASONING_CACHE_MAX:
        expired = [k for k, (_, ts) in _REASONING_CACHE.items() if now - ts > _REASONING_CACHE_TTL]
        for k in expired:
            del _REASONING_CACHE[k]
        while len(_REASONING_CACHE) >= _REASONING_CACHE_MAX:
            oldest = next(iter(_REASONING_CACHE))
            del _REASONING_CACHE[oldest]
    _REASONING_CACHE[key] = (reasoning, now)


def _get_cached_reasoning(content: str) -> str | None:
    """获取缓存的 reasoning_content（带 TTL 检查）"""
    entry = _REASONING_CACHE.get(_content_hash(content))
    if not entry:
        return None
    reasoning, ts = entry
    if time.time() - ts > _REASONING_CACHE_TTL:
        del _REASONING_CACHE[_content_hash(content)]
        return None
    return reasoning


# ── OpenAI Responses API 端点 ──


def _clean_schema(obj: Any) -> Any:
    """递归清除 JSON Schema 中上游不支持的字段（additionalProperties, strict）"""
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k in ("additionalProperties", "strict"):
            continue
        if isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


_APPLY_PATCH_DESCRIPTION = (
    "Edit files using the apply_patch tool. "
    "ALWAYS use this tool to write file content. "
    "Call this function with a JSON object containing an 'input' string with a V4A patch. "
    "The patch MUST start with '*** Begin Patch' and end with '*** End Patch'. "
    "Each file operation: '*** Add File: <path>', '*** Update File: <path>', or '*** Delete File: <path>'. "
    "Lines: '-line' (removed), '+line' (added), ' line' (context, leading space). "
    "Use relative paths only."
)


def _convert_tools(tools: list) -> list:
    """将工具定义从 Responses API 格式转换为 Chat Completions 格式。

    支持的 Responses API 工具类型:
    - function: 标准函数工具（可能有或没有 function 嵌套）
    - custom: 自由格式工具（如 apply_patch），转为 function + input 参数
    - tool_search: MCP 工具搜索，转为 function（name="tool_search"）
    - namespace: MCP 命名空间，递归展平内部 tools
    - 其他（web_search, local_shell 等）: 跳过

    参考 codex-app-transfer 的 convert_responses_tool_to_chat_tool。
    """
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type", "")
        name = tool.get("name", "")
        description = tool.get("description", "")

        # ── 已经是 Chat Completions 格式（有 function 嵌套） ──
        if tool_type == "function" and "function" in tool:
            func = dict(tool["function"])
            if "parameters" in func:
                func["parameters"] = _clean_schema(func["parameters"])
            result.append({"type": "function", "function": func})
            continue

        # ── Responses API function 格式（flat: type, name, parameters） ──
        if tool_type == "function":
            if not name:
                logger.debug("Skipping function tool with empty name")
                continue
            func = {"name": name, "description": description}
            params = tool.get("parameters")
            if params:
                func["parameters"] = _clean_schema(params)
                if "type" not in func["parameters"]:
                    func["parameters"]["type"] = "object"
            result.append({"type": "function", "function": func})
            continue

        # ── custom 类型（apply_patch 等自由格式工具） ──
        # 参考 codex-app-transfer: custom → function + input 参数
        if tool_type == "custom":
            if not name:
                logger.debug("Skipping custom tool with empty name")
                continue
            if name == "apply_patch" and "parameters" not in tool:
                func = {
                    "name": name,
                    "description": _APPLY_PATCH_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "V4A patch content"},
                        },
                        "required": ["input"],
                    },
                }
            else:
                func = {
                    "name": name,
                    "description": description or "Free-form input passed verbatim to the tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "Free-form input passed verbatim to the tool."},
                        },
                        "required": ["input"],
                    },
                }
            result.append({"type": "function", "function": func})
            continue

        # ── tool_search 类型（Codex 0.130+ MCP 工具搜索） ──
        if tool_type == "tool_search":
            func = {
                "name": "tool_search",
                "description": description,
                "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
            }
            result.append({"type": "function", "function": func})
            continue

        # ── namespace 类型（MCP 命名空间，递归展平） ──
        if tool_type == "namespace" and isinstance(tool.get("tools"), list):
            result.extend(_convert_tools(tool["tools"]))
            continue

        # ── 有 name 但无 type（兼容旧格式） ──
        if name and not tool_type:
            func = {"name": name, "description": description}
            params = tool.get("parameters")
            if params:
                func["parameters"] = _clean_schema(params)
                if "type" not in func["parameters"]:
                    func["parameters"]["type"] = "object"
            result.append({"type": "function", "function": func})
            continue

        # ── 其他类型（web_search, local_shell, file_search 等）跳过 ──
        logger.debug("Skipping unsupported tool type: %s (name=%s)", tool_type, name)

    return result


def _convert_tool_choice(tc: Any) -> Any:
    """将 tool_choice 从 Responses API 格式转换为 Chat Completions 格式。

    参考 codex-app-transfer 的 normalize_tool_choice。
    映射:
    - 字符串 ("auto", "none", "required") → 直接透传
    - {type: "auto"} → "auto"
    - {type: "none"} → "none"
    - {type: "required"} / {type: "tool"} / {type: "any"} → "required"
    - {type: "function", name: "..."} → {type: "function", function: {name: "..."}}
    """
    if tc is None:
        return "auto"
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict):
        # 已经有 function.name 的格式，直接透传
        func = tc.get("function")
        if isinstance(func, dict) and func.get("name"):
            return tc
        tc_type = tc.get("type", "")
        if tc_type in ("auto",):
            return "auto"
        if tc_type in ("none",):
            return "none"
        if tc_type in ("required", "tool", "any"):
            return "required"
        if tc_type == "function":
            # Responses API 格式: {type: "function", name: "my_func"}
            # 转换为 Chat Completions 格式: {type: "function", function: {name: "my_func"}}
            name = tc.get("name", "")
            if name:
                return {"type": "function", "function": {"name": name}}
            return "required"
    return "auto"


def _flush_pending_as_placeholders(
    repaired: list[dict],
    pending_call_ids: list[str],
    pending_names: dict[str, str],
) -> None:
    """flush 未关闭的 tool_calls：补占位 tool 消息"""
    for call_id in pending_call_ids:
        name = pending_names.get(call_id, "")
        repaired.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"[System: Tool execution skipped. No result for tool '{name}'.]",
            }
        )
    pending_call_ids.clear()
    pending_names.clear()


def _repair_tool_call_ids(messages: list[dict], cache) -> list[dict]:
    """修复孤儿 tool 消息，确保每个 tool 消息有对应的 assistant tool_call。
    参考 codex-app-transfer 的 repair_tool_call_ids 实现。"""
    pending_call_ids: list[str] = []
    pending_names: dict[str, str] = {}  # call_id → name
    last_assistant_idx: int | None = None
    repaired: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            _flush_pending_as_placeholders(repaired, pending_call_ids, pending_names)
            calls = msg.get("tool_calls") or []
            pending_call_ids = [c["id"] for c in calls if c.get("id", "").strip()]
            pending_names = {c["id"]: c.get("function", {}).get("name", "") for c in calls if c.get("id", "").strip()}
            last_assistant_idx = len(repaired)
            repaired.append(msg)

        elif role == "tool":
            existing = (msg.get("tool_call_id") or "").strip()
            if not existing and pending_call_ids:
                # Path A1: 无 tool_call_id，从 pending 补
                call_id = pending_call_ids.pop(0)
                pending_names.pop(call_id, None)
                msg = dict(msg)
                msg["tool_call_id"] = call_id
                repaired.append(msg)
            elif existing and existing in pending_call_ids:
                # Path B1: tool_call_id 在 pending 中，ack 通过
                pending_call_ids.remove(existing)
                pending_names.pop(existing, None)
                repaired.append(msg)
            elif existing:
                # Path B2/B3: tool_call_id 不在 pending 中，查缓存重建
                entry = cache.get(existing)
                if entry:
                    # 缓存命中：注入到前 assistant
                    name, args = entry
                    placeholder_tc = {
                        "id": existing,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                    if last_assistant_idx is not None:
                        repaired[last_assistant_idx].setdefault("tool_calls", []).append(placeholder_tc)
                    else:
                        last_assistant_idx = len(repaired)
                        repaired.append({"role": "assistant", "content": "", "tool_calls": [placeholder_tc]})
                    repaired.append(msg)
                else:
                    # 缓存未命中：转为 user 消息（避免空 name 导致上游 400）
                    content = msg.get("content", "")
                    if content:
                        repaired.append({"role": "user", "content": f"[Tool result] {content}"})
            else:
                # 无 call_id 且无 pending，真正的孤儿
                # 若有多轮积压的 pending 未 flush，先 flush 确保 assistant→tool 配对完整
                _flush_pending_as_placeholders(repaired, pending_call_ids, pending_names)
                # 将孤儿转为 user 消息，保留信息不丢失
                content = msg.get("content", "")
                if content:
                    repaired.append({"role": "user", "content": f"[Tool result] {content}"})

        elif role in ("user", "system", "developer"):
            _flush_pending_as_placeholders(repaired, pending_call_ids, pending_names)
            last_assistant_idx = None
            repaired.append(msg)

        else:
            repaired.append(msg)

    _flush_pending_as_placeholders(repaired, pending_call_ids, pending_names)
    return repaired


def _extract_apply_patch_input(arguments: str) -> str:
    """从 apply_patch 的 arguments JSON 中提取 V4A patch 文本。
    处理多种格式：标准 {input:...}、其他 key、裸文本。"""
    if not arguments:
        return ""
    try:
        data = json.loads(arguments)
        if isinstance(data, dict):
            # 尝试各种可能的 key
            for key in ("input", "patch", "diff", "apply_patch", "input_text", "content"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            # 没找到 string 字段，返回原始 JSON
            return arguments
    except (json.JSONDecodeError, ValueError):
        pass
    # 非 JSON，可能是裸 V4A 文本
    return arguments


def _ensure_thinking_tool_call_reasoning(messages: list[dict]) -> None:
    """确保带 tool_calls 的 assistant 消息有 reasoning_content（DeepSeek 要求）。
    原地修改 messages 列表。"""
    has_tool_loop = False
    for msg in messages:
        if msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls")):
            has_tool_loop = True
            break
    if not has_tool_loop:
        return
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls") and not msg.get("reasoning_content"):
            msg["reasoning_content"] = " "


def _responses_to_chat(body: dict) -> dict:
    """将 OpenAI Responses API 请求体转换为 Chat Completions 格式（参考 codex_deepseek_proxy）"""
    ROLE_MAP = {"developer": "system"}
    messages: list[dict] = []
    _discovered_tools: list[dict] = []  # tool_search_output 发现的工具

    # instructions → system message
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # input → messages
    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        # function_call 批量合并（参考 codex_deepseek_proxy）
        pending_tool_calls: list[dict] = []
        pending_reasoning = ""

        def _flush_tool_calls() -> None:
            nonlocal pending_tool_calls, pending_reasoning
            if pending_tool_calls:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": pending_tool_calls,
                }
                if pending_reasoning:
                    msg["reasoning_content"] = pending_reasoning
                elif _get_cached_reasoning(f"fc:{pending_tool_calls[0].get('id', '')}"):
                    msg["reasoning_content"] = _get_cached_reasoning(f"fc:{pending_tool_calls[0].get('id', '')}")
                else:
                    msg["reasoning_content"] = " "
                messages.append(msg)
                pending_tool_calls = []
                pending_reasoning = ""

        for item in inp:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")

            if item_type == "message" or (item_type is None and "role" in item):
                _flush_tool_calls()
                role = item.get("role", "user")
                role = ROLE_MAP.get(role, role)
                content = item.get("content", "")
                refusal_text = ""
                if isinstance(content, list):
                    texts = []
                    refusal_text = ""
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        c_type = c.get("type")
                        if c_type in ("text", "input_text", "output_text"):
                            t = c.get("text", "")
                            if t.strip():
                                texts.append(t)
                        elif c_type == "refusal":
                            refusal_text = c.get("refusal", "")
                    content = "\n".join(texts)
                    # refusal content block → 附加 refusal 字段
                    if refusal_text and not content:
                        content = None
                if content is None and refusal_text:
                    # refusal 消息（模型拒绝回答）
                    msg = {"role": role, "content": None, "refusal": refusal_text}
                    messages.append(msg)
                elif isinstance(content, str) and content.strip():
                    msg = {"role": role, "content": content.strip()}
                    if role == "assistant":
                        if item.get("reasoning_content"):
                            msg["reasoning_content"] = item["reasoning_content"]
                        else:
                            cached = _get_cached_reasoning(content.strip())
                            if cached:
                                msg["reasoning_content"] = cached
                    messages.append(msg)

            elif item_type == "function_call":
                # 累积连续的 function_call，稍后合并为一个 assistant 消息
                pending_tool_calls.append(
                    {
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }
                )
                if item.get("reasoning_content") and not pending_reasoning:
                    pending_reasoning = item["reasoning_content"]

            elif item_type == "function_call_output":
                _flush_tool_calls()
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                content = output
                if get_settings().logging.artifact_storage_enabled:
                    from prism_router.artifact_store import get_artifact_store

                    artifact_store = get_artifact_store()
                    if artifact_store.should_store(output):
                        artifact_id = artifact_store.store(call_id=item.get("call_id", ""), output=output)
                        content = artifact_store.make_summary(output, artifact_id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": content,
                    }
                )

            elif item_type == "tool_search_call":
                # Codex 0.130+ MCP 工具搜索 → 转为 function_call（name="tool_search"）
                _flush_tool_calls()
                pending_tool_calls.append(
                    {
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": "tool_search",
                            "arguments": json.dumps({"query": item.get("query", "")}, ensure_ascii=False),
                        },
                    }
                )

            elif item_type == "tool_search_output":
                # tool_search 结果 → tool 消息
                _flush_tool_calls()
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                call_id = item.get("call_id", "")
                if not call_id:
                    logger.warning("tool_search_output missing call_id, result may be lost")
                content = output
                if get_settings().logging.artifact_storage_enabled:
                    from prism_router.artifact_store import get_artifact_store

                    artifact_store = get_artifact_store()
                    if artifact_store.should_store(output):
                        artifact_id = artifact_store.store(call_id=call_id, output=output)
                        content = artifact_store.make_summary(output, artifact_id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    }
                )
                # 收集发现的工具（稍后注入 tools 列表）
                discovered = item.get("tools")
                if isinstance(discovered, list):
                    _discovered_tools.extend(discovered)

        _flush_tool_calls()

        # 修复孤儿 tool 消息 + 重排（参考 codex-app-transfer 的 repair_tool_call_ids）
        messages = _repair_tool_call_ids(messages, get_tool_call_cache())

        # 确保带 tool_calls 的 assistant 消息有 reasoning_content（DeepSeek 要求）
        _ensure_thinking_tool_call_reasoning(messages)

    chat_body: dict[str, Any] = {
        "model": body.get("model", "prism-auto"),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    # max_output_tokens → max_tokens
    max_output = body.get("max_output_tokens")
    if max_output is not None:
        chat_body["max_tokens"] = max_output

    # 透传通用参数
    for key in (
        "temperature",
        "top_p",
        "seed",
        "stop",
        "user",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "n",
    ):
        if key in body:
            chat_body[key] = body[key]

    # reasoning → reasoning_effort（Responses API 用 reasoning.effort，Chat Completions 用 reasoning_effort）
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat_body["reasoning_effort"] = reasoning["effort"]
    elif isinstance(reasoning, str):
        chat_body["reasoning_effort"] = reasoning

    # parallel_tool_calls 透传（Chat Completions 原生支持）
    if "parallel_tool_calls" in body:
        chat_body["parallel_tool_calls"] = body["parallel_tool_calls"]

    # text → response_format（Responses API 用 text.format，Chat Completions 用 response_format）
    text_cfg = body.get("text")
    if isinstance(text_cfg, dict):
        fmt = text_cfg.get("format")
        if isinstance(fmt, dict):
            if fmt.get("type") == "json_object":
                chat_body["response_format"] = {"type": "json_object"}
            elif fmt.get("type") == "json_schema":
                chat_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": fmt.get("name", "output"),
                        "schema": fmt.get("schema", {}),
                        "strict": fmt.get("strict", False),
                    },
                }

    # tools 转换
    tools = body.get("tools")
    if tools and isinstance(tools, list):
        converted = _convert_tools(tools)
        if converted:
            chat_body["tools"] = converted

    # 如果 messages 中有 tool_calls 但没有 tools 定义，从 tool_calls 中提取
    if "tools" not in chat_body:
        for msg in messages:
            tcs = msg.get("tool_calls")
            if msg.get("role") == "assistant" and isinstance(tcs, list):
                extracted = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        if isinstance(func, dict):
                            extracted.append(
                                {
                                    "type": "function",
                                    "function": {
                                        "name": func.get("name", ""),
                                        "parameters": {"type": "object", "properties": {}},
                                    },
                                }
                            )
                if extracted:
                    chat_body["tools"] = extracted
                break

    # tool_choice 转换
    if "tool_choice" in body:
        chat_body["tool_choice"] = _convert_tool_choice(body["tool_choice"])

    # 注入 tool_search_output 发现的工具
    if _discovered_tools:
        existing = chat_body.get("tools") or []
        for dt in _discovered_tools:
            if isinstance(dt, dict) and dt.get("name"):
                existing.append(
                    {
                        "type": "function",
                        "function": {
                            "name": dt["name"],
                            "description": dt.get("description", ""),
                            "parameters": dt.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )
        if existing:
            chat_body["tools"] = existing

    return chat_body


def _build_usage(chat_usage: dict) -> dict[str, Any]:
    """将 Chat Completions usage 转换为 Responses API usage 格式（含 details）"""
    input_tokens = chat_usage.get("prompt_tokens", 0)
    output_tokens = chat_usage.get("completion_tokens", 0)
    total_tokens = chat_usage.get("total_tokens", 0)
    # total_tokens 缺失时自动计算（参考 codex-app-transfer converter.rs:3737）
    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    # 参考 codex-app-transfer converter.rs:1719-1733
    # Codex CLI 0.128 要求这些字段必须存在（缺失会导致流断开）
    prompt_details = chat_usage.get("prompt_tokens_details")
    cached = 0
    if isinstance(prompt_details, dict):
        cached = prompt_details.get("cached_tokens", 0) or 0
    usage["input_tokens_details"] = {"cached_tokens": cached}

    completion_details = chat_usage.get("completion_tokens_details")
    reasoning = 0
    if isinstance(completion_details, dict):
        reasoning = completion_details.get("reasoning_tokens", 0) or 0
    usage["output_tokens_details"] = {"reasoning_tokens": reasoning}
    return usage


def _chat_to_responses(chat_resp: dict, resp_id: str, model: str, body: dict | None = None) -> dict:
    """将 Chat Completions 响应转换为 OpenAI Responses API 格式"""
    if body is None:
        body = {}
    output_items = []
    choices = chat_resp.get("choices", [])
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"

    # 根据 finish_reason 决定 status
    is_complete = finish_reason in ("stop", "tool_calls", "end_turn", "function_call")
    item_status = "completed" if is_complete else "incomplete"
    resp_status = "completed" if is_complete else "incomplete"

    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # think_tag 兜底拆分：从 content 提取 <think>...</think> 为 reasoning_content
        if content and not msg.get("reasoning_content"):
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                msg = dict(msg)
                msg["reasoning_content"] = think_match.group(1).strip()
                content = (content[: think_match.start()] + content[think_match.end() :]).strip()

        # 文本输出
        if content:
            item: dict[str, Any] = {
                "type": "message",
                "id": f"msg_{resp_id}",
                "role": "assistant",
                "status": item_status,
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
            if msg.get("reasoning_content"):
                item["reasoning_content"] = msg["reasoning_content"]
                if content:
                    _cache_reasoning(content, msg["reasoning_content"])
            output_items.append(item)

        # 工具调用输出（每个使用唯一 id）
        for tc_idx, tc in enumerate(tool_calls or []):
            func = tc.get("function", {})
            output_items.append(
                {
                    "type": "function_call",
                    "id": f"fc_{resp_id}_{tc_idx}",
                    "call_id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                    "status": item_status,
                }
            )

    # usage 转换（含 cached_tokens / reasoning_tokens）
    usage = _build_usage(chat_resp.get("usage", {}))

    # 参考 codex-app-transfer converter.rs:352-375 build_envelope
    result: dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": resp_status,
        "model": model,
        "output": output_items,
        "usage": usage,
        "tools": body.get("tools", []),
        "tool_choice": body.get("tool_choice", "auto"),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "reasoning": body.get("reasoning", {"effort": None, "summary": None}),
        "text": body.get("text", {"format": {"type": "text"}}),
        "metadata": body.get("metadata"),
        "previous_response_id": body.get("previous_response_id"),
        "instructions": body.get("instructions"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_output_tokens": body.get("max_output_tokens"),
        "truncation": body.get("truncation", "disabled"),
    }
    # incomplete 时附加 incomplete_details
    if not is_complete:
        reason_map = {"length": "max_output_tokens", "content_filter": "content_filter"}
        result["incomplete_details"] = {"reason": reason_map.get(finish_reason, "unknown")}
    return result


def _start_reasoning_events(sse_fn, reasoning_item_id: str, seq: int) -> tuple[list[str], int]:
    """生成 reasoning item 开始的 SSE 事件序列。返回 (事件列表, 更新后的 seq)。"""
    events = []
    events.append(
        sse_fn(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": reasoning_item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "role": "assistant",
                    "summary": [],
                    "content": None,
                    "encrypted_content": None,
                },
            },
        )
    )
    events.append(
        sse_fn(
            "response.reasoning_summary_part.added",
            {
                "type": "response.reasoning_summary_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "summary_text", "text": ""},
            },
        )
    )
    seq += 1
    events.append(
        sse_fn(
            "response.reasoning_summary_text.delta",
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "**Thinking**\n\n",
                "sequence_number": seq,
            },
        )
    )
    return events, seq


def _close_reasoning_events(sse_fn, reasoning_item_id: str, accumulated_reasoning: str) -> tuple[list[str], dict]:
    """生成 reasoning item 关闭的 SSE 事件序列。返回 (事件列表, reasoning_item)。"""
    events = []
    events.append(
        sse_fn(
            "response.reasoning_summary_text.done",
            {
                "type": "response.reasoning_summary_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": accumulated_reasoning,
            },
        )
    )
    events.append(
        sse_fn(
            "response.reasoning_summary_part.done",
            {
                "type": "response.reasoning_summary_part.done",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "summary_text", "text": accumulated_reasoning},
            },
        )
    )
    reasoning_item = {
        "id": reasoning_item_id,
        "type": "reasoning",
        "status": "completed",
        "role": "assistant",
        "summary": [{"type": "summary_text", "text": accumulated_reasoning}],
    }
    events.append(
        sse_fn(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": reasoning_item,
            },
        )
    )
    return events, reasoning_item


async def _wrap_stream_as_responses(body_iterator, resp_id: str, model: str, body: dict | None = None):
    """将 Chat Completions SSE 流转换为 Responses API SSE 事件流（参考 codex-proxy）"""
    if body is None:
        body = {}
    text_item_id = f"item_{resp_id}"
    has_text = False
    text_started = False
    content_part_opened = False
    created_sent = False
    accumulated_text = ""
    accumulated_reasoning = ""
    accumulated_annotations: list[dict] = []
    reasoning_started = False
    reasoning_item_id = ""
    output_items: list[dict] = []
    # tool call 跟踪：index → {id, name, arguments, item_id, started}
    tool_calls_acc: dict[int, dict] = {}
    final_usage: dict = {}
    finish_reason: str | None = None
    # think_tag 兜底拆分状态
    think_tag_mode = False
    think_tag_buffer = ""
    # namespace 映射：function.name → namespace.name（参考 codex-app-transfer converter.rs:336）
    tool_namespace_map: dict[str, str] = {}
    if body and body.get("tools"):
        for tool in body["tools"]:
            if isinstance(tool, dict) and tool.get("type") == "namespace" and isinstance(tool.get("tools"), list):
                ns_name = tool.get("name", "")
                for inner in tool["tools"]:
                    if isinstance(inner, dict) and inner.get("type") == "function" and inner.get("name"):
                        tool_namespace_map[inner["name"]] = ns_name
    seq = 0

    def _sse(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        async for raw_line in body_iterator:
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = raw_line

            for single_line in line.split("\n"):
                single_line = single_line.strip()
                if not single_line:
                    continue

                if single_line.startswith("data: "):
                    data_str = single_line[6:]
                elif single_line.startswith("data:"):
                    data_str = single_line[5:]
                else:
                    continue

                if data_str.strip() == "[DONE]":
                    # 缓存 reasoning_content
                    if accumulated_text and accumulated_reasoning:
                        _cache_reasoning(accumulated_text, accumulated_reasoning)

                    # 关闭 reasoning item（如果 reasoning 还在进行中，没有后续文本）
                    if reasoning_started:
                        reasoning_started = False
                        events, reasoning_item = _close_reasoning_events(_sse, reasoning_item_id, accumulated_reasoning)
                        for ev in events:
                            yield ev
                        output_items.append(reasoning_item)

                    # 文本完成事件
                    text_out_idx = 1 if accumulated_reasoning else 0
                    if has_text:
                        yield _sse(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "text": accumulated_text,
                                "item_id": text_item_id,
                                "output_index": text_out_idx,
                                "content_index": 0,
                            },
                        )
                        # annotation 事件（预留：当前上游不返回 annotations）
                        for ann_idx, ann in enumerate(accumulated_annotations):
                            yield _sse(
                                "response.output_text.annotation.done",
                                {
                                    "type": "response.output_text.annotation.done",
                                    "item_id": text_item_id,
                                    "output_index": text_out_idx,
                                    "content_index": 0,
                                    "annotation_index": ann_idx,
                                    "annotation": ann,
                                },
                            )
                        # content_part.done
                        if content_part_opened:
                            yield _sse(
                                "response.content_part.done",
                                {
                                    "type": "response.content_part.done",
                                    "output_index": text_out_idx,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": accumulated_text,
                                        "annotations": accumulated_annotations,
                                    },
                                },
                            )

                    # 构建 output items（已在顶部初始化，reasoning item 可能已追加）
                    if has_text:
                        text_item: dict[str, Any] = {
                            "id": text_item_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": accumulated_text,
                                    "annotations": accumulated_annotations,
                                }
                            ],
                        }
                        if accumulated_reasoning:
                            text_item["reasoning_content"] = accumulated_reasoning
                        yield _sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": text_out_idx,
                                "item": text_item,
                            },
                        )
                        output_items.append(text_item)

                    # 工具调用完成事件
                    tc_cache = get_tool_call_cache()
                    for idx in sorted(tool_calls_acc.keys()):
                        acc = tool_calls_acc[idx]
                        base_idx = (1 if accumulated_reasoning else 0) + (1 if has_text else 0)
                        out_idx = base_idx + sorted(tool_calls_acc.keys()).index(idx)

                        # 缓存 tool call（供 repair_tool_call_ids 使用）
                        if acc["id"] and acc["name"]:
                            tc_cache.save(acc["id"], acc["name"], acc["arguments"])

                        # apply_patch 特殊处理：转为 custom_tool_call
                        # 参考 codex-app-transfer converter.rs:828-1050
                        is_apply_patch = acc["name"] == "apply_patch"
                        is_tool_search = acc["name"] == "tool_search"
                        is_legacy_mcp = acc["name"] in (
                            "list_mcp_resources",
                            "list_mcp_resource_templates",
                            "read_mcp_resource",
                        )

                        if is_apply_patch:
                            input_text = _extract_apply_patch_input(acc["arguments"])
                            # custom_tool_call_input.done 事件
                            yield _sse(
                                "response.custom_tool_call_input.done",
                                {
                                    "type": "response.custom_tool_call_input.done",
                                    "item_id": acc["item_id"],
                                    "output_index": out_idx,
                                },
                            )
                            func_item: dict[str, Any] = {
                                "id": acc["item_id"],
                                "type": "custom_tool_call",
                                "status": "completed",
                                "call_id": acc["id"],
                                "name": acc["name"],
                                "input": input_text,
                            }
                        elif is_tool_search or is_legacy_mcp:
                            # tool_search arguments 解析为 JSON 对象
                            # 参考 codex-app-transfer converter.rs:989-998
                            try:
                                args_obj = json.loads(acc["arguments"]) if acc["arguments"] else {}
                            except (json.JSONDecodeError, ValueError):
                                args_obj = {"raw": acc["arguments"]}
                            # function_call_arguments.done 事件
                            yield _sse(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "item_id": acc["item_id"],
                                    "output_index": out_idx,
                                    "arguments": acc["arguments"],
                                },
                            )
                            func_item = {
                                "id": acc["item_id"],
                                "type": "tool_search_call",
                                "status": "completed",
                                "call_id": acc["id"],
                                "name": acc["name"],
                                "arguments": args_obj,
                                "execution": "client",
                            }
                        else:
                            # function_call_arguments.done 事件
                            yield _sse(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "item_id": acc["item_id"],
                                    "output_index": out_idx,
                                    "arguments": acc["arguments"],
                                },
                            )
                            func_item = {
                                "id": acc["item_id"],
                                "type": "function_call",
                                "status": "completed",
                                "call_id": acc["id"],
                                "name": acc["name"],
                                "arguments": acc["arguments"],
                            }
                            # namespace 映射
                            ns = tool_namespace_map.get(acc["name"])
                            if ns:
                                func_item["namespace"] = ns
                        if accumulated_reasoning:
                            func_item["reasoning_content"] = accumulated_reasoning
                        yield _sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": out_idx,
                                "item": func_item,
                            },
                        )
                        output_items.append(func_item)

                    # 空响应兜底（无文本、无工具调用、且 reasoning 也没产生）
                    if not has_text and not tool_calls_acc and not accumulated_reasoning:
                        yield _sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "type": "message",
                                    "id": text_item_id,
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                },
                            },
                        )
                        yield _sse(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "output_index": 0,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                            },
                        )
                        yield _sse(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": 0,
                                "content_index": 0,
                                "delta": "",
                            },
                        )
                        yield _sse(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "text": "",
                                "item_id": text_item_id,
                                "output_index": 0,
                                "content_index": 0,
                            },
                        )
                        yield _sse(
                            "response.content_part.done",
                            {
                                "type": "response.content_part.done",
                                "output_index": 0,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                            },
                        )
                        yield _sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": 0,
                                "item": {
                                    "type": "message",
                                    "id": text_item_id,
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "", "annotations": []}],
                                },
                            },
                        )

                    # finish_reason → status / incomplete_details
                    # 参考 codex-app-transfer converter.rs:1598-1607
                    if finish_reason in ("stop", "tool_calls", "function_call"):
                        resp_status = "completed"
                        incomplete_details = None
                    elif finish_reason == "length":
                        resp_status = "incomplete"
                        incomplete_details = {"reason": "max_output_tokens"}
                    elif finish_reason == "content_filter":
                        resp_status = "incomplete"
                        incomplete_details = {"reason": "content_filter"}
                    elif finish_reason:
                        resp_status = "incomplete"
                        incomplete_details = {"reason": finish_reason}
                    elif not created_sent:
                        # 流被中断（没有 finish_reason 也没有 [DONE]）
                        resp_status = "incomplete"
                        incomplete_details = {"reason": "interrupted"}
                    else:
                        resp_status = "completed"
                        incomplete_details = None

                    # response.completed（回灌原始 request 字段）
                    # 参考 codex-app-transfer converter.rs:365-375
                    completed_resp: dict[str, Any] = {
                        "id": resp_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "status": resp_status,
                        "model": model,
                        "output": output_items,
                        "usage": _build_usage(final_usage),
                        "incomplete_details": incomplete_details,
                        "max_output_tokens": body.get("max_output_tokens"),
                        "truncation": body.get("truncation", "disabled"),
                        "tools": body.get("tools"),
                        "tool_choice": body.get("tool_choice"),
                        "reasoning": body.get("reasoning"),
                        "previous_response_id": body.get("previous_response_id"),
                        "instructions": body.get("instructions"),
                        "text": body.get("text"),
                        "metadata": body.get("metadata"),
                        "temperature": body.get("temperature"),
                        "top_p": body.get("top_p"),
                        "parallel_tool_calls": body.get("parallel_tool_calls", True),
                    }
                    yield _sse(
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": completed_resp,
                        },
                    )
                    return

                try:
                    chunk = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue

                if chunk.get("usage"):
                    final_usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                # finish_reason 跟踪（用于 incomplete_details）
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                delta = choices[0].get("delta", {})

                # 首个有效 chunk：response.created + response.in_progress
                # 参考 codex-app-transfer converter.rs:352-375 build_envelope
                if not created_sent:
                    c = delta.get("content")
                    has_real_content = c is not None and c != ""
                    has_tc = bool(delta.get("tool_calls"))
                    has_rc = bool(delta.get("reasoning_content"))
                    if has_real_content or has_tc or has_rc:
                        created_sent = True
                        # 构建完整 envelope（18 个字段）
                        envelope = {
                            "id": resp_id,
                            "object": "response",
                            "created_at": int(time.time()),
                            "status": "in_progress",
                            "model": model,
                            "tools": body.get("tools", []),
                            "tool_choice": body.get("tool_choice", "auto"),
                            "parallel_tool_calls": body.get("parallel_tool_calls", True),
                            "reasoning": body.get("reasoning", {"effort": None, "summary": None}),
                            "text": body.get("text", {"format": {"type": "text"}}),
                            "metadata": body.get("metadata"),
                            "previous_response_id": body.get("previous_response_id"),
                            "instructions": body.get("instructions"),
                            "temperature": body.get("temperature"),
                            "top_p": body.get("top_p"),
                            "max_output_tokens": body.get("max_output_tokens"),
                            "truncation": body.get("truncation", "disabled"),
                            "output": [],
                            "usage": None,
                        }
                        yield _sse(
                            "response.created",
                            {
                                "type": "response.created",
                                "response": envelope,
                            },
                        )
                        yield _sse(
                            "response.in_progress",
                            {
                                "type": "response.in_progress",
                                "response": envelope,
                            },
                        )

                # reasoning_content — 完整 reasoning 生命周期
                # 参考 codex-app-transfer: emit_reasoning_delta
                # 也支持 reasoning_details 数组格式（某些 OpenAI 模型使用）
                reasoning_delta = delta.get("reasoning_content")
                if not reasoning_delta:
                    # 尝试 reasoning_details 数组
                    reasoning_details = delta.get("reasoning_details")
                    if isinstance(reasoning_details, list):
                        for detail in reasoning_details:
                            if isinstance(detail, dict) and detail.get("text"):
                                reasoning_delta = (reasoning_delta or "") + detail["text"]
                if reasoning_delta:
                    accumulated_reasoning += reasoning_delta
                    if not reasoning_started:
                        reasoning_started = True
                        reasoning_item_id = f"item_{resp_id}_reasoning"
                        events, seq = _start_reasoning_events(_sse, reasoning_item_id, seq)
                        for ev in events:
                            yield ev
                    seq += 1
                    yield _sse(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": reasoning_delta,
                            "sequence_number": seq,
                        },
                    )

                # 文本内容（含 think_tag 兜底拆分）
                # 参考 codex-app-transfer converter.rs:1156-1175
                # MiniMax 等 provider 把 thinking 塞进 content 的 <think>...</think> 标签
                content_delta = delta.get("content")
                if content_delta:
                    # think_tag 兜底拆分（仅在无原生 reasoning_content 时生效）
                    # 处理跨 chunk 的标签：缓冲内容，检查完整标签后再拆分
                    if think_tag_mode:
                        think_tag_buffer += content_delta
                        if "</think>" in think_tag_buffer:
                            # think_tag 结束：从缓冲区提取 reasoning + 剩余文本
                            before, after = think_tag_buffer.split("</think>", 1)
                            accumulated_reasoning = before
                            think_tag_mode = False
                            think_tag_buffer = ""
                            # 发送 reasoning 生命周期事件
                            if not reasoning_started:
                                reasoning_started = True
                                reasoning_item_id = f"item_{resp_id}_reasoning"
                                yield _sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": 0,
                                        "item": {
                                            "id": reasoning_item_id,
                                            "type": "reasoning",
                                            "status": "in_progress",
                                            "role": "assistant",
                                            "summary": [],
                                            "content": None,
                                            "encrypted_content": None,
                                        },
                                    },
                                )
                                yield _sse(
                                    "response.reasoning_summary_part.added",
                                    {
                                        "type": "response.reasoning_summary_part.added",
                                        "output_index": 0,
                                        "content_index": 0,
                                        "part": {"type": "summary_text", "text": ""},
                                    },
                                )
                            seq += 1
                            yield _sse(
                                "response.reasoning_summary_text.delta",
                                {
                                    "type": "response.reasoning_summary_text.delta",
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": "**Thinking**\n\n" + accumulated_reasoning,
                                    "sequence_number": seq,
                                },
                            )
                            events, reasoning_item = _close_reasoning_events(
                                _sse, reasoning_item_id, accumulated_reasoning
                            )
                            for ev in events:
                                yield ev
                            output_items.append(reasoning_item)
                            reasoning_started = False
                            # 如果 `</think>` 后还有文本，继续处理
                            if after:
                                content_delta = after
                            else:
                                continue
                        else:
                            # think_tag 累积中，不发送文本事件（已在上方缓冲）
                            continue
                    elif "<think>" in content_delta and not accumulated_reasoning and not reasoning_started:
                        # think_tag 开始：拆分前缀文本 + 进入 think_tag 模式
                        before, after = content_delta.split("<think>", 1)
                        if before:
                            # <think> 前有文本，先作为正常文本输出
                            content_delta = before
                            # think_tag 缓冲初始化（after 部分）
                            think_tag_mode = True
                            think_tag_buffer = after
                            # 不 continue，走下面的文本输出逻辑
                        else:
                            think_tag_mode = True
                            think_tag_buffer = after
                            continue
                    # reasoning 完成 → 关闭 reasoning item
                    if reasoning_started and not has_text:
                        reasoning_started = False
                        events, reasoning_item = _close_reasoning_events(_sse, reasoning_item_id, accumulated_reasoning)
                        for ev in events:
                            yield ev
                        output_items.append(reasoning_item)

                    has_text = True
                    if not text_started:
                        text_started = True
                        text_out_idx = 1 if accumulated_reasoning else 0
                        yield _sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": text_out_idx,
                                "item": {
                                    "id": text_item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                },
                            },
                        )
                        yield _sse(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "output_index": text_out_idx,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                            },
                        )
                        content_part_opened = True
                    accumulated_text += content_delta
                    seq += 1
                    yield _sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "output_index": text_out_idx,
                            "content_index": 0,
                            "delta": content_delta,
                            "sequence_number": seq,
                        },
                    )

                # 工具调用
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        item_id = f"item_{resp_id}_tc{idx}"
                        tool_calls_acc[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                            "item_id": item_id,
                            "started": False,
                        }
                    acc = tool_calls_acc[idx]
                    if tc_delta.get("id"):
                        acc["id"] = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        acc["name"] = func["name"]
                    args_delta = func.get("arguments", "")
                    if args_delta:
                        acc["arguments"] += args_delta
                        base_idx = (1 if accumulated_reasoning else 0) + (1 if has_text else 0)
                        out_idx = base_idx + sorted(tool_calls_acc.keys()).index(idx)
                        is_apply_patch = acc["name"] == "apply_patch"
                        # legacy MCP 工具重定向（参考 codex-app-transfer converter.rs:135-148）
                        # list_mcp_resources / read_mcp_resource / list_mcp_resource_templates → tool_search_call
                        is_legacy_mcp = acc["name"] in (
                            "list_mcp_resources",
                            "list_mcp_resource_templates",
                            "read_mcp_resource",
                        )
                        is_tool_search = acc["name"] == "tool_search"
                        if not acc["started"]:
                            acc["started"] = True
                            if is_apply_patch:
                                item_type = "custom_tool_call"
                            elif is_tool_search or is_legacy_mcp:
                                item_type = "tool_search_call"
                            else:
                                item_type = "function_call"
                            item_data: dict[str, Any] = {
                                "id": acc["item_id"],
                                "type": item_type,
                                "status": "in_progress",
                                "call_id": acc["id"],
                                "name": acc["name"],
                                "arguments": "",
                            }
                            # namespace 映射（参考 codex-app-transfer converter.rs:719-729）
                            ns = tool_namespace_map.get(acc["name"])
                            if ns:
                                item_data["namespace"] = ns
                            yield _sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": out_idx,
                                    "item": item_data,
                                },
                            )
                        # 事件类型选择
                        if is_apply_patch:
                            event_type = "response.custom_tool_call_input.delta"
                        elif is_tool_search or is_legacy_mcp:
                            event_type = "response.tool_search_call_arguments.delta"
                        else:
                            event_type = "response.function_call_arguments.delta"
                        yield _sse(
                            event_type,
                            {
                                "type": event_type,
                                "item_id": acc["item_id"],
                                "output_index": out_idx,
                                "delta": args_delta,
                            },
                        )

    except (RuntimeError, GeneratorExit):
        # 流式中断：缓存已积累的 tool call + 标记 incomplete
        try:
            tc_cache = get_tool_call_cache()
            for out_idx, acc in tool_calls_acc.items():
                if acc.get("id") and acc.get("name"):
                    tc_cache.save(acc["id"], acc["name"], acc["arguments"])
                # 发送 incomplete 的 output_item.done
                if acc.get("item_id"):
                    item_data = {
                        "type": acc.get("type", "function_call"),
                        "id": acc["item_id"],
                        "call_id": acc["id"],
                        "name": acc["name"],
                        "arguments": acc["arguments"],
                        "status": "incomplete",
                    }
                    yield _sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": out_idx,
                            "item": item_data,
                        },
                    )
            # 发送 incomplete 的 response.completed
            yield _sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "status": "incomplete",
                        "incomplete_details": {"reason": "stream_interrupted"},
                        "model": model,
                        "output": [],
                    },
                },
            )
        except Exception:
            pass
    finally:
        # 主动关闭内层 iterator，防止 Starlette 清理时 aclose() 冲突
        if hasattr(body_iterator, "aclose"):
            try:
                await body_iterator.aclose()
            except (Exception, GeneratorExit):
                pass


async def responses_endpoint(request: Request):
    """OpenAI Responses API 端点：转换格式后调用现有 chat_completions 管线"""
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body")

    model = body.get("model", "prism-auto")
    is_stream = body.get("stream", False)
    logger.info("Responses API: model=%s, stream=%s", model, is_stream)

    if not body.get("input") and not body.get("instructions"):
        return _error(400, "input or instructions is required")

    # 转换为 Chat Completions 格式
    chat_body = _responses_to_chat(body)
    is_stream = chat_body.get("stream", False)

    # 生成 Responses API 格式的 ID
    from prism_router.db import gen_ulid

    resp_id = f"resp_{gen_ulid()[:24]}"

    # 直接调用核心处理逻辑（无需伪造 Request）
    from prism_router.handlers import process_chat_request

    response = await process_chat_request(chat_body)

    # 非流式：转换响应体
    if isinstance(response, JSONResponse):
        try:
            chat_resp = json.loads(bytes(response.body))
        except Exception:
            return response  # 无法解析，原样返回
        # 错误响应转 Responses API 格式
        if response.status_code >= 400 or "error" in chat_resp:
            err = chat_resp.get("error", {})
            return JSONResponse(
                content={
                    "id": resp_id,
                    "object": "response",
                    "status": "failed",
                    "error": {
                        "type": err.get("type", "api_error"),
                        "code": err.get("code", str(response.status_code)),
                        "message": err.get("message", str(chat_resp)),
                    },
                },
                status_code=response.status_code,
            )
        responses_body = _chat_to_responses(chat_resp, resp_id, chat_body.get("model", ""), body=body)
        return JSONResponse(content=responses_body, status_code=response.status_code)

    # 流式：包装 SSE 事件转换
    if isinstance(response, StreamingResponse):
        model = chat_body.get("model", "")
        wrapped = _wrap_stream_as_responses(response.body_iterator, resp_id, model, body=body)
        return StreamingResponse(
            wrapped,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-Id": resp_id,
            },
        )

    # 其他响应（如错误）原样返回
    return response


async def _auto_compact_messages(body: dict, model: str, settings, req_log) -> dict:
    """主动上下文压缩：当消息数过多时裁剪旧消息。

    两种策略：
    1. 先尝试调用上游 LLM 做摘要压缩
    2. 失败则降级为直接裁剪，保留 system + 最近 N 条

    参考 codex-app-transfer 的 compact.rs 实现。
    """
    messages = body.get("messages", [])
    if len(messages) <= 10:
        return body

    # 保留最近 8 条消息（4 轮对话），压缩之前的
    keep_count = 8
    old_messages = messages[:-keep_count]
    recent_messages = messages[-keep_count:]

    # 策略 1：尝试调用上游 LLM 做摘要压缩
    compact_messages = old_messages + [{"role": "user", "content": COMPACT_SUMMARIZATION_PROMPT}]
    from prism_router.routing import route

    try:
        route_result = await route(model=model, messages=compact_messages, settings=settings)
        compact_body = {"model": model, "messages": compact_messages, "stream": False, "max_tokens": 4096}
        from prism_router.handlers import execute_request

        response = await execute_request(route_result, compact_body, stream=False, settings=settings)
        if isinstance(response, JSONResponse):
            chat_resp = json.loads(bytes(response.body))
            summary = chat_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if summary and len(summary) > 100:
                compacted = [{"role": "system", "content": f"{COMPACT_SUMMARY_PREFIX}\n{summary}"}] + recent_messages
                req_log.info("Auto-compact (LLM): %d msgs → %d msgs", len(messages), len(compacted))
                return dict(body, messages=compacted)
    except Exception as e:
        req_log.debug("Auto-compact LLM failed: %s, falling back to trim", e)

    # 策略 2：降级为直接裁剪
    # 提取 system 消息 + 最近消息
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) > keep_count:
        trimmed = non_system[-keep_count:]
        notice = {
            "role": "system",
            "content": f"[System: {len(non_system) - keep_count} earlier messages were trimmed to fit context window. The conversation continues from here.]",
        }
        compacted = system_msgs + [notice] + trimmed
        req_log.info("Auto-compact (trim): %d msgs → %d msgs", len(messages), len(compacted))
        return dict(body, messages=compacted)

    return body


# ── /responses/compact 端点（参考 codex-app-transfer） ──

COMPACT_SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n"
    "- **All user messages so far, verbatim or near-verbatim, in chronological order** — this preserves intent shifts that get lost otherwise\n"
    "- **Next Step** — the immediate next action aligned with the user's most recent explicit request. Include a **verbatim direct quote** from the most recent user message showing exactly where you left off; this prevents task drift.\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)

COMPACT_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking process. "
    "You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"
)

COMPACT_MAX_OUTPUT_TOKENS = 20000


async def compact_endpoint(request: Request):
    """OpenAI Responses API /responses/compact 端点。

    Codex CLI 在累计 token 超过 auto_compact_token_limit 时调用此端点，
    期望后端做上下文压缩——把整段对话历史摘要成一段简短的纯文本 summary。

    参考 codex-app-transfer 的 compact.rs 实现。
    """
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body")

    settings = get_settings()
    model = body.get("model", "")

    # 如果没指定模型，用配置中的 rewriter_model 或 complex tier
    if not model:
        if settings.rewriting.enabled and settings.rewriting.rewriter_model:
            model = settings.rewriting.rewriter_model
        else:
            model = settings.routing.complex or settings.routing.mid or ""

    # 提取 input（对话历史）
    raw_input = body.get("input")
    if raw_input is None or raw_input == []:
        input_items = []
    elif isinstance(raw_input, list):
        input_items = raw_input
    elif isinstance(raw_input, str):
        input_items = [{"type": "message", "role": "user", "content": raw_input}] if raw_input.strip() else []
    elif isinstance(raw_input, dict):
        input_items = [raw_input]
    else:
        input_items = [{"type": "message", "role": "user", "content": str(raw_input)}]

    # 注入压缩提示词作为最后一条 user message
    input_items.append(
        {
            "type": "message",
            "role": "user",
            "content": COMPACT_SUMMARIZATION_PROMPT,
        }
    )

    # 构造 Responses API 请求体，复用 _responses_to_chat 转换
    synthetic_body = {
        "model": model,
        "input": input_items,
        "stream": False,
        "max_output_tokens": COMPACT_MAX_OUTPUT_TOKENS,
    }

    # 透传 reasoning 和 tools（保持 thinking 模式兼容）
    if body.get("reasoning"):
        synthetic_body["reasoning"] = body["reasoning"]
    if body.get("tools"):
        synthetic_body["tools"] = body["tools"]

    # 转换为 Chat Completions 格式
    chat_body = _responses_to_chat(synthetic_body)

    # 路由到合适的模型
    from prism_router.db import gen_ulid
    from prism_router.routing import route

    request_id = gen_ulid()

    try:
        route_result = await route(
            model=model,
            messages=chat_body.get("messages", []),
            settings=settings,
        )
    except Exception as e:
        return _error(500, f"Compact routing failed: {e}", "server_error")

    # 调用上游（非流式）
    try:
        from prism_router.handlers import execute_request

        response = await execute_request(route_result, chat_body, stream=False, settings=settings)
    except Exception as e:
        return _error(502, f"Compact upstream failed: {e}", "upstream_error")

    if isinstance(response, JSONResponse):
        try:
            chat_resp = json.loads(bytes(response.body))
        except Exception:
            return _error(502, "Compact upstream returned invalid JSON", "upstream_error")

        if response.status_code >= 400 or "error" in chat_resp:
            err = chat_resp.get("error", {})
            return JSONResponse(
                content={
                    "id": f"resp_{request_id}",
                    "object": "response",
                    "status": "failed",
                    "error": {
                        "type": err.get("type", "api_error"),
                        "code": err.get("code", str(response.status_code)),
                        "message": err.get("message", str(chat_resp)),
                    },
                },
                status_code=response.status_code,
            )

        # 提取 summary
        choices = chat_resp.get("choices", [])
        if not choices:
            return _error(502, "Compact upstream returned empty choices", "upstream_error")

        summary = choices[0].get("message", {}).get("content", "")
        if not summary:
            return _error(502, "Compact upstream returned empty summary", "upstream_error")

        # 构造 compact 响应
        encrypted_content = f"{COMPACT_SUMMARY_PREFIX}\n{summary}"
        resp_id = f"resp_{request_id}"
        return JSONResponse(
            content={
                "id": resp_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": model,
                "output": [
                    {
                        "type": "compaction",
                        "encrypted_content": encrypted_content,
                    }
                ],
                "usage": _build_usage(chat_resp.get("usage", {})),
            }
        )

    return _error(502, "Compact upstream returned unexpected response type", "upstream_error")


# ── /v1/models 端点（含 auto_compact_token_limit） ──
# 参考 codex-app-transfer model_catalog.rs
# Codex CLI 读取 models 列表来获取 context_window 和 auto_compact_token_limit

AUTO_COMPACT_TRIGGER_PERCENT = 80  # context_window × 80% = auto_compact_token_limit


async def models_endpoint(request: Request):
    """返回可用模型列表，含 context_window 和 auto_compact_token_limit。

    Codex CLI 读取此端点来决定何时触发 /responses/compact。
    """
    settings = get_settings()
    models_list = []

    for model_key, cfg in settings.models.items():
        context_window = getattr(cfg, "context", 128000) or 128000
        auto_compact = context_window * AUTO_COMPACT_TRIGGER_PERCENT // 100
        models_list.append(
            {
                "id": model_key,
                "object": "model",
                "created": 0,
                "owned_by": "prism-router",
                "context_window": context_window,
                "max_context_window": context_window,
                "effective_context_window_percent": 95,
                "auto_compact_token_limit": auto_compact,
                "description": getattr(cfg, "description", ""),
                "source": getattr(cfg, "source", "unknown"),
            }
        )

    return JSONResponse(
        content={
            "object": "list",
            "data": models_list,
        }
    )
