"""Request builder and DeepSeek native Anthropic compatibility sanitizer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import dump_raw_messages_request
from providers.exceptions import InvalidRequestError

# Block types not supported on DeepSeek partial Anthropic-compatible API.
_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)

# Block types silently stripped for DeepSeek since the content is typically
# also provided via tool_result (e.g. Claude Code attaches PDFs as document
# blocks alongside a Read tool_result containing the text).
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}

# DeepSeek models whose Anthropic-compatible endpoint accepts ``image`` blocks.
# Verified live 2026-10-02: deepseek-v4-flash reads text/shapes from PNGs;
# deepseek-v4-pro answers "cannot see the image". Neither accepts ``document``.
_VISION_CAPABLE_MODELS = frozenset({"deepseek-v4-flash"})
_HOISTED_IMAGE_TEXT = "[image attached after this tool result]"
_HOISTED_IMAGE_BLOCK = {"type": "text", "text": _HOISTED_IMAGE_TEXT}


def _model_supports_images(model: Any) -> bool:
    """True when the DeepSeek model id accepts image input."""
    return isinstance(model, str) and model in _VISION_CAPABLE_MODELS


def _strip_unsupported_attachment_blocks(
    messages: Any, *, keep_images: bool = False
) -> Any:
    """Remove image/document blocks that DeepSeek cannot process.

    Claude Code sends PDFs as ``document`` blocks alongside a Read ``tool_result``
    that already contains the extracted text. Stripping preserves the request
    instead of failing with an unsupported block error.

    With ``keep_images`` (vision-capable model), top-level images are kept and
    images nested in a ``tool_result`` (e.g. Claude Code's Read of a PNG) are
    hoisted to the end of the same user message, because DeepSeek requires
    ``tool_result.content`` to be a string.
    """
    if not isinstance(messages, list):
        return messages

    strippable = (
        _STRIPPABLE_MESSAGE_BLOCK_TYPES - {"image"}
        if keep_images
        else _STRIPPABLE_MESSAGE_BLOCK_TYPES
    )
    stripped: list[Any] = []
    top_level_dropped: dict[str, int] = {}
    nested_dropped: dict[str, int] = {}
    placeholder_replacements = 0
    hoisted_images = 0

    for message in messages:
        if not isinstance(message, dict):
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue

        new_content: list[Any] = []
        hoisted: list[Any] = []
        message_dropped_attachment = False
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in strippable:
                    top_level_dropped[btype] = top_level_dropped.get(btype, 0) + 1
                    message_dropped_attachment = True
                    continue
                if btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        filtered_inner: list[Any] = []
                        hoisted_before = len(hoisted)
                        for sub in inner:
                            if not isinstance(sub, dict):
                                filtered_inner.append(sub)
                                continue
                            sub_type = sub.get("type")
                            if keep_images and sub_type == "image":
                                hoisted.append(sub)
                                continue
                            if sub_type in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                                nested_dropped[sub_type] = (
                                    nested_dropped.get(sub_type, 0) + 1
                                )
                                continue
                            filtered_inner.append(sub)
                        if len(hoisted) > hoisted_before and not filtered_inner:
                            filtered_inner = [_HOISTED_IMAGE_BLOCK]
                        elif not filtered_inner:
                            filtered_inner = [_OMITTED_ATTACHMENT_BLOCK]
                            placeholder_replacements += 1
                        new_block = dict(block)
                        new_block["content"] = filtered_inner
                        new_content.append(new_block)
                        continue
            new_content.append(block)
        # Anthropic requires tool_result blocks first; images go after them.
        new_content.extend(hoisted)
        hoisted_images += len(hoisted)
        if not new_content and message_dropped_attachment:
            new_content = [_OMITTED_ATTACHMENT_BLOCK]
            placeholder_replacements += 1
        new_msg = dict(message)
        new_msg["content"] = new_content
        stripped.append(new_msg)

    if hoisted_images:
        logger.info(
            "DEEPSEEK_REQUEST: hoisted {} image block(s) out of tool_result "
            "for vision-capable model",
            hoisted_images,
        )
    if top_level_dropped or nested_dropped:
        logger.warning(
            "DEEPSEEK_REQUEST: stripped unsupported attachment blocks "
            "(top_level={} nested_in_tool_result={} placeholder_tool_results={}). "
            "This DeepSeek model cannot read them; the model will not see this content.",
            dict(top_level_dropped),
            dict(nested_dropped),
            placeholder_replacements,
        )
    return stripped


def _is_server_listed_tool(tool: Mapping[str, Any]) -> bool:
    """True for Anthropic web_search / web_fetch-style tool definitions (listed tools)."""
    name = (tool.get("name") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.get("type")
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(
    blocks: Any, *, where: str, allow_images: bool = False
) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "image" and allow_images:
            continue
        if btype in _UNSUPPORTED_MESSAGE_BLOCK_TYPES:
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result" and "content" in block:
            _walk_block_list_for_unsupported(
                block["content"],
                where=f"{where} (tool_result content)",
                allow_images=allow_images,
            )


def _validate_deepseek_native_request_dict(
    data: dict[str, Any], *, allow_images: bool = False
) -> None:
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError(
            "DeepSeek native does not support mcp_servers on requests."
        )

    for tool in data.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek native does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(data.get("messages") or ()):
        if not isinstance(message, dict):
            continue
        c = message.get("content")
        if isinstance(c, list):
            _walk_block_list_for_unsupported(
                c, where=f"messages[{i}].content", allow_images=allow_images
            )
        if isinstance(c, str) and "<think>" in c:
            # Unusual, but block encoded redacted content — treat as unsafe for DeepSeek.
            pass

    system = data.get("system")
    if isinstance(system, list):
        _walk_block_list_for_unsupported(system, where="system")


def _has_tool_history_blocks(message: Mapping[str, Any]) -> bool:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if role == "assistant" and btype == "tool_use":
            return True
        if role == "user" and btype == "tool_result":
            return True
    return False


def _has_replayable_thinking_before_tool_use(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False

    has_thinking = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking" and isinstance(block.get("thinking"), str):
            has_thinking = bool(block["thinking"])
            continue
        if btype == "tool_use":
            return has_thinking
    return False


def _has_tool_history(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_tool_history_blocks(message):
            return True
    return False


def _has_replayable_tool_thinking(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_replayable_thinking_before_tool_use(
            message
        ):
            return True
    return False


def _remove_deepseek_thinking_hints(data: dict[str, Any]) -> None:
    """Remove request hints that can keep DeepSeek in thinking mode after fallback."""
    output_config = data.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        cleaned_output_config = dict(output_config)
        cleaned_output_config.pop("effort", None)
        if cleaned_output_config:
            data["output_config"] = cleaned_output_config
        else:
            data.pop("output_config", None)

    context_management = data.get("context_management")
    if not isinstance(context_management, dict):
        return
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return
    filtered_edits = [
        edit
        for edit in edits
        if not (
            isinstance(edit, dict)
            and isinstance(edit.get("type"), str)
            and edit["type"].startswith("clear_thinking_")
        )
    ]
    if len(filtered_edits) == len(edits):
        return
    cleaned_context_management = dict(context_management)
    if filtered_edits:
        cleaned_context_management["edits"] = filtered_edits
        data["context_management"] = cleaned_context_management
    else:
        cleaned_context_management.pop("edits", None)
        if cleaned_context_management:
            data["context_management"] = cleaned_context_management
        else:
            data.pop("context_management", None)


def sanitize_deepseek_messages_for_native(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Filter assistant content for DeepSeek: unsigned ``thinking`` is allowed; no ``redacted_thinking``."""
    if not isinstance(messages, list):
        return messages

    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        if not thinking_enabled:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
            ]
        else:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict) and block.get("type") == "redacted_thinking"
                )
            ]
        new_msg = dict(message)
        new_msg["content"] = filtered or ""
        sanitized.append(new_msg)
    return sanitized


def _serialize_tool_result_content(content: Any) -> str:
    """Serialize tool_result content to string for DeepSeek API.

    DeepSeek's Anthropic-compatible API expects tool_result.content to be a string,
    not an array of content blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_tool_result_content(messages: Any) -> Any:
    """Normalize tool_result content to strings for DeepSeek API compatibility."""
    if not isinstance(messages, list):
        return messages

    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue

        # Process content blocks
        new_content: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            if block.get("type") == "tool_result":
                # Normalize tool_result content to string
                normalized_block = dict(block)
                normalized_block["content"] = _serialize_tool_result_content(
                    block.get("content")
                )
                new_content.append(normalized_block)
            else:
                new_content.append(block)

        new_msg = dict(message)
        new_msg["content"] = new_content
        normalized.append(new_msg)

    return normalized


def _strip_reasoning_content_when_native(messages: Any) -> Any:
    """``reasoning_content`` is OpenAI-helper metadata; not part of native Anthropic body."""
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        msg = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(msg)
    return out


# Placeholder used when a tool call's real result was trimmed from context before
# the request reached the proxy. Satisfies the Anthropic/DeepSeek pairing contract
# without fabricating a substantive answer.
_TRIMMED_TOOL_RESULT_TEXT = "[tool result omitted: trimmed from conversation context before reaching the provider]"


def _assistant_tool_use_ids(message: Any) -> list[str]:
    """Ordered ``tool_use`` ids declared in an assistant message."""
    ids: list[str] = []
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ids
    content = message.get("content")
    if not isinstance(content, list):
        return ids
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tid = block.get("id")
            if isinstance(tid, str):
                ids.append(tid)
    return ids


def _user_tool_result_ids(message: Any) -> set[str]:
    """``tool_use_id`` values answered by ``tool_result`` blocks in a user message."""
    ids: set[str] = set()
    if not isinstance(message, dict) or message.get("role") != "user":
        return ids
    content = message.get("content")
    if not isinstance(content, list):
        return ids
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tid = block.get("tool_use_id")
            if isinstance(tid, str):
                ids.add(tid)
    return ids


def _placeholder_tool_result(tool_use_id: str) -> dict[str, str]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": _TRIMMED_TOOL_RESULT_TEXT,
    }


def _orphan_tool_result_as_text(block: dict[str, Any]) -> dict[str, str]:
    """Reframe an orphaned ``tool_result`` as a plain text block.

    Conversion (vs dropping) preserves the payload — e.g. the extracted text of a
    PDF that Claude Code front-loads as a ``tool_result`` in the first message, a
    pattern DeepSeek otherwise rejects with HTTP 400 — while removing the unpaired
    ``tool_use_id`` that violates the contract. Content has already been
    normalized to a string by :func:`_normalize_tool_result_content`.
    """
    raw = block.get("content")
    text = raw if isinstance(raw, str) and raw.strip() else _TRIMMED_TOOL_RESULT_TEXT
    return {"type": "text", "text": text}


def _reconcile_tool_pairs(messages: Any) -> tuple[Any, dict[str, int]]:
    """Repair orphaned ``tool_use`` / ``tool_result`` pairs before forwarding.

    DeepSeek's Anthropic-compatible endpoint enforces the Messages tool-pairing
    contract and rejects violations with HTTP 400 ``invalid_request_error``
    (surfaced to the client as the opaque ``Invalid request sent to provider.``).
    Claude Code conversations routinely leave one of two violations behind — from
    mid-tool-use context trimming, or from front-loading file contents as a
    ``tool_result`` in the first message — and both are repaired here,
    meaning-preserved:

    1. Assistant ``tool_use`` with no matching ``tool_result`` in the *next*
       message -> synthesize a placeholder ``tool_result`` (the real result was
       trimmed from context). When no user message follows, one is inserted.
    2. User ``tool_result`` with no matching ``tool_use`` in the *previous*
       message -> reframe the orphaned block as a text block (its payload is
       preserved; the unpaired ``tool_use_id`` that triggers the 400 is removed).

    Returns ``(messages, stats)`` where ``stats`` counts the repairs applied.
    """
    stats = {"added_tool_results": 0, "orphan_tool_results_as_text": 0}
    if not isinstance(messages, list) or not messages:
        return messages, stats

    # Pass A: reframe orphan tool_result blocks (no matching tool_use in prev msg)
    # as text so their payload survives without violating the pairing contract.
    pass_a: list[Any] = []
    for i, message in enumerate(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), list)
        ):
            prev = messages[i - 1] if i > 0 else None
            valid_ids = set(_assistant_tool_use_ids(prev))
            new_content: list[Any] = []
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") not in valid_ids
                ):
                    new_content.append(_orphan_tool_result_as_text(block))
                    stats["orphan_tool_results_as_text"] += 1
                    continue
                new_content.append(block)
            new_msg = dict(message)
            new_msg["content"] = new_content
            pass_a.append(new_msg)
        else:
            pass_a.append(message)

    # Pass B: every assistant tool_use must be answered in the next user message.
    result: list[Any] = []
    n = len(pass_a)
    i = 0
    while i < n:
        message = pass_a[i]
        result.append(message)
        ids = _assistant_tool_use_ids(message)
        if ids:
            nxt = pass_a[i + 1] if i + 1 < n else None
            if isinstance(nxt, dict) and nxt.get("role") == "user":
                missing = [tid for tid in ids if tid not in _user_tool_result_ids(nxt)]
                if missing:
                    placeholders = [_placeholder_tool_result(tid) for tid in missing]
                    new_next = dict(nxt)
                    content = new_next.get("content")
                    if isinstance(content, list):
                        new_next["content"] = placeholders + list(content)
                    elif isinstance(content, str) and content:
                        new_next["content"] = placeholders + [
                            {"type": "text", "text": content}
                        ]
                    else:
                        new_next["content"] = placeholders
                    stats["added_tool_results"] += len(missing)
                    result.append(new_next)
                    i += 2
                    continue
            else:
                # No following user message: insert one carrying the placeholders.
                placeholders = [_placeholder_tool_result(tid) for tid in ids]
                result.append({"role": "user", "content": placeholders})
                stats["added_tool_results"] += len(ids)
        i += 1

    return result, stats


def _inject_extra_system_prompt(
    data: dict[str, Any], *, thinking_enabled: bool = False
) -> None:
    """Append extras to the system prompt sent to DeepSeek.

    Two appended blocks (in this order, at the END so they override earlier
    instructions):

    1. ``EXTRA_SYSTEM_PROMPT`` env var content (e.g. force Spanish responses).
       Skipped if the env var is empty.
    2. Dynamic routing banner directive — forces every response to start with
       a line ``🤖 Modelo: <id> | Thinking: <ON|OFF>`` so the user can see at
       a glance which DeepSeek model and reasoning mode handled the turn.
    """
    blocks_to_append: list[dict[str, str]] = []

    extra = os.environ.get("EXTRA_SYSTEM_PROMPT", "").strip()
    if extra:
        blocks_to_append.append({"type": "text", "text": extra})

    if os.environ.get("ENABLE_ROUTING_BANNER", "true").lower() in ("1", "true", "yes"):
        model_id = str(data.get("model") or "?")
        model_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
        thinking_status = "ON" if thinking_enabled else "OFF"
        banner_line = f"🤖 Modelo: {model_name} | Thinking: {thinking_status}"
        banner_directive = (
            "INSTRUCCIÓN OBLIGATORIA DE BANNER DE ROUTING: Antes de cualquier "
            "otro contenido, tu respuesta DEBE comenzar EXACTAMENTE con la "
            "siguiente línea literal (incluyendo el emoji, sin bloque de "
            "código, sin variaciones, sin traducir):\n\n"
            f"{banner_line}\n\n"
            "Después deja una línea en blanco y continúa con tu respuesta "
            "normal. Esta regla es OBLIGATORIA en TODAS las respuestas, "
            "incluidas las muy cortas (sí, no, OK). Si la respuesta es de "
            "una sola línea, el banner sigue siendo la primera línea. NO "
            "modifiques el texto del banner."
        )
        blocks_to_append.append({"type": "text", "text": banner_directive})

    if not blocks_to_append:
        return

    existing = data.get("system")

    if existing is None or existing == "":
        data["system"] = blocks_to_append
    elif isinstance(existing, str):
        data["system"] = [{"type": "text", "text": existing}] + blocks_to_append
    elif isinstance(existing, list):
        data["system"] = list(existing) + blocks_to_append
    else:
        logger.warning(
            "DEEPSEEK_REQUEST: cannot inject extras, unexpected system type: {}",
            type(existing).__name__,
        )


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build a DeepSeek ``/v1/messages`` JSON body (Anthropic format)."""
    logger.debug(
        "DEEPSEEK_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    keep_images = _model_supports_images(data.get("model"))
    if "messages" in data:
        data["messages"] = _strip_unsupported_attachment_blocks(
            data["messages"], keep_images=keep_images
        )
    _validate_deepseek_native_request_dict(data, allow_images=keep_images)
    data.pop("extra_body", None)

    has_tool_history = _has_tool_history(data)
    has_replayable_tool_thinking = _has_replayable_tool_thinking(data)
    unsafe_tool_followup = has_tool_history and not has_replayable_tool_thinking
    effective_thinking_enabled = thinking_enabled and not unsafe_tool_followup

    _inject_extra_system_prompt(data, thinking_enabled=effective_thinking_enabled)
    logger.info(
        "MODEL_ROUTING: model={} thinking={} (requested_thinking={})",
        data.get("model"),
        effective_thinking_enabled,
        thinking_enabled,
    )
    if thinking_enabled:
        if unsafe_tool_followup:
            logger.debug(
                "DEEPSEEK_REQUEST: disabling thinking for tool follow-up without "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
            _remove_deepseek_thinking_hints(data)
        elif has_tool_history:
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for tool follow-up with "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
        elif data.get("tools") or data.get("tool_choice"):
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for initial tool request "
                "model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )

    thinking_cfg = data.pop("thinking", None)
    if effective_thinking_enabled and isinstance(thinking_cfg, dict):
        thinking_payload: dict[str, Any] = {"type": "enabled"}
        budget_tokens = thinking_cfg.get("budget_tokens")
        if isinstance(budget_tokens, int):
            thinking_payload["budget_tokens"] = budget_tokens
        data["thinking"] = thinking_payload

    if "messages" in data:
        data["messages"] = _strip_reasoning_content_when_native(
            _normalize_tool_result_content(
                sanitize_deepseek_messages_for_native(
                    data["messages"],
                    thinking_enabled=effective_thinking_enabled,
                )
            )
        )
        # Final repair: reconcile orphaned tool_use/tool_result pairs left behind
        # by context trimming. Without this DeepSeek rejects the turn with HTTP
        # 400 invalid_request_error -> opaque "Invalid request sent to provider".
        data["messages"], tool_pair_stats = _reconcile_tool_pairs(data["messages"])
        if any(tool_pair_stats.values()):
            logger.warning(
                "DEEPSEEK_REQUEST: reconciled orphaned tool pairs "
                "(added_tool_results={added_tool_results} "
                "orphan_tool_results_as_text={orphan_tool_results_as_text}); these "
                "would otherwise be a 400 invalid_request_error from DeepSeek.",
                **tool_pair_stats,
            )
    if "max_tokens" not in data or data.get("max_tokens") is None:
        data["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    data["stream"] = True

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        data.get("model"),
        len(data.get("messages", [])),
        len(data.get("tools", [])),
    )
    return data
