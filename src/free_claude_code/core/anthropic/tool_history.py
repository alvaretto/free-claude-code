"""Image detection and tool_use/tool_result pairing repair for Messages history."""

from collections.abc import Iterable
from typing import Any

from .content import get_block_attr

# Placeholder for a tool call whose real result was trimmed from context before
# the request reached the proxy. Satisfies the pairing contract without
# fabricating a substantive answer.
TRIMMED_TOOL_RESULT_TEXT = (
    "[tool result omitted: trimmed from conversation context before reaching "
    "the provider]"
)


def has_image_content(messages: Iterable[Any]) -> bool:
    """True when any message carries an image, top-level or inside a tool_result."""
    for message in messages:
        content = get_block_attr(message, "content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = get_block_attr(block, "type")
            if block_type == "image":
                return True
            if block_type == "tool_result":
                inner = get_block_attr(block, "content")
                if isinstance(inner, list) and any(
                    get_block_attr(sub, "type") == "image" for sub in inner
                ):
                    return True
    return False


def _tool_use_ids(message: dict[str, Any] | None) -> list[str]:
    if message is None or message.get("role") != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        block["id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance(block.get("id"), str)
    ]


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return set()
    return {
        block["tool_use_id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_use_id"), str)
    }


def _placeholder_tool_result(tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": TRIMMED_TOOL_RESULT_TEXT,
    }


def _orphan_tool_result_blocks(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Reframe an unpaired tool_result as plain user content, keeping its payload."""
    inner = block.get("content")
    if isinstance(inner, str):
        text = inner if inner.strip() else TRIMMED_TOOL_RESULT_TEXT
        return [{"type": "text", "text": text}]
    if isinstance(inner, list):
        kept = [
            sub
            for sub in inner
            if isinstance(sub, dict) and sub.get("type") in {"text", "image"}
        ]
        if kept:
            return kept
    return [{"type": "text", "text": TRIMMED_TOOL_RESULT_TEXT}]


def reconcile_tool_pairs(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Repair orphaned tool_use / tool_result pairs in JSON-dumped messages.

    Claude Code conversations can leave two violations behind (mid-tool-use
    context trimming, or file contents front-loaded as a tool_result in the
    first message). Providers that enforce the pairing contract reject both:

    1. Assistant ``tool_use`` with no matching ``tool_result`` in the next
       message -> synthesize a placeholder ``tool_result`` (inserting a user
       message when none follows).
    2. User ``tool_result`` with no matching ``tool_use`` in the previous
       message -> reframe it as plain user content, preserving its payload.

    Returns ``(messages, stats)`` with the number of repairs of each kind.
    """
    stats = {"added_tool_results": 0, "orphan_tool_results_as_content": 0}

    reframed: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            reframed.append(message)
            continue
        valid_ids = set(_tool_use_ids(messages[index - 1] if index else None))
        new_content: list[Any] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") not in valid_ids
            ):
                new_content.extend(_orphan_tool_result_blocks(block))
                stats["orphan_tool_results_as_content"] += 1
                continue
            new_content.append(block)
        reframed.append({**message, "content": new_content})

    repaired: list[dict[str, Any]] = []
    index = 0
    while index < len(reframed):
        message = reframed[index]
        repaired.append(message)
        ids = _tool_use_ids(message)
        if ids:
            following = reframed[index + 1] if index + 1 < len(reframed) else None
            if following is not None and following.get("role") == "user":
                answered = _tool_result_ids(following)
                missing = [tool_id for tool_id in ids if tool_id not in answered]
                if missing:
                    content = following.get("content")
                    if isinstance(content, list):
                        rest = list(content)
                    elif isinstance(content, str) and content:
                        rest = [{"type": "text", "text": content}]
                    else:
                        rest = []
                    placeholders = [_placeholder_tool_result(i) for i in missing]
                    repaired.append({**following, "content": [*placeholders, *rest]})
                    stats["added_tool_results"] += len(missing)
                    index += 2
                    continue
            elif following is None or following.get("role") != "user":
                repaired.append(
                    {
                        "role": "user",
                        "content": [_placeholder_tool_result(i) for i in ids],
                    }
                )
                stats["added_tool_results"] += len(ids)
        index += 1

    return repaired, stats
