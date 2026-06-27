"""Request builder for the Sakana AI (Fugu) provider."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def _inject_extra_system_prompt(
    body: dict[str, Any], *, thinking_enabled: bool = False
) -> None:
    """Append language enforcement + routing banner to the OpenAI ``system`` message.

    Parity with ``providers/{deepseek,zai_anthropic,open_router}``: those inject the
    same two segments so every turn (a) obeys ``EXTRA_SYSTEM_PROMPT`` (force Spanish)
    and (b) starts with a ``🤖 Modelo: ... | Thinking: ...`` banner. Sakana rides the
    OpenAI-chat transport (``build_base_request_body`` inserts the system prompt as the
    first ``{"role": "system"}`` message), so the segments are appended to that message
    (or a new system message is prepended when none exists).
    """
    segments: list[str] = []

    extra = os.environ.get("EXTRA_SYSTEM_PROMPT", "").strip()
    if extra:
        segments.append(extra)

    if os.environ.get("ENABLE_ROUTING_BANNER", "true").lower() in ("1", "true", "yes"):
        model_id = str(body.get("model") or "?")
        model_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
        thinking_status = "ON" if thinking_enabled else "OFF"
        banner_line = f"🤖 Modelo: {model_name} | Thinking: {thinking_status}"
        segments.append(
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

    if not segments:
        return

    appended = "\n\n".join(segments)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    first = messages[0] if messages else None
    if (
        isinstance(first, dict)
        and first.get("role") == "system"
        and isinstance(first.get("content"), str)
    ):
        existing = first["content"]
        first["content"] = f"{existing}\n\n{appended}" if existing else appended
    else:
        messages.insert(0, {"role": "system", "content": appended})


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from an Anthropic request for Sakana Fugu."""
    logger.debug(
        "SAKANA_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)
    if extra_body:
        body["extra_body"] = extra_body

    # Restore language enforcement + routing banner (parity with DeepSeek/Z.ai/OpenRouter).
    _inject_extra_system_prompt(body, thinking_enabled=thinking_enabled)

    logger.info(
        "MODEL_ROUTING: model={} thinking={} provider=sakana",
        body.get("model"),
        thinking_enabled,
    )
    logger.debug(
        "SAKANA_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
