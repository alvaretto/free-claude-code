"""VISION_MODEL routing, prompt extras and tool-history repair."""

from typing import Any

import pytest

from free_claude_code.application.prompt_extras import (
    apply_prompt_extras,
    routing_banner_line,
)
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.tool_history import (
    TRIMMED_TOOL_RESULT_TEXT,
    has_image_content,
    reconcile_tool_pairs,
)
from free_claude_code.core.reasoning import ReasoningControl
from free_claude_code.providers.history_replay import normalize_messages_history

_IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
}


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "MODEL": "deepseek/deepseek-v4-flash",
        "MODEL_OPUS": "deepseek/deepseek-v4-pro",
        "REASONING_OPUS": "high",
    }
    values.update(overrides)
    return Settings(**values)


def _request(content: Any, *, model: str = "claude-opus-4-8", **extra: Any):
    return MessagesRequest.model_validate(
        {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": content}],
            **extra,
        }
    )


def _read_png_request() -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-8",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [_IMAGE],
                        }
                    ],
                },
            ],
        }
    )


# ---------------------------------------------------------------- images


def test_has_image_content_top_level_and_nested() -> None:
    assert has_image_content(_request([{"type": "text", "text": "x"}, _IMAGE]).messages)
    assert has_image_content(_read_png_request().messages)
    assert not has_image_content(_request("plain text").messages)
    assert not has_image_content(
        _request(
            [{"type": "document", "source": {"type": "file", "file_id": "f"}}]
        ).messages
    )


# ---------------------------------------------------------------- vision route


def test_vision_model_overrides_tier_for_image_requests() -> None:
    router = ModelRouter(_settings(VISION_MODEL="deepseek/deepseek-v4-flash"))

    routed = router.resolve_messages_request(_request([_IMAGE]))

    assert routed.resolved.primary.provider_model_ref == "deepseek/deepseek-v4-flash"
    assert routed.request.model == "deepseek-v4-flash"
    assert routed.resolved.reasoning_preference is ReasoningPreference.OFF
    assert routed.reasoning.control is ReasoningControl.OFF


def test_vision_model_catches_images_inside_tool_results() -> None:
    router = ModelRouter(_settings(VISION_MODEL="deepseek/deepseek-v4-flash"))

    routed = router.resolve_messages_request(_read_png_request())

    assert routed.resolved.primary.provider_model == "deepseek-v4-flash"


def test_text_requests_keep_their_tier_route() -> None:
    router = ModelRouter(_settings(VISION_MODEL="deepseek/deepseek-v4-flash"))

    routed = router.resolve_messages_request(_request("hola"))

    assert routed.resolved.primary.provider_model == "deepseek-v4-pro"
    assert routed.reasoning.control is ReasoningControl.ON


def test_without_vision_model_images_follow_the_tier() -> None:
    routed = ModelRouter(_settings()).resolve_messages_request(_request([_IMAGE]))

    assert routed.resolved.primary.provider_model == "deepseek-v4-pro"


def test_vision_model_requires_a_configured_provider() -> None:
    with pytest.raises(ValueError, match="VISION_MODEL"):
        _settings(VISION_MODEL="nonexistent/model")


# ---------------------------------------------------------------- prompt extras


def _routed(settings: Settings, request: MessagesRequest):
    return ModelRouter(settings).resolve_messages_request(request)


def test_prompt_extras_noop_when_unset() -> None:
    settings = _settings()
    routed = _routed(settings, _request("hola", system="base"))

    assert apply_prompt_extras(routed, settings) is routed


def test_extra_prompt_and_banner_appended_after_string_system() -> None:
    settings = _settings(
        EXTRA_SYSTEM_PROMPT="Responde en español.", ENABLE_ROUTING_BANNER="true"
    )
    routed = apply_prompt_extras(
        _routed(settings, _request("hola", system="base")), settings
    )

    system = routed.request.system
    assert isinstance(system, list)
    assert [block.text for block in system][:2] == ["base", "Responde en español."]
    assert routing_banner_line("deepseek-v4-pro", thinking=True) in system[2].text


def test_banner_reports_vision_model_with_thinking_off() -> None:
    settings = _settings(
        VISION_MODEL="deepseek/deepseek-v4-flash", ENABLE_ROUTING_BANNER="true"
    )
    routed = apply_prompt_extras(_routed(settings, _request([_IMAGE])), settings)

    system = routed.request.system
    assert isinstance(system, list) and len(system) == 1
    assert "🤖 Modelo: deepseek-v4-flash | Thinking: OFF" in system[0].text


def test_extras_preserve_existing_system_blocks() -> None:
    settings = _settings(EXTRA_SYSTEM_PROMPT="extra")
    request = _request(
        "hola",
        system=[{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}],
    )
    routed = apply_prompt_extras(_routed(settings, request), settings)

    system = routed.request.system
    assert isinstance(system, list)
    assert [block.text for block in system] == ["a", "extra"]
    assert system[0].model_dump()["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------- tool history


def test_reconcile_adds_placeholder_for_unanswered_tool_use() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        },
        {"role": "user", "content": "sigue"},
    ]

    repaired, stats = reconcile_tool_pairs(messages)

    assert stats == {"added_tool_results": 1, "orphan_tool_results_as_content": 0}
    assert repaired[1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": TRIMMED_TOOL_RESULT_TEXT,
        },
        {"type": "text", "text": "sigue"},
    ]


def test_reconcile_inserts_user_turn_when_tool_use_is_last() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        }
    ]

    repaired, stats = reconcile_tool_pairs(messages)

    assert stats["added_tool_results"] == 1
    assert repaired[-1]["role"] == "user"
    assert repaired[-1]["content"][0]["tool_use_id"] == "t1"


def test_reconcile_reframes_orphan_tool_result_keeping_payload() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "ghost",
                    "content": [{"type": "text", "text": "PDF text"}, _IMAGE],
                }
            ],
        }
    ]

    repaired, stats = reconcile_tool_pairs(messages)

    assert stats["orphan_tool_results_as_content"] == 1
    assert repaired[0]["content"] == [{"type": "text", "text": "PDF text"}, _IMAGE]


def test_reconcile_leaves_valid_history_untouched() -> None:
    request = _read_png_request()
    dumped = [m.model_dump(mode="json", exclude_none=True) for m in request.messages]

    repaired, stats = reconcile_tool_pairs(dumped)

    assert not any(stats.values())
    assert repaired == dumped


def test_normalize_messages_history_applies_repair() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "lee f"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                    ],
                },
                {"role": "user", "content": "sigue"},
            ],
        }
    )

    normalized = normalize_messages_history(request)

    content = normalized.messages[2].content
    assert isinstance(content, list)
    assert content[0].type == "tool_result"
