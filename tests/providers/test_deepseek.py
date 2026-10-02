"""Tests for DeepSeek native Anthropic Messages provider."""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import (
    ContentBlockImage,
    Message,
    MessagesRequest,
    Tool,
)
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.deepseek import (
    DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
    DEEPSEEK_DEFAULT_BASE,
    DeepSeekProvider,
)
from providers.exceptions import InvalidRequestError


@pytest.fixture
def deepseek_config():
    return ProviderConfig(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.anthropic_messages.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config)


def test_default_base_url_alias():
    assert DEEPSEEK_DEFAULT_BASE == DEEPSEEK_ANTHROPIC_DEFAULT_BASE
    assert DEEPSEEK_ANTHROPIC_DEFAULT_BASE == "https://api.deepseek.com/anthropic"


def test_init(deepseek_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = DeepSeekProvider(deepseek_config)
    assert provider._api_key == "test_deepseek_key"
    assert provider._base_url == "https://api.deepseek.com/anthropic"
    assert mock_client.called


def test_request_headers_includes_x_api_key(deepseek_provider):
    h = deepseek_provider._request_headers()
    assert h["x-api-key"] == "test_deepseek_key"
    assert h["Content-Type"] == "application/json"
    assert h["Accept"] == "text/event-stream"


def test_build_request_body_native_shape(deepseek_provider):
    request = MessagesRequest(
        model="deepseek-v4-pro",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="S",
    )
    body = deepseek_provider._build_request_body(request)
    assert body["model"] == "deepseek-v4-pro"
    assert body["stream"] is True
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] in (
        "Hello",
        [{"type": "text", "text": "Hello"}],
    )
    # The routing banner (and any EXTRA_SYSTEM_PROMPT) is appended, so ``system``
    # becomes a list of text blocks whose first block preserves the original prompt.
    if isinstance(body["system"], list):
        assert body["system"][0]["text"] == "S"
    else:
        assert body["system"] == "S"
    assert body["max_tokens"] == 100


def test_system_role_message_is_hoisted_into_system_field(deepseek_provider):
    """Claude Code >= 2.1.x emits ``system``-role messages inside ``messages``.

    These must be normalized into the top-level ``system`` field so the FastAPI
    ingress does not 422 and DeepSeek's native endpoint only sees user/assistant
    roles. Regression test for the 422 literal_error on ``messages[1].role``.
    """
    request = MessagesRequest(
        model="deepseek-v4-pro",
        max_tokens=100,
        messages=[
            Message(role="user", content="Hola"),
            Message(role="system", content="system note"),
            Message(role="user", content="sigue"),
        ],
    )
    # Normalization happens at parse time (model_validator).
    assert [m.role for m in request.messages] == ["user", "user"]
    assert request.system is not None
    assert any("system note" in b.text for b in request.system)

    body = deepseek_provider._build_request_body(request)
    assert all(m["role"] in ("user", "assistant") for m in body["messages"])


def test_build_request_body_default_max_tokens(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = deepseek_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert "extra_body" not in body


def test_build_request_body_tool_list_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert body["tools"][0]["name"] == "Read"


def test_build_request_body_tool_choice_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert body["tool_choice"] == {"type": "auto"}


def test_build_request_body_respects_global_thinking_disable():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._build_request_body(request)
    assert "thinking" not in body


def test_preserve_unsigned_thinking_when_thinking_on(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "plain",
                            "signature": None,
                        },
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    blocks = body["messages"][0]["content"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "plain"


def test_strip_redacted_thinking_when_thinking_on(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    types = {b["type"] for b in body["messages"][0]["content"]}
    assert "redacted_thinking" not in types
    assert "text" in types


def test_tool_history_with_replayable_thinking_preserves_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "hidden",
                            "signature": "sig_123",
                        },
                        {"type": "redacted_thinking", "data": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "output_config": {"effort": "high"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert body["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }
    assert body["output_config"] == {"effort": "high"}
    assistant_blocks = body["messages"][0]["content"]
    assert [block["type"] for block in assistant_blocks] == ["thinking", "tool_use"]
    assert assistant_blocks[0]["thinking"] == "hidden"
    assert assistant_blocks[0]["signature"] == "sig_123"
    assert assistant_blocks[1]["name"] == "Read"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_tool_history_with_unsigned_thinking_preserves_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plain"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["content"][0] == {
        "type": "thinking",
        "thinking": "plain",
    }


def test_tool_history_without_thinking_disables_thinking_and_hints(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "other_edit", "keep": "all"},
                ],
                "other": True,
            },
            "output_config": {"effort": "high", "format": "text"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert "thinking" not in body
    assert body["context_management"] == {
        "edits": [{"type": "other_edit", "keep": "all"}],
        "other": True,
    }
    assert body["output_config"] == {"format": "text"}
    assert body["tools"][0]["name"] == "Read"
    assert body["tool_choice"] == {"type": "auto"}
    assert body["messages"][0]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_tool_history_with_empty_thinking_disables_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert "thinking" not in body
    assert [block["type"] for block in body["messages"][0]["content"]] == ["tool_use"]


def test_thinking_off_strips_thinking_history():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "sec"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
        }
    )
    body = provider._build_request_body(request)
    for b in body["messages"][0]["content"]:
        assert b["type"] != "thinking"
    assert "sec" not in str(body["messages"])


def test_passthrough_tool_use_and_result(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "n",
                            "input": {"a": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["messages"][0]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_preflight_strips_user_image():
    """Image blocks are silently stripped (DeepSeek lacks vision); request must not fail."""
    request = MessagesRequest(
        model="m",
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "YQ==",
                        },
                    )
                ],
            )
        ],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    # Should not raise; image is stripped.
    provider.preflight_stream(request, thinking_enabled=True)
    body = provider._build_request_body(request)
    content = body["messages"][0]["content"]
    block_types = [b["type"] for b in content] if isinstance(content, list) else []
    assert "image" not in block_types


def test_preflight_rejects_mcp_servers():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        mcp_servers=[{"type": "url", "url": "https://x"}],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="mcp_servers"):
        provider.preflight_stream(request)


def test_preflight_rejects_listed_server_tools_in_tools_list():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        tools=[Tool(name="web_search", type="web_search_20250305", input_schema={})],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="web_search"):
        provider.preflight_stream(request)


def test_preflight_rejects_server_tool_result_blocks():
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"q": "a"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s1",
                            "content": [],
                        },
                    ],
                }
            ],
        }
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match=r"web_search_tool_result|server"):
        provider.preflight_stream(request)


def test_strip_reasoning_content_not_forwarded(deepseek_provider):
    # ``reasoning_content`` is for OpenAI helpers only, not in native body.
    request = MessagesRequest(
        model="m",
        messages=[
            Message(
                role="assistant",
                content="hi",
                reasoning_content="r",
            )
        ],
    )
    body = deepseek_provider._build_request_body(request)
    assert "reasoning_content" not in body["messages"][0]


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )
    called: dict[str, str] = {}

    async def fake_send(request, *args, **kwargs):
        called["path"] = request.url.path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_closed = False
        mock_resp.raise_for_status = lambda: None

        async def aiter():
            if False:  # pragma: no cover
                yield ""

        mock_resp.aiter_lines = aiter
        mock_resp.aclose = AsyncMock()
        return mock_resp

    deepseek_provider._client.send = fake_send
    _ = [x async for x in deepseek_provider.stream_response(request, request_id="r1")]

    assert called["path"] == "/anthropic/messages"


def test_drops_extra_body_from_canonical_request(deepseek_provider):
    raw = {
        "model": "m",
        "max_tokens": 3,
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"note": 1},
    }
    r = MessagesRequest.model_validate(raw)
    body = deepseek_provider._build_request_body(r)
    assert "extra_body" not in body


def test_normalizes_tool_result_content_array_to_string(deepseek_provider):
    """Test that tool_result content arrays are normalized to strings for DeepSeek API."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "list_dir",
                            "input": {"path": "/"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "file1.txt"},
                                {"type": "text", "text": "file2.txt"},
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    # Verify tool_result content is now a string
    user_msg = body["messages"][1]
    tool_result = user_msg["content"][0]
    assert tool_result["type"] == "tool_result"
    assert isinstance(tool_result["content"], str)
    assert "file1.txt" in tool_result["content"]
    assert "file2.txt" in tool_result["content"]


def test_strips_document_blocks_for_deepseek(deepseek_provider):
    """Document blocks (e.g. PDFs from Claude Code) are stripped since DeepSeek can't process them."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "PDF text extracted",
                        },
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_abc"},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    # Document block is stripped. The tool_result is orphaned (no preceding
    # tool_use), which DeepSeek rejects with HTTP 400, so it is reframed as a
    # text block — the extracted PDF text is preserved, the unpaired
    # ``tool_use_id`` is removed.
    content = body["messages"][0]["content"]
    block_types = [block["type"] for block in content]
    assert "document" not in block_types
    assert "tool_result" not in block_types
    assert "text" in block_types
    assert any(
        b["type"] == "text" and "PDF text extracted" in b["text"] for b in content
    )


def test_strips_image_blocks_for_deepseek(deepseek_provider):
    """Image blocks are stripped for DeepSeek since it doesn't support vision."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][0]["content"]
    block_types = [block["type"] for block in content]
    assert "image" not in block_types
    assert "text" in block_types


def test_normalizes_tool_result_content_dict_to_string(deepseek_provider):
    """Test that tool_result content dicts are normalized to JSON strings."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_data",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": {"status": "success", "data": [1, 2, 3]},
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    # Verify tool_result content is now a JSON string
    user_msg = body["messages"][1]
    tool_result = user_msg["content"][0]
    assert tool_result["type"] == "tool_result"
    assert isinstance(tool_result["content"], str)
    assert "status" in tool_result["content"]
    assert "success" in tool_result["content"]


def test_strips_image_block_inside_tool_result(deepseek_provider):
    """Image blocks nested inside tool_result.content are stripped, not rejected."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"path": "shot.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "screenshot saved"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]["content"][0]
    assert tool_result["type"] == "tool_result"
    # After stripping + string-normalization, no base64/image marker survives.
    assert isinstance(tool_result["content"], str)
    assert "screenshot saved" in tool_result["content"]
    assert "base64" not in tool_result["content"]
    assert "abc" not in tool_result["content"]


def test_image_only_tool_result_replaced_with_placeholder(deepseek_provider):
    """A tool_result whose only inner block is an image becomes a placeholder string."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Screenshot",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert isinstance(tool_result["content"], str)
    assert tool_result["content"] != ""
    assert "attachment omitted" in tool_result["content"].lower()
    assert "image or document inputs" in tool_result["content"].lower()


def test_document_only_tool_result_replaced_with_generic_placeholder(
    deepseek_provider,
):
    """A document-only tool_result uses the generic attachment placeholder."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "paper.pdf"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "file",
                                        "file_id": "file_pdf",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert isinstance(tool_result["content"], str)
    assert "attachment omitted" in tool_result["content"].lower()
    assert "document inputs" in tool_result["content"].lower()
    assert "image omitted" not in tool_result["content"].lower()


def test_image_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level image-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "attachment omitted" in content[0]["text"].lower()
    assert "image or document inputs" in content[0]["text"].lower()


def test_document_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level document-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_pdf"},
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "attachment omitted" in content[0]["text"].lower()
    assert "document inputs" in content[0]["text"].lower()


def test_warns_when_stripping_attachment_blocks(deepseek_provider, caplog):
    """A warning is emitted when image/document blocks are dropped so users notice."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Screenshot",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stripped unsupported attachment blocks" in r.message for r in warnings)


def test_no_warning_when_no_attachments(deepseek_provider, caplog):
    """No warning is emitted on plain text-only requests."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)

    assert not any(
        "stripped unsupported attachment blocks" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# Orphaned tool_use / tool_result reconciliation
#
# DeepSeek's native Anthropic endpoint rejects unpaired tool blocks with HTTP
# 400 invalid_request_error (surfaced to the client as the opaque
# "Invalid request sent to provider."). Claude Code conversations trimmed
# mid-tool-use routinely leave these behind; build_request_body must repair them.
# ---------------------------------------------------------------------------


def _roles_and_blocks(body):
    """[(role, [block types or 'str']) ...] for compact assertions."""
    out = []
    for m in body["messages"]:
        content = m["content"]
        if isinstance(content, str):
            out.append((m["role"], "str"))
        else:
            out.append((m["role"], [b.get("type") for b in content]))
    return out


def test_orphan_tool_use_gets_placeholder_tool_result(deepseek_provider):
    """Assistant tool_use with no following tool_result -> placeholder injected."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calc"},
                        {
                            "type": "tool_use",
                            "id": "toolu_z",
                            "name": "calc",
                            "input": {},
                        },
                    ],
                },
                {"role": "user", "content": "sigue"},
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    user_after = body["messages"][2]
    assert user_after["role"] == "user"
    assert isinstance(user_after["content"], list)
    first = user_after["content"][0]
    assert first["type"] == "tool_result"
    assert first["tool_use_id"] == "toolu_z"
    # The original user text survives after the synthesized result.
    assert any(
        b.get("type") == "text" and b.get("text") == "sigue"
        for b in user_after["content"]
    )


def test_orphan_tool_use_trailing_inserts_user_message(deepseek_provider):
    """Assistant tool_use as the final message -> a user message is appended."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_a",
                            "name": "calc",
                            "input": {},
                        },
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    last = body["messages"][-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "toolu_a"


def test_orphan_tool_result_reframed_as_text(deepseek_provider):
    """User tool_result with no preceding tool_use -> reframed as text (payload kept)."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_zzz",
                            "content": "42",
                        },
                        {"type": "text", "text": "y esto"},
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    last = body["messages"][-1]
    assert not any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in last["content"]
    )
    # The orphan result's payload survives as a text block.
    assert any(
        b.get("type") == "text" and b.get("text") == "42" for b in last["content"]
    )
    assert any(
        b.get("type") == "text" and b.get("text") == "y esto" for b in last["content"]
    )


def test_orphan_only_tool_result_message_keeps_text(deepseek_provider):
    """A user message of only-orphan tool_results survives as text (never empty)."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "ghost", "content": "x"},
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    for m in body["messages"]:
        assert m["content"] != []
    last = body["messages"][-1]
    assert last["content"][0]["type"] == "text"
    assert last["content"][0]["text"] == "x"


def test_well_paired_tools_are_unchanged(deepseek_provider):
    """Correctly paired tool_use/tool_result must not be touched (idempotent)."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_ok",
                            "name": "calc",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_ok",
                            "content": "7",
                        },
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert _roles_and_blocks(body) == [
        ("user", "str")
        if isinstance(body["messages"][0]["content"], str)
        else ("user", ["text"]),
        ("assistant", ["tool_use"]),
        ("user", ["tool_result"]),
    ]
    assert body["messages"][2]["content"][0]["tool_use_id"] == "toolu_ok"


def test_reconcile_logs_warning_when_repairing(deepseek_provider, caplog):
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_w",
                            "name": "calc",
                            "input": {},
                        },
                    ],
                },
                {"role": "user", "content": "sigue"},
            ],
        }
    )
    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)
    assert any(
        "reconciled orphaned tool pairs" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


_PNG_SOURCE = {"type": "base64", "media_type": "image/png", "data": "abc"}
_PDF_SOURCE = {"type": "base64", "media_type": "application/pdf", "data": "pdf"}


def _read_tool_exchange(model: str, tool_results: list[dict]) -> MessagesRequest:
    """Assistant Read tool_use(s) answered by the given tool_result blocks."""
    return MessagesRequest.model_validate(
        {
            "model": model,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tr["tool_use_id"],
                            "name": "Read",
                            "input": {"path": "f"},
                        }
                        for tr in tool_results
                    ],
                },
                {"role": "user", "content": tool_results},
            ],
        }
    )


def test_vision_model_keeps_top_level_image(deepseek_provider):
    """deepseek-v4-flash accepts images, so they are forwarded untouched."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image", "source": _PNG_SOURCE},
                    ],
                }
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text", "image"]
    assert content[1]["source"]["data"] == "abc"


def test_non_vision_deepseek_model_still_strips_image(deepseek_provider):
    """deepseek-v4-pro cannot see images: they are still stripped."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-pro",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image", "source": _PNG_SOURCE},
                    ],
                }
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert "image" not in [b["type"] for b in body["messages"][0]["content"]]


def test_vision_model_hoists_image_out_of_tool_result(deepseek_provider):
    """Read of a PNG: the image moves after the tool_result, which stays a string."""
    request = _read_tool_exchange(
        "deepseek-v4-flash",
        [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "image", "source": _PNG_SOURCE}],
            }
        ],
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][1]["content"]
    assert [b["type"] for b in content] == ["tool_result", "image"]
    assert isinstance(content[0]["content"], str)
    assert "image attached after this tool result" in content[0]["content"]
    assert "abc" not in content[0]["content"]
    assert content[1]["source"]["data"] == "abc"


def test_vision_model_keeps_tool_result_text_when_hoisting(deepseek_provider):
    request = _read_tool_exchange(
        "deepseek-v4-flash",
        [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {"type": "text", "text": "screenshot saved"},
                    {"type": "image", "source": _PNG_SOURCE},
                ],
            }
        ],
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][1]["content"]
    assert [b["type"] for b in content] == ["tool_result", "image"]
    assert content[0]["content"] == "screenshot saved"


def test_vision_model_still_strips_documents(deepseek_provider):
    """No DeepSeek model reads PDFs, even the vision-capable one."""
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize"},
                        {"type": "document", "source": _PDF_SOURCE},
                    ],
                }
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert [b["type"] for b in body["messages"][0]["content"]] == ["text"]


def test_hoist_placeholder_is_per_tool_result(deepseek_provider):
    """A document-only tool_result gets the omitted placeholder, not the hoist note,
    even when a sibling tool_result in the same message hoisted an image."""
    request = _read_tool_exchange(
        "deepseek-v4-flash",
        [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "image", "source": _PNG_SOURCE}],
            },
            {
                "type": "tool_result",
                "tool_use_id": "t2",
                "content": [{"type": "document", "source": _PDF_SOURCE}],
            },
        ],
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][1]["content"]
    assert [b["type"] for b in content] == ["tool_result", "tool_result", "image"]
    assert "image attached after this tool result" in content[0]["content"]
    assert "attachment omitted" in content[1]["content"]
