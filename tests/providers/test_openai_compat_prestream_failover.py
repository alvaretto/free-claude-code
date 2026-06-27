"""OpenAI-compat transport honours ``raise_on_prestream_error``.

A pre-stream failure (the upstream stream never opens — e.g. exhausted 429) must
raise :class:`PreStreamProviderError` WITHOUT emitting ``message_start`` first, so
``_stream_with_failover`` can transparently retry on the tier fallback model.

Legacy callers (flag unset) keep the eager ``message_start`` + in-stream error
envelope, byte-for-byte. The success path defers ``message_start`` until the
upstream stream opens and still relays it before content.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.exceptions import PreStreamProviderError
from providers.nvidia_nim import NvidiaNimProvider
from providers.rate_limit import GlobalRateLimiter
from tests.providers.test_nvidia_nim import MockRequest


def _config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=100,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )


def _rate_limit_429() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limited",
        response=Response(429, request=Request("POST", "http://x")),
        body={},
    )


def _content_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(delta=MagicMock(content=text, reasoning_content=""), finish_reason=None)
    ]
    chunk.usage = None
    return chunk


@pytest.mark.asyncio
async def test_prestream_failover_raises_on_exhausted_429_when_flag_set():
    """Exhausted 429 + raise_on_prestream_error=True → the FIRST event pull raises
    PreStreamProviderError (no message_start emitted), so the turn is safe to fail
    over onto the tier fallback model."""
    GlobalRateLimiter.reset_instance()
    try:
        provider = NvidiaNimProvider(_config(), nim_settings=NimSettings())
        req = MockRequest()
        stream = provider.stream_response(req, raise_on_prestream_error=True)

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=_rate_limit_429(),
            ) as mock_create,
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(PreStreamProviderError),
        ):
            # Must RAISE on the first pull instead of yielding message_start.
            await anext(stream)

        # Retries exhausted (1 + 3) before surfacing the pre-stream failure.
        assert mock_create.await_count == 4
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.asyncio
async def test_legacy_emits_in_stream_error_on_exhausted_429():
    """Default flag (False) preserves the in-stream error envelope: message_start
    is emitted eagerly and the turn ends with a well-formed error stream."""
    GlobalRateLimiter.reset_instance()
    try:
        provider = NvidiaNimProvider(_config(), nim_settings=NimSettings())
        req = MockRequest()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=_rate_limit_429(),
            ) as mock_create,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            events = [e async for e in provider.stream_response(req)]

        assert mock_create.await_count == 4
        blob = "".join(events)
        assert "message_start" in blob
        assert "rate limit" in blob.lower()
        assert "message_stop" in blob
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.asyncio
async def test_message_start_deferred_then_streams_on_success_when_flag_set():
    """With the flag set, message_start is deferred until the stream opens, then
    relayed before content (ordering preserved)."""
    GlobalRateLimiter.reset_instance()
    try:
        provider = NvidiaNimProvider(_config(), nim_settings=NimSettings())
        req = MockRequest()

        async def ok_stream():
            yield _content_chunk("Hi")

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=ok_stream(),
        ) as mock_create:
            events = [
                e
                async for e in provider.stream_response(
                    req, raise_on_prestream_error=True
                )
            ]

        assert mock_create.await_count == 1
        blob = "".join(events)
        assert "message_start" in blob
        assert "Hi" in blob
        # message_start must precede the streamed content.
        assert blob.index("message_start") < blob.index("Hi")
    finally:
        GlobalRateLimiter.reset_instance()
