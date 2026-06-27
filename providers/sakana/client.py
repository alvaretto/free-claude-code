"""Sakana AI (Fugu) provider implementation.

Sakana Fugu exposes an OpenAI-compatible ``/v1/chat/completions`` API
(https://api.sakana.ai/v1), so this provider rides the shared
:class:`OpenAIChatTransport` exactly like Fireworks. Models: ``fugu`` and
``fugu-ultra``. The API is multimodal (text + image) upstream, but THIS proxy's
OpenAI-chat conversion path does not emit image blocks yet, so do NOT point
``VISION_MODEL`` at it without extending ``core.anthropic.conversion``.
"""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body

SAKANA_BASE_URL = "https://api.sakana.ai/v1"


class SakanaProvider(OpenAIChatTransport):
    """Sakana Fugu provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="SAKANA",
            base_url=config.base_url or SAKANA_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build request body for Sakana Fugu."""
        if thinking_enabled is None:
            thinking_enabled = self._is_thinking_enabled(request)
        return build_request_body(
            request,
            thinking_enabled=thinking_enabled,
        )
