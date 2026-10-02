"""Operator-configured additions to the system prompt of routed requests."""

from dataclasses import replace

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import SystemContent
from free_claude_code.core.reasoning import ReasoningControl

from .routing import RoutedMessagesRequest


def routing_banner_line(model: str, *, thinking: bool) -> str:
    """Literal first line the model is asked to print."""
    return f"🤖 Modelo: {model} | Thinking: {'ON' if thinking else 'OFF'}"


def _banner_directive(line: str) -> str:
    return (
        "INSTRUCCIÓN OBLIGATORIA DE BANNER DE ROUTING: Antes de cualquier otro "
        "contenido, tu respuesta DEBE comenzar EXACTAMENTE con la siguiente línea "
        "literal (incluyendo el emoji, sin bloque de código, sin variaciones, sin "
        f"traducir):\n\n{line}\n\n"
        "Después deja una línea en blanco y continúa con tu respuesta normal. Esta "
        "regla es OBLIGATORIA en TODAS las respuestas, incluidas las muy cortas "
        "(sí, no, OK). NO modifiques el texto del banner."
    )


def apply_prompt_extras(
    routed: RoutedMessagesRequest, settings: Settings
) -> RoutedMessagesRequest:
    """Append ``EXTRA_SYSTEM_PROMPT`` and the routing banner directive.

    Runs after routing so the banner names the model that will actually answer
    (including vision overrides) and the effective reasoning control. Fallback
    targets keep the primary's banner.
    """
    extras: list[str] = []
    if settings.extra_system_prompt is not None:
        extras.append(settings.extra_system_prompt)
    if settings.routing_banner:
        line = routing_banner_line(
            routed.resolved.primary.provider_model,
            thinking=routed.reasoning.control is ReasoningControl.ON,
        )
        extras.append(_banner_directive(line))
    if not extras:
        return routed

    system = routed.request.system
    if system is None or system == "":
        existing: list[SystemContent] = []
    elif isinstance(system, str):
        existing = [SystemContent(type="text", text=system)]
    else:
        existing = list(system)
    appended = [SystemContent(type="text", text=text) for text in extras]
    request = routed.request.model_copy(update={"system": [*existing, *appended]})
    return replace(routed, request=request)
