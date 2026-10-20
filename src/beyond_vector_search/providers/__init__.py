"""Provider construction. SDK imports happen inside each backend, not here."""

from __future__ import annotations

from beyond_vector_search.config import Settings
from beyond_vector_search.providers.base import Provider
from beyond_vector_search.providers.claude import ClaudeProvider, MissingExtraError
from beyond_vector_search.providers.openai_api import OpenAIProvider

__all__ = ["MissingExtraError", "Provider", "make_provider"]


def make_provider(settings: Settings) -> Provider:
    if settings.provider == "openai":
        return OpenAIProvider(settings.openai_model, max_tokens=settings.max_tokens)
    return ClaudeProvider(settings.claude_model, max_tokens=settings.max_tokens)
