"""
gestalt_extractor.llm
=====================
LLM Provider abstraction, live OpenAI client, deterministic heuristic mock provider,
and dynamic provider factory.
"""

import logging
from typing import Optional

from gestalt_extractor.config import ExtractorConfig, get_default_config
from gestalt_extractor.llm.base import (
    LLMProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    SchemaValidationError,
)
from gestalt_extractor.llm.openai_provider import OpenAIProvider
from gestalt_extractor.llm.mock_provider import DeterministicMockProvider, MockProvider

logger = logging.getLogger(__name__)


def get_provider(config: Optional[ExtractorConfig] = None) -> LLMProvider:
    """Instantiates and returns the configured LLM extraction provider.

    Inspects config flags (--mock, GESTALT_MOCK env var, or absence of OPENAI_API_KEY).
    If mock mode is active, returns DeterministicMockProvider.
    Otherwise, returns OpenAIProvider configured with a DeterministicMockProvider
    fallback instance to ensure zero-crash robustness under API rate-limits or timeouts.

    Args:
        config: Optional ExtractorConfig instance. If None, default config is loaded.

    Returns:
        Configured LLMProvider instance.
    """
    if config is None:
        config = get_default_config()

    if config.is_mock_enabled():
        logger.debug("Provider factory: selecting DeterministicMockProvider (mock mode enabled).")
        return DeterministicMockProvider()

    logger.debug("Provider factory: selecting OpenAIProvider with mock fallback.")
    mock_fallback = DeterministicMockProvider()
    return OpenAIProvider(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        fallback_provider=mock_fallback,
    )


__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "DeterministicMockProvider",
    "MockProvider",
    "get_provider",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "SchemaValidationError",
]
