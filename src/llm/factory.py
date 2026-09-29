from src.config.settings import settings
from src.llm.base import BaseLLMProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.mock_provider import MockProvider
from src.utils.logging import logger


def get_llm_provider(provider_name: str = None) -> BaseLLMProvider:
    """Factory function returning the configured LLM provider."""
    provider = (provider_name or settings.llm_provider).lower()
    if provider == "openai":
        if settings.openai_api_key:
            return OpenAIProvider()
        else:
            logger.warning("OpenAI provider selected but OPENAI_API_KEY is not set. Falling back to MockProvider.")
            return MockProvider()
    elif provider == "mock":
        return MockProvider()
    else:
        logger.warning(f"Unknown LLM provider '{provider}'. Falling back to MockProvider.")
        return MockProvider()
