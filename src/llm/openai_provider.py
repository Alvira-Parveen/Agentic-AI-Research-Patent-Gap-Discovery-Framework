from typing import Optional, Type, TypeVar
import json
from openai import OpenAI
from src.llm.base import BaseLLMProvider
from src.config.settings import settings
from src.utils.logging import logger

T = TypeVar("T")


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.openai_base_url

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set. OpenAIProvider will fail if invoked.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        if not self.client:
            raise RuntimeError("OpenAIProvider cannot make API calls because OPENAI_API_KEY is missing.")

        temp = temperature if temperature is not None else settings.temperature
        max_t = max_tokens if max_tokens is not None else settings.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
