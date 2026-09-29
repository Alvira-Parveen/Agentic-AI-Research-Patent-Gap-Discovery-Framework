import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from src.utils.logging import logger

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """Generates raw text response from the LLM."""
        pass

    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> T:
        """Generates structured response validated against a Pydantic schema."""
        raw_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True
        )
        
        # Clean JSON markdown fences
        cleaned = self._clean_json_string(raw_text)
        try:
            parsed = json.loads(cleaned)
            return schema_cls.model_validate(parsed)
        except Exception as e:
            logger.error(f"Failed to validate response against {schema_cls.__name__}: {e}\nRaw output: {raw_text[:300]}")
            # Attempt relaxed parsing or fallback default
            raise ValueError(f"LLM output validation error: {e}")

    @staticmethod
    def _clean_json_string(text: str) -> str:
        """Extracts JSON substring if wrapped in markdown codeblocks or text."""
        text = text.strip()
        # Remove ```json ... ```
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()
        # Find first { or [ to last } or ]
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace+1]
        return text
