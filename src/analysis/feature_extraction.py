import re
from typing import Optional, List
from src.schemas.ideas import ExtractedFeatures
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.rag.prompts import SYSTEM_STRICT_EVIDENCE, FEATURE_EXTRACTION_PROMPT
from src.utils.logging import logger


class FeatureExtractor:
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def extract(self, idea_text: str, domain: Optional[str] = None) -> ExtractedFeatures:
        """
        Extracts structured technical features from an idea description.
        Combines deterministic parsing with structured LLM extraction.
        """
        prompt = FEATURE_EXTRACTION_PROMPT.format(idea=idea_text)
        try:
            structured = self.llm.generate_structured(
                prompt=prompt,
                schema_cls=ExtractedFeatures,
                system_prompt=SYSTEM_STRICT_EVIDENCE
            )
            if domain:
                structured.domain = domain
            return structured
        except Exception as e:
            logger.warning(f"Structured feature extraction failed: {e}. Falling back to deterministic extraction.")
            return self._deterministic_extract(idea_text, domain)

    def _deterministic_extract(self, idea_text: str, domain: Optional[str] = None) -> ExtractedFeatures:
        words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', idea_text) if w.lower() not in [
            "with", "that", "this", "from", "using", "into", "over", "system", "method"
        ]]
        techs = words[:4] if words else ["Machine Learning", "Data Processing"]
        return ExtractedFeatures(
            title=f"Technical Proposal: {idea_text[:50]}...",
            description=idea_text,
            domain=domain or "Artificial Intelligence & Applied Systems",
            technologies=[t.capitalize() for t in techs],
            technical_features=[f"{t.capitalize()} architecture" for t in techs],
            methods=["Algorithmic classification", "Contextual embedding search", "Feature vector aggregation"],
            inputs=["Raw input data stream", "Sensor / image / text inputs"],
            outputs=["Detection results", "Automated analytical recommendations"],
            key_components=["Preprocessing Module", "Inference Subsystem", "Evaluation Pipeline"]
        )
