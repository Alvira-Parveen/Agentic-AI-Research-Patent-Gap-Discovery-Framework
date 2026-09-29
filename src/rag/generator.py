from typing import Optional
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.rag.context_builder import build_rag_context
from src.rag.prompts import SYSTEM_STRICT_EVIDENCE, RAG_GENERATION_PROMPT
from src.schemas.retrieval import RetrievalResult


class RAGGenerator:
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def generate_rag_report(self, idea_text: str, retrieval_result: RetrievalResult) -> str:
        context_str = build_rag_context(retrieval_result.all_results)
        prompt = RAG_GENERATION_PROMPT.format(idea=idea_text, context=context_str)
        return self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_STRICT_EVIDENCE
        )
