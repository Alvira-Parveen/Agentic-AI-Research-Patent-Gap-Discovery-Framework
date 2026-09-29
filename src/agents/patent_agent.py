from typing import Dict, Any, List, Optional
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.schemas.analysis import PatentAnalysisItem
from src.schemas.retrieval import RetrievedChunk
from src.rag.prompts import SYSTEM_STRICT_EVIDENCE, PATENT_ANALYSIS_PROMPT
from src.utils.logging import logger


class PatentAnalysisAgent:
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes each retrieved patent chunk against the user idea.
        Updates state with 'patent_analysis'.
        """
        retrieved_patents: List[RetrievedChunk] = state.get("retrieved_patents", [])
        features = state.get("extracted_features")
        features_str = ", ".join(features.technical_features) if features else state.get("user_idea", "")

        analyses: List[PatentAnalysisItem] = []
        # Group chunks by document to avoid redundant evaluations
        processed_docs = set()

        for chunk in retrieved_patents:
            if chunk.document_id in processed_docs:
                continue
            processed_docs.add(chunk.document_id)

            prompt = PATENT_ANALYSIS_PROMPT.format(
                idea_features=features_str,
                doc_id=chunk.document_id,
                title=chunk.document_title,
                similarity_score=chunk.similarity_score,
                citation=chunk.citation,
                excerpt=chunk.text[:1200]
            )

            try:
                item = self.llm.generate_structured(
                    prompt=prompt,
                    schema_cls=PatentAnalysisItem,
                    system_prompt=SYSTEM_STRICT_EVIDENCE
                )
                # Enforce that deterministic similarity score is preserved
                item.similarity_score = chunk.similarity_score
                if chunk.citation not in item.evidence:
                    item.evidence.append(chunk.citation)
                analyses.append(item)
            except Exception as e:
                logger.warning(f"Patent analysis failed for {chunk.document_id}: {e}. Using deterministic fallback.")
                fallback = PatentAnalysisItem(
                    patent_id=chunk.document_id,
                    patent_title=chunk.document_title,
                    similarity_score=chunk.similarity_score,
                    technical_problem="Prior art discloses methods related to the query domain.",
                    proposed_solution=f"Technical implementation detailed in {chunk.citation}.",
                    matching_features=[f"Matching domain component from {chunk.document_id}"],
                    different_features=["Specific architecture proposed in user idea"],
                    claim_relevance=f"Relevant prior-art disclosure cited in {chunk.citation}.",
                    evidence=[chunk.citation],
                    analysis=f"The disclosure in {chunk.document_title} shares underlying concepts with the proposed idea."
                )
                analyses.append(fallback)

        state["patent_analysis"] = analyses
        return state
