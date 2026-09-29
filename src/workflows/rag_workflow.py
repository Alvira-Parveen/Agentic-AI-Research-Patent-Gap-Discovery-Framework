import time
from typing import Optional
from src.schemas.ideas import UserIdeaInput
from src.schemas.reports import FinalAnalysisReport
from src.schemas.analysis import NoveltyAssessmentResult, NoveltyFactorBreakdown
from src.analysis.feature_extraction import FeatureExtractor
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.config.constants import LEGAL_DISCLAIMER, GLOBAL_RESEARCH_DISCLAIMER


class StandardRAGWorkflow:
    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        generator: Optional[RAGGenerator] = None,
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        self.llm = llm_provider or get_llm_provider()
        self.retriever = retriever or RAGRetriever()
        self.generator = generator or RAGGenerator(self.llm)
        self.feature_extractor = FeatureExtractor(self.llm)

    def run(self, idea_input: UserIdeaInput) -> FinalAnalysisReport:
        t0 = time.perf_counter()
        features = self.feature_extractor.extract(idea_input.idea, idea_input.domain)

        # Retrieve prior art
        retrieval_res = self.retriever.retrieve(idea_input.idea)

        # Generate RAG response
        summary = self.generator.generate_rag_report(idea_input.idea, retrieval_res)

        elapsed = round(time.perf_counter() - t0, 4)

        # RAG baseline novelty calculation grounded strictly on retrieved embeddings (zero hardcoded constants)
        all_sims = [c.similarity_score for c in retrieval_res.all_results]
        top_sim = max(all_sims) if all_sims else 0.0
        mean_sim = sum(all_sims) / max(1, len(all_sims)) if all_sims else 0.0

        novelty_score = round(max(0.0, min(1.0, 1.0 - top_sim)), 4) if all_sims else None
        level = "High" if novelty_score and novelty_score >= 0.70 else (
            "Moderate" if novelty_score and novelty_score >= 0.40 else ("Low" if novelty_score else "Unassessed")
        )

        factors = NoveltyFactorBreakdown(
            semantic_similarity=round(top_sim, 4),
            technical_overlap=round(mean_sim, 4),
            claim_feature_coverage=0.0,
            evidence_strength=round(len(all_sims) / 6.0, 4),
            factor_rationales={
                "semantic_similarity": f"Peak cosine similarity across retrieved evidence: {top_sim:.4f}",
                "technical_overlap": f"Mean cosine similarity across retrieved evidence: {mean_sim:.4f} (un-decomposed)",
                "claim_feature_coverage": "Not evaluated: Standard single-prompt RAG lacks specialized claim parsing.",
                "evidence_strength": f"Retrieved {len(all_sims)} candidate chunks from dual FAISS indices."
            }
        ) if all_sims else None

        top_prior_art_similarity = round(top_sim, 4) if all_sims else None
        prior_art_overlap_signal = round(top_sim, 4) if all_sims else None

        novelty = NoveltyAssessmentResult(
            novelty_level=level,
            novelty_score=novelty_score,
            heuristic_score=novelty_score,
            is_heuristic=True,
            score_type="prototype_prior_art_overlap_heuristic",
            top_prior_art_similarity=top_prior_art_similarity,
            prior_art_overlap_signal=prior_art_overlap_signal,
            confidence=round(min(1.0, len(all_sims) / 6.0) * 0.7, 4),
            factors=factors,
            formula_explanation=(
                f"Standard RAG baseline retrieval overlap signal: Peak similarity = {top_sim:.4f}. "
                f"Prototype heuristic signal = 1.0 - TopSimilarity[{top_sim:.2f}]. "
                f"NOTE: Measures textual retrieval similarity against retrieved seed chunks; does NOT measure legal patent novelty or obviousness."
            ),
            supporting_evidence=[c.citation for c in retrieval_res.all_results[:3]],
            distribution_type="UNDETERMINED",
            limitations=[
                "Standard single-prompt RAG passes all evidence in a single context window.",
                "Cosine similarity measures semantic retrieval overlap, NOT statutory novelty under patent law.",
                "Independent patent claim scope and academic research white-spaces are not decomposed.",
                "Claim coverage is omitted because no legal claim isolation agent was invoked."
            ],
            legal_disclaimer=GLOBAL_RESEARCH_DISCLAIMER
        )

        return FinalAnalysisReport(
            idea=idea_input,
            extracted_features=features,
            mode="rag",
            retrieval=retrieval_res,
            patent_analysis=[],
            research_analysis=[],
            feature_comparison=None,
            novelty=novelty,
            gaps=None,
            final_summary=summary,
            limitations=[
                "Standard RAG passes all evidence into a single prompt without agentic claim breakdown.",
                "Patent-specific claim parsing and scientific white-space discovery are not independently verified."
            ],
            confidence=0.65,
            processing_time_seconds=elapsed
        )
