import time
from typing import Optional
from src.schemas.ideas import UserIdeaInput
from src.schemas.reports import FinalAnalysisReport
from src.analysis.feature_extraction import FeatureExtractor
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.rag.prompts import SYSTEM_STRICT_EVIDENCE, SINGLE_LLM_PROMPT
from src.config.constants import LEGAL_DISCLAIMER, GLOBAL_RESEARCH_DISCLAIMER
from src.schemas.analysis import NoveltyAssessmentResult, NoveltyFactorBreakdown


class SingleLLMWorkflow:
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()
        self.feature_extractor = FeatureExtractor(self.llm)

    def run(self, idea_input: UserIdeaInput) -> FinalAnalysisReport:
        t0 = time.perf_counter()
        features = self.feature_extractor.extract(idea_input.idea, idea_input.domain)

        prompt = SINGLE_LLM_PROMPT.format(idea=idea_input.idea)
        summary = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_STRICT_EVIDENCE)

        elapsed = round(time.perf_counter() - t0, 4)

        # Baseline qualitative assessment without retrieved evidence (zero hardcoded numbers)
        novelty_baseline = NoveltyAssessmentResult(
            novelty_level="Unassessed (Parametric baseline)",
            novelty_score=None,
            heuristic_score=None,
            score_type="unassessed",
            confidence=0.0,
            factors=None,
            formula_explanation=(
                "Baseline Mode A operates strictly on internal LLM parametric weights without retrieval grounding. "
                "Numerical overlap scores are intentionally withheld to prevent predetermined baseline bias."
            ),
            supporting_evidence=[],
            distribution_type="INSUFFICIENT_EVIDENCE",
            limitations=[
                "Zero external documents retrieved from prior-art databases.",
                "Assessment relies exclusively on LLM internal training weights.",
                "High risk of hallucinations or ungrounded prior-art assumptions.",
                "Numerical novelty scores and factor breakdowns are unavailable for Mode A."
            ],
            legal_disclaimer=GLOBAL_RESEARCH_DISCLAIMER
        )

        return FinalAnalysisReport(
            idea=idea_input,
            extracted_features=features,
            mode="single_llm",
            retrieval=None,
            patent_analysis=[],
            research_analysis=[],
            feature_comparison=None,
            novelty=novelty_baseline,
            gaps=None,
            final_summary=summary,
            limitations=[
                "Single LLM baseline does not query prior-art databases.",
                "Real patent claims and research papers were not retrieved or examined.",
                "Citations to external literature are completely absent."
            ],
            confidence=0.0,
            processing_time_seconds=elapsed
        )
