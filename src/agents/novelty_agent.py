from typing import Dict, Any, Optional
from src.analysis.novelty_scoring import NoveltyScoringEngine
from src.schemas.analysis import NoveltyAssessmentResult
from src.utils.logging import logger


class NoveltyAgent:
    def __init__(self, scoring_engine: Optional[NoveltyScoringEngine] = None):
        self.engine = scoring_engine or NoveltyScoringEngine()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes transparent multi-factor novelty assessment.
        Updates state with 'novelty_assessment'.
        """
        retrieval_result = state.get("retrieval_result")
        patent_analyses = state.get("patent_analysis", [])
        paper_analyses = state.get("research_analysis", [])
        feature_comparison = state.get("feature_comparison")

        result: NoveltyAssessmentResult = self.engine.calculate_novelty(
            retrieval_result=retrieval_result,
            patent_analyses=patent_analyses,
            paper_analyses=paper_analyses,
            feature_comparison=feature_comparison
        )

        logger.info(f"NoveltyAgent assessed level='{result.novelty_level}' (Score: {result.novelty_score:.4f}, Conf: {result.confidence:.2f})")
        state["novelty_assessment"] = result
        return state
