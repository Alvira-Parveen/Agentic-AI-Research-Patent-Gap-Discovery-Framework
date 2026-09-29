from typing import Dict, Any, Optional
from src.analysis.gap_discovery import GapDiscoveryEngine
from src.schemas.analysis import GapDiscoveryResult
from src.utils.logging import logger


class GapAgent:
    def __init__(self, engine_or_llm: Optional[Any] = None):
        if isinstance(engine_or_llm, GapDiscoveryEngine):
            self.engine = engine_or_llm
        elif engine_or_llm is not None and hasattr(engine_or_llm, "generate"):
            self.engine = GapDiscoveryEngine(llm_provider=engine_or_llm)
        else:
            self.engine = GapDiscoveryEngine()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes research and patent gaps from literature analysis.
        Updates state with 'gap_analysis'.
        """
        features = state.get("extracted_features")
        patent_analyses = state.get("patent_analysis", [])
        paper_analyses = state.get("research_analysis", [])
        feature_comparison = state.get("feature_comparison")

        result: GapDiscoveryResult = self.engine.discover_gaps(
            extracted_features=features,
            patent_analyses=patent_analyses,
            paper_analyses=paper_analyses,
            feature_comparison=feature_comparison
        )

        logger.info(f"GapAgent identified {len(result.unexplored_combinations)} potential gaps and {len(result.underrepresented_features)} underrepresented features.")
        state["gap_analysis"] = result
        return state
