from typing import Dict, Any, Optional
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.schemas.reports import FinalAnalysisReport
from src.schemas.ideas import UserIdeaInput
from src.rag.prompts import SYSTEM_STRICT_EVIDENCE, REPORT_SYNTHESIS_PROMPT
from src.utils.logging import logger


class ReportAgent:
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes the complete explainable final report.
        Updates state with 'final_report'.
        """
        idea_input: UserIdeaInput = state["idea_input"]
        features = state["extracted_features"]
        retrieval = state.get("retrieval_result")
        pat_analyses = state.get("patent_analysis", [])
        pap_analyses = state.get("research_analysis", [])
        feat_comp = state.get("feature_comparison")
        novelty = state.get("novelty_assessment")
        gaps = state.get("gap_analysis")
        proc_time = state.get("processing_time", 0.0)

        all_citations = []
        if novelty and novelty.supporting_evidence:
            all_citations.extend(novelty.supporting_evidence)
        if gaps and gaps.evidence_citations:
            all_citations.extend(gaps.evidence_citations)
        unique_citations = list(set(all_citations))

        prompt = REPORT_SYNTHESIS_PROMPT.format(
            idea_title=features.title,
            novelty_level=novelty.novelty_level if novelty else "Unassessed",
            novelty_score=f"{novelty.novelty_score:.2f}" if novelty else "N/A",
            citations=", ".join(unique_citations) or "None",
            gaps="; ".join(gaps.unexplored_combinations[:2]) if gaps else "None"
        )

        try:
            summary = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_STRICT_EVIDENCE
            )
        except Exception as e:
            logger.warning(f"Report synthesis failed: {e}. Using deterministic summary.")
            summary = (
                f"Preliminary prior-art overlap assessment for '{features.title}' is rated {novelty.novelty_level if novelty else 'Moderate'} "
                f"(Prototype Heuristic Score: {novelty.novelty_score if novelty else 0.50:.2f}). "
                f"Retrieved prior art indicates established foundational methods in {', '.join(unique_citations[:3])}. "
                f"However, specific potential white-space signals were identified across patent and academic literature regarding "
                f"{'; '.join(gaps.unexplored_combinations[:1]) if gaps else 'potential unexplored system combinations'}."
            )

        limitations = [
            "Evaluated strictly against the local 40-document PBL-3 seed corpus (20 patents, 20 research papers).",
            "This analysis is an AI-assisted technical screening and does NOT constitute a legal patentability opinion.",
            "Absence of evidence in the local corpus indicates technical underrepresentation and does not establish global absence.",
            "Full prior art searching requires global multi-jurisdiction patent database queries."
        ]

        report = FinalAnalysisReport(
            idea=idea_input,
            extracted_features=features,
            mode="multi_agent_rag",
            retrieval=retrieval,
            patent_analysis=pat_analyses,
            research_analysis=pap_analyses,
            feature_comparison=feat_comp,
            novelty=novelty,
            gaps=gaps,
            final_summary=summary,
            limitations=limitations,
            confidence=novelty.confidence if novelty else 0.70,
            processing_time_seconds=proc_time
        )

        state["final_report"] = report
        return state
