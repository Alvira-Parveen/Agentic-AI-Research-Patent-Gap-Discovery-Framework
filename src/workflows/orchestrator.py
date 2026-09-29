from typing import Union
from src.schemas.ideas import UserIdeaInput
from src.schemas.reports import FinalAnalysisReport, ComparativeAnalysisResult
from src.workflows.single_llm import SingleLLMWorkflow
from src.workflows.rag_workflow import StandardRAGWorkflow
from src.workflows.multi_agent_workflow import MultiAgentRAGWorkflow
from src.utils.logging import logger


class IdeaAnalysisOrchestrator:
    def __init__(self):
        self.single_llm = SingleLLMWorkflow()
        self.rag = StandardRAGWorkflow()
        self.multi_agent = MultiAgentRAGWorkflow()

    def analyze(self, idea: Union[str, UserIdeaInput], mode: str = "multi_agent_rag") -> FinalAnalysisReport:
        """
        Executes idea analysis in one of three experimental modes:
        - 'single_llm'
        - 'rag'
        - 'multi_agent_rag'
        """
        if isinstance(idea, str):
            idea_input = UserIdeaInput(idea=idea)
        else:
            idea_input = idea

        mode_clean = mode.lower().strip()
        logger.info(f"Executing analyze_idea with mode='{mode_clean}' for query: '{idea_input.idea[:60]}...'")

        if mode_clean in ["single_llm", "baseline", "mode_a"]:
            return self.single_llm.run(idea_input)
        elif mode_clean in ["rag", "llm_rag", "mode_b"]:
            return self.rag.run(idea_input)
        elif mode_clean in ["multi_agent_rag", "multi_agent", "mode_c"]:
            return self.multi_agent.run(idea_input)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from 'single_llm', 'rag', 'multi_agent_rag'.")

    def compare(self, idea: Union[str, UserIdeaInput]) -> ComparativeAnalysisResult:
        """
        Executes the same query through all three modes for comparative evaluation.
        """
        if isinstance(idea, str):
            idea_input = UserIdeaInput(idea=idea)
        else:
            idea_input = idea

        logger.info(f"Comparing all 3 modes on idea: '{idea_input.idea[:60]}...'")
        res_a = self.single_llm.run(idea_input)
        res_b = self.rag.run(idea_input)
        res_c = self.multi_agent.run(idea_input)

        nov_a_str = f"{res_a.novelty.novelty_score:.2f}" if (res_a.novelty and res_a.novelty.novelty_score is not None) else "Unassessed"
        nov_b_str = f"{res_b.novelty.novelty_score:.2f}" if (res_b.novelty and res_b.novelty.novelty_score is not None) else "Unassessed"
        nov_c_str = f"{res_c.novelty.novelty_score:.2f}" if (res_c.novelty and res_c.novelty.novelty_score is not None) else "Unassessed"

        summary = (
            f"Comparison completed. "
            f"Mode A (Single LLM): {res_a.processing_time_seconds:.2f}s, 0 citations, novelty={nov_a_str}. "
            f"Mode B (RAG): {res_b.processing_time_seconds:.2f}s, {len(res_b.retrieval.all_results) if res_b.retrieval else 0} retrieved chunks, novelty={nov_b_str}. "
            f"Mode C (Multi-Agent RAG): {res_c.processing_time_seconds:.2f}s, {len(res_c.patent_analysis)} patents + {len(res_c.research_analysis)} papers analyzed, "
            f"novelty={nov_c_str} ({res_c.novelty.novelty_level if res_c.novelty else 'N/A'}), {len(res_c.gaps.unexplored_combinations) if res_c.gaps else 0} gaps discovered."
        )

        metrics = {
            "processing_time": {
                "single_llm": res_a.processing_time_seconds,
                "rag": res_b.processing_time_seconds,
                "multi_agent_rag": res_c.processing_time_seconds
            },
            "evidence_count": {
                "single_llm": 0,
                "rag": len(res_b.retrieval.all_results) if res_b.retrieval else 0,
                "multi_agent_rag": len(res_c.novelty.supporting_evidence) if res_c.novelty else 0
            },
            "novelty_score": {
                "single_llm": res_a.novelty.novelty_score if res_a.novelty else 0.0,
                "rag": res_b.novelty.novelty_score if res_b.novelty else 0.0,
                "multi_agent_rag": res_c.novelty.novelty_score if res_c.novelty else 0.0
            },
            "gaps_count": {
                "single_llm": 0,
                "rag": 0,
                "multi_agent_rag": len(res_c.gaps.unexplored_combinations) if res_c.gaps else 0
            }
        }

        return ComparativeAnalysisResult(
            idea=idea_input,
            single_llm=res_a,
            rag=res_b,
            multi_agent_rag=res_c,
            comparison_summary=summary,
            metrics=metrics
        )


def analyze_idea(idea: str, mode: str = "multi_agent_rag") -> FinalAnalysisReport:
    """Convenience functional interface."""
    orchestrator = IdeaAnalysisOrchestrator()
    return orchestrator.analyze(idea, mode)
