import time
from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, START, END

from src.schemas.ideas import UserIdeaInput, ExtractedFeatures
from src.schemas.retrieval import RetrievalResult, RetrievedChunk
from src.schemas.analysis import (
    PatentAnalysisItem,
    PaperAnalysisItem,
    FeatureComparisonResult,
    NoveltyAssessmentResult,
    GapDiscoveryResult,
)
from src.schemas.reports import FinalAnalysisReport
from src.analysis.feature_extraction import FeatureExtractor
from src.analysis.feature_comparison import FeatureComparator
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.patent_agent import PatentAnalysisAgent
from src.agents.research_agent import ResearchAnalysisAgent
from src.agents.novelty_agent import NoveltyAgent
from src.agents.gap_agent import GapAgent
from src.agents.report_agent import ReportAgent
from src.llm.factory import get_llm_provider
from src.utils.logging import logger


class MultiAgentState(TypedDict, total=False):
    user_idea: str
    idea_input: UserIdeaInput
    extracted_features: ExtractedFeatures
    retrieval_result: RetrievalResult
    retrieved_patents: List[RetrievedChunk]
    retrieved_papers: List[RetrievedChunk]
    patent_analysis: List[PatentAnalysisItem]
    research_analysis: List[PaperAnalysisItem]
    feature_comparison: FeatureComparisonResult
    novelty_assessment: NoveltyAssessmentResult
    gap_analysis: GapDiscoveryResult
    final_report: FinalAnalysisReport
    processing_time: float


class MultiAgentRAGWorkflow:
    def __init__(self):
        llm = get_llm_provider()
        self.feature_extractor = FeatureExtractor(llm)
        self.retrieval_agent = RetrievalAgent()
        self.patent_agent = PatentAnalysisAgent(llm)
        self.research_agent = ResearchAnalysisAgent(llm)
        self.feature_comparator = FeatureComparator()
        self.novelty_agent = NoveltyAgent()
        self.gap_agent = GapAgent(llm)
        self.report_agent = ReportAgent(llm)

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(MultiAgentState)

        # Node 1: Feature Extraction
        def extract_features_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Feature Extraction")
            idea_input = state["idea_input"]
            features = self.feature_extractor.extract(idea_input.idea, idea_input.domain)
            return {"extracted_features": features}

        # Node 2: Retrieval Agent
        def retrieval_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Retrieval Agent")
            return self.retrieval_agent.run(state)

        # Node 3: Patent Analysis Agent
        def patent_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Patent Analysis Agent")
            return self.patent_agent.run(state)

        # Node 4: Research Analysis Agent
        def research_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Research Analysis Agent")
            return self.research_agent.run(state)

        # Node 5: Feature Comparison
        def feature_comparison_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Feature Comparison")
            feats = state["extracted_features"].technical_features
            pat_analyses = state.get("patent_analysis", [])
            pap_analyses = state.get("research_analysis", [])
            retrieval_res = state.get("retrieval_result")
            candidate_chunks = retrieval_res.all_results if retrieval_res else None
            comp = self.feature_comparator.compare(
                feats, pat_analyses, pap_analyses, candidate_chunks=candidate_chunks
            )
            return {"feature_comparison": comp}

        # Node 6: Novelty Agent
        def novelty_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Novelty Assessment Agent")
            return self.novelty_agent.run(state)

        # Node 7: Gap Discovery Agent
        def gap_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Gap Discovery Agent")
            return self.gap_agent.run(state)

        # Node 8: Report Agent
        def report_step(state: MultiAgentState) -> MultiAgentState:
            logger.info("LangGraph Node: Report Agent")
            return self.report_agent.run(state)

        # Register nodes
        workflow.add_node("feature_extraction", extract_features_step)
        workflow.add_node("retrieval", retrieval_step)
        workflow.add_node("patent_analysis", patent_step)
        workflow.add_node("research_analysis", research_step)
        workflow.add_node("feature_comparison", feature_comparison_step)
        workflow.add_node("novelty_assessment", novelty_step)
        workflow.add_node("gap_discovery", gap_step)
        workflow.add_node("report_synthesis", report_step)

        # Connect sequential multi-agent execution pipeline
        workflow.add_edge(START, "feature_extraction")
        workflow.add_edge("feature_extraction", "retrieval")
        workflow.add_edge("retrieval", "patent_analysis")
        workflow.add_edge("patent_analysis", "research_analysis")
        workflow.add_edge("research_analysis", "feature_comparison")
        workflow.add_edge("feature_comparison", "novelty_assessment")
        workflow.add_edge("novelty_assessment", "gap_discovery")
        workflow.add_edge("gap_discovery", "report_synthesis")
        workflow.add_edge("report_synthesis", END)

        return workflow.compile()

    def run(self, idea_input: UserIdeaInput) -> FinalAnalysisReport:
        t0 = time.perf_counter()
        init_state: MultiAgentState = {
            "user_idea": idea_input.idea,
            "idea_input": idea_input,
            "processing_time": 0.0
        }

        final_state = self.graph.invoke(init_state)
        elapsed = round(time.perf_counter() - t0, 4)

        report = final_state.get("final_report")
        if report:
            report.processing_time_seconds = elapsed
            return report

        raise RuntimeError("MultiAgent workflow failed to produce a final report.")
