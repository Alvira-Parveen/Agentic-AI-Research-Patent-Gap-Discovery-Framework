import pytest
from src.schemas.ideas import UserIdeaInput, ExtractedFeatures
from src.workflows.orchestrator import IdeaAnalysisOrchestrator, analyze_idea
from src.analysis.feature_comparison import FeatureComparator
from src.analysis.gap_discovery import GapDiscoveryEngine
from src.schemas.analysis import FeatureComparisonResult, FeatureComparisonItem, PatentAnalysisItem, PaperAnalysisItem
from src.schemas.retrieval import RetrievedChunk
from src.schemas.documents import DocumentType


def test_single_llm_workflow_parametric_baseline():
    orchestrator = IdeaAnalysisOrchestrator()
    idea = UserIdeaInput(idea="AI system for predicting battery degradation in electric vehicles using recurrent neural networks.")
    res = orchestrator.analyze(idea, mode="single_llm")

    assert res.mode == "single_llm"
    assert res.idea.idea == idea.idea
    assert len(res.extracted_features.technical_features) > 0
    assert res.retrieval is None
    assert res.processing_time_seconds > 0.0

    # Verify research validity: Mode A must not fabricate novelty scores or factors
    assert res.novelty.novelty_score is None
    assert res.novelty.confidence == 0.0
    assert res.novelty.factors is None
    assert res.novelty.novelty_level == "Unassessed (Parametric baseline)"
    assert res.novelty.supporting_evidence == []
    assert "legal patentability opinion" in res.novelty.legal_disclaimer.lower()
    assert any("Mode A" in lim or "internal training" in lim for lim in res.novelty.limitations)


def test_rag_workflow_dynamic_novelty():
    orchestrator = IdeaAnalysisOrchestrator()
    idea = UserIdeaInput(idea="AI system using drone multispectral imagery for crop disease classification.")
    res = orchestrator.analyze(idea, mode="rag")

    assert res.mode == "rag"
    assert res.retrieval is not None
    assert len(res.retrieval.all_results) > 0
    assert res.novelty is not None
    assert res.novelty.novelty_score is not None
    # Verify score is dynamically computed from retrieved chunk similarity scores
    top_score = res.retrieval.all_results[0].similarity_score
    expected_score = round(max(0.0, min(1.0, 1.0 - top_score)), 4)
    assert res.novelty.novelty_score == expected_score


def test_feature_comparator_cosine_similarity_classes():
    comparator = FeatureComparator()
    chunk = RetrievedChunk(
        chunk_id="C1",
        document_id="PAT-001",
        document_type=DocumentType.PATENT,
        document_title="Agricultural Multispectral Drone",
        section="claim",
        text="A drone system configured for multispectral imaging of crop fields.",
        similarity_score=0.85,
        rank=1,
        citation="[PAT-001, Claim 1]"
    )

    # Identical / highly similar text -> MATCH or PARTIAL_MATCH
    match_type, sim = comparator.assess_feature_presence(
        "multispectral imaging for crop fields", [chunk]
    )
    assert match_type in ["MATCH", "PARTIAL_MATCH"]
    assert sim > 0.38

    # Completely unrelated text -> NO_MATCH or UNCERTAIN
    match_type_unrelated, sim_unrelated = comparator.assess_feature_presence(
        "quantum cryogenic refrigeration superconducting qubits", [chunk]
    )
    assert match_type_unrelated in ["NO_MATCH", "UNCERTAIN"]
    assert sim_unrelated < 0.38


def test_gap_discovery_no_artificial_coupling_and_insufficient_evidence():
    engine = GapDiscoveryEngine()

    # Case 1: When no candidate prior-art is retrieved, return insufficient_evidence
    ext_feats = ExtractedFeatures(title="Idea", description="Desc", technical_features=["feature_one", "feature_two"])
    feat_comp_empty = FeatureComparisonResult(
        idea_features=["feature_one", "feature_two"],
        comparison_matrix=[],
        common_elements=[],
        underrepresented_elements=["feature_one", "feature_two"],
        cross_domain_insights="No matches."
    )

    result_empty = engine._deterministic_gap_discovery(
        ext_feats, [], [], feat_comp_empty
    )
    assert len(result_empty.unexplored_combinations) == 0
    assert result_empty.gap_type == "insufficient_evidence"
    assert "Insufficient evidence" in result_empty.patent_vs_paper_differences

    # Case 2: When an underrepresented feature exists, gaps are derived strictly from that feature
    # and MUST NOT contain artificial canned pairing
    feat_comp_gap = FeatureComparisonResult(
        idea_features=["feature_alpha", "unexplored_telemetry"],
        comparison_matrix=[
            FeatureComparisonItem(
                feature="feature_alpha",
                user_feature="feature_alpha",
                presence_in_idea=True,
                patent_coverage="Fully Covered",
                paper_coverage="Uncovered",
                matched_prior_art=["PAT-001"],
                gap_status="Partially Covered",
                match_type="MATCH",
                similarity_or_match_score=0.9
            ),
            FeatureComparisonItem(
                feature="unexplored_telemetry",
                user_feature="unexplored_telemetry",
                presence_in_idea=True,
                patent_coverage="Uncovered",
                paper_coverage="Uncovered",
                matched_prior_art=[],
                gap_status="Potential Gap",
                match_type="NO_MATCH",
                similarity_or_match_score=0.1
            )
        ],
        common_elements=["feature_alpha"],
        underrepresented_elements=["unexplored_telemetry"],
        cross_domain_insights="Underrepresented telemetry."
    )

    pat_analysis = [
        PatentAnalysisItem(
            patent_id="PAT-001", patent_title="Alpha Patent", similarity_score=0.9,
            technical_problem="", proposed_solution="Alpha mechanism",
            matching_features=["feature_alpha"], different_features=[],
            claim_relevance="", evidence=["[PAT-001]"], analysis=""
        )
    ]

    ext_feats_gap = ExtractedFeatures(title="Novel Idea", description="Desc", technical_features=["feature_alpha", "unexplored_telemetry"])
    result_gap = engine._deterministic_gap_discovery(
        ext_feats_gap, pat_analysis, [], feat_comp_gap
    )
    assert len(result_gap.unexplored_combinations) >= 1
    combo_text = result_gap.unexplored_combinations[0]
    assert "unexplored_telemetry" in combo_text
    assert "feature_alpha + unexplored_telemetry" not in combo_text


def test_compare_all_modes():
    orchestrator = IdeaAnalysisOrchestrator()
    idea = "AI system using drone multispectral imagery for crop disease classification and treatment prescription."
    comp = orchestrator.compare(idea)

    assert comp.single_llm.mode == "single_llm"
    assert comp.rag.mode == "rag"
    assert comp.multi_agent_rag.mode == "multi_agent_rag"
    assert "metrics" in comp.model_dump()
    assert comp.metrics["evidence_count"]["single_llm"] == 0
    # Mode A novelty score is None
    assert comp.single_llm.novelty.novelty_score is None
