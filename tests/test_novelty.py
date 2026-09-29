import pytest
from src.analysis.novelty_scoring import NoveltyScoringEngine
from src.schemas.retrieval import RetrievalResult, RetrievedChunk
from src.schemas.analysis import (
    PatentAnalysisItem,
    PaperAnalysisItem,
    FeatureComparisonResult,
    FeatureComparisonItem,
)
from src.schemas.documents import DocumentType


def _create_mock_data():
    chunks = [
        RetrievedChunk(
            chunk_id="C1", document_id="PAT-001", document_type=DocumentType.PATENT,
            document_title="Drone Crop Sensing", section="claim", text="Drone multispectral imaging",
            similarity_score=0.80, rank=1, citation="[PAT-001, Claim 1]"
        ),
        RetrievedChunk(
            chunk_id="C2", document_id="PAP-001", document_type=DocumentType.PAPER,
            document_title="Crop Classification", section="abstract", text="Deep learning for disease detection",
            similarity_score=0.70, rank=2, citation="[PAP-001, Abstract]"
        )
    ]
    ret_res = RetrievalResult(
        query="Drone crop sensing", patents=[chunks[0]], papers=[chunks[1]], all_results=chunks
    )

    pat_analyses = [
        PatentAnalysisItem(
            patent_id="PAT-001", patent_title="Drone Crop Sensing", similarity_score=0.80,
            technical_problem="Crop disease", proposed_solution="Multispectral drone",
            matching_features=["Multispectral imaging"], different_features=["Prescription maps"],
            claim_relevance="Claim 1 discloses drone imaging.", evidence=["[PAT-001, Claim 1]"],
            analysis="High overlap on imaging."
        )
    ]

    pap_analyses = [
        PaperAnalysisItem(
            paper_id="PAP-001", paper_title="Crop Classification", similarity_score=0.70,
            problem_addressed="Fungal detection", methodology="CNN classification",
            findings="High accuracy", limitations="Small dataset",
            overlap_with_idea=["CNN classification"], differences=["No drone flight"],
            evidence=["[PAP-001, Abstract]"], analysis="Scientific precedent."
        )
    ]

    feat_comp = FeatureComparisonResult(
        idea_features=["Multispectral imaging", "CNN classification", "Variable rate prescription"],
        comparison_matrix=[
            FeatureComparisonItem(feature="Multispectral imaging", presence_in_idea=True, patent_coverage="Fully Covered", paper_coverage="Uncovered", gap_status="Well Covered"),
            FeatureComparisonItem(feature="CNN classification", presence_in_idea=True, patent_coverage="Uncovered", paper_coverage="Fully Covered", gap_status="Well Covered"),
            FeatureComparisonItem(feature="Variable rate prescription", presence_in_idea=True, patent_coverage="Uncovered", paper_coverage="Uncovered", gap_status="Potential Gap"),
        ],
        common_elements=["Multispectral imaging", "CNN classification"],
        underrepresented_elements=["Variable rate prescription"],
        cross_domain_insights="2 covered, 1 gap."
    )

    return ret_res, pat_analyses, pap_analyses, feat_comp


def test_novelty_scoring_formula():
    engine = NoveltyScoringEngine(weight_semantic=0.40, weight_overlap=0.35, weight_claim=0.25)
    ret_res, pat_analyses, pap_analyses, feat_comp = _create_mock_data()

    result = engine.calculate_novelty(ret_res, pat_analyses, pap_analyses, feat_comp)

    assert 0.0 <= result.novelty_score <= 1.0
    assert result.novelty_level in ["Low", "Moderate", "High"]
    assert 0.0 <= result.confidence <= 1.0
    assert "legal patentability opinion" in result.legal_disclaimer.lower()
    assert "POI" in result.formula_explanation
    assert result.is_heuristic is True
    assert "Prototype Multi-Factor Prior-Art Overlap Heuristic" in result.formula_explanation


def test_novelty_configurable_weights_and_rationales():
    engine = NoveltyScoringEngine(weight_semantic=0.60, weight_overlap=0.20, weight_claim=0.20)
    assert engine.w_semantic == 0.60
    assert engine.w_overlap == 0.20
    assert engine.w_claim == 0.20

    ret_res, pat_analyses, pap_analyses, feat_comp = _create_mock_data()
    result = engine.calculate_novelty(ret_res, pat_analyses, pap_analyses, feat_comp)

    assert result.factors is not None
    assert len(result.factors.factor_rationales) >= 3
    assert result.distribution_type in ["single_reference_dominated", "distributed_across_corpus", "INSUFFICIENT_EVIDENCE"]
    assert result.heuristic_score == result.novelty_score


def test_claim_evidence_scoring_levels():
    engine = NoveltyScoringEngine()

    # Level 1: Explicit claim section AND matching features -> 0.90
    claim_chunk = RetrievedChunk(
        chunk_id="C1", document_id="PAT-1", document_type=DocumentType.PATENT,
        document_title="Title", section="claim", text="A sensor system...",
        similarity_score=0.8, rank=1, citation="[PAT-1]"
    )
    score_lvl1, rat_lvl1 = engine._compute_claim_evidence_level([claim_chunk], ["feature_a"])
    assert score_lvl1 == 0.90
    assert "claim section" in rat_lvl1.lower()

    # Level 2: Non-claim section BUT matching features -> 0.70
    desc_chunk = RetrievedChunk(
        chunk_id="C2", document_id="PAT-2", document_type=DocumentType.PATENT,
        document_title="Title", section="description", text="Description...",
        similarity_score=0.8, rank=1, citation="[PAT-2]"
    )
    score_lvl2, rat_lvl2 = engine._compute_claim_evidence_level([desc_chunk], ["feature_a"])
    assert score_lvl2 == 0.70
    assert "specification disclosure" in rat_lvl2.lower()

    # Level 3: Claim section retrieved BUT no identified matching features -> 0.25
    score_lvl3, rat_lvl3 = engine._compute_claim_evidence_level([claim_chunk], [])
    assert score_lvl3 == 0.25

    # Level 4: No claims and no matches -> 0.0
    score_lvl4, rat_lvl4 = engine._compute_claim_evidence_level([], [])
    assert score_lvl4 == 0.0


def test_reference_distribution_detection():
    engine = NoveltyScoringEngine()

    # Single patent dominates (has 3 matching features out of 3 -> ratio 1.0 >= 0.70)
    single_dom_pats = [
        PatentAnalysisItem(
            patent_id="PAT-001", patent_title="P1", similarity_score=0.9,
            technical_problem="Prob", proposed_solution="Sol",
            matching_features=["feat1", "feat2", "feat3"], different_features=[],
            claim_relevance="Claims feat1, feat2, feat3", evidence=["[PAT-001]"], analysis="A"
        )
    ]
    dist_type1, _ = engine._assess_reference_distribution(single_dom_pats, 3)
    assert dist_type1 == "single_reference_dominated"

    # Distributed: 2 patents each match 1 feature (no single reference covers >= 70%)
    distributed_pats = [
        PatentAnalysisItem(
            patent_id="PAT-001", patent_title="P1", similarity_score=0.7,
            technical_problem="Prob", proposed_solution="Sol",
            matching_features=["feat1"], different_features=[],
            claim_relevance="", evidence=["[PAT-001]"], analysis="A"
        ),
        PatentAnalysisItem(
            patent_id="PAT-002", patent_title="P2", similarity_score=0.7,
            technical_problem="Prob", proposed_solution="Sol",
            matching_features=["feat2"], different_features=[],
            claim_relevance="", evidence=["[PAT-002]"], analysis="B"
        )
    ]
    dist_type2, _ = engine._assess_reference_distribution(distributed_pats, 3)
    assert dist_type2 == "distributed_across_corpus"
