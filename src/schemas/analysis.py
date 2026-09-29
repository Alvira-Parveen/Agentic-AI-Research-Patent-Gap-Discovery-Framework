from typing import List, Optional
from pydantic import BaseModel, Field


class PatentAnalysisItem(BaseModel):
    patent_id: str = Field(..., description="e.g. PAT-001")
    patent_title: str = Field(..., description="Title of patent")
    similarity_score: float = Field(..., description="Deterministic similarity score from retriever")
    technical_problem: str = Field("", description="Problem addressed by patent")
    proposed_solution: str = Field("", description="Solution disclosed in patent")
    matching_features: List[str] = Field(default_factory=list, description="Overlapping features with user idea")
    different_features: List[str] = Field(default_factory=list, description="Distinct features not in user idea")
    claim_relevance: str = Field("", description="Relevance to patent claims")
    evidence: List[str] = Field(default_factory=list, description="Supporting citations and excerpt quotes")
    analysis: str = Field(..., description="Detailed technical analysis")


class PaperAnalysisItem(BaseModel):
    paper_id: str = Field(..., description="e.g. PAP-001")
    paper_title: str = Field(..., description="Title of research paper")
    similarity_score: float = Field(..., description="Deterministic similarity score from retriever")
    problem_addressed: str = Field("", description="Scientific research question addressed")
    methodology: str = Field("", description="Technical methodology and algorithms")
    findings: str = Field("", description="Key empirical findings or claims")
    limitations: str = Field("", description="Reported limitations in paper")
    overlap_with_idea: List[str] = Field(default_factory=list, description="Overlapping scientific concepts")
    differences: List[str] = Field(default_factory=list, description="Distinct scientific approaches")
    evidence: List[str] = Field(default_factory=list, description="Supporting paper citations and excerpts")
    analysis: str = Field(..., description="Detailed research analysis")


class FeatureComparisonItem(BaseModel):
    feature: str = Field(..., description="Technical feature name / phrase")
    user_feature: str = Field("", description="Exact user idea feature")
    presence_in_idea: bool = Field(True, description="Whether feature is declared in user idea")
    patent_coverage: str = Field("Uncovered", description="Fully Covered | Partially Covered | Uncovered")
    paper_coverage: str = Field("Uncovered", description="Fully Covered | Partially Covered | Uncovered")
    matched_prior_art: List[str] = Field(default_factory=list, description="List of citing patent/paper IDs")
    gap_status: str = Field("Potential Gap", description="Well Covered | Partially Covered | Potential Gap")
    match_type: str = Field("NO_MATCH", description="MATCH | PARTIAL_MATCH | UNCERTAIN | NO_MATCH")
    similarity_or_match_score: float = Field(0.0, description="Semantic similarity score to closest evidence chunk")
    document_id: str = Field("", description="Best matching document ID")
    document_type: str = Field("", description="patent | paper")
    source_chunk_id: str = Field("", description="Best matching chunk ID")
    evidence_excerpt: str = Field("", description="Text excerpt demonstrating match or partial match")
    explanation: str = Field("", description="Technical explanation of match or discrepancy")
    evidence_quality: str = Field("native", description="native | ocr_derived")


class FeatureComparisonResult(BaseModel):
    idea_features: List[str] = Field(default_factory=list)
    comparison_matrix: List[FeatureComparisonItem] = Field(default_factory=list)
    common_elements: List[str] = Field(default_factory=list)
    underrepresented_elements: List[str] = Field(default_factory=list)
    cross_domain_insights: str = Field("", description="Observations comparing patent literature vs scientific literature")


class NoveltyFactorBreakdown(BaseModel):
    semantic_similarity: float = Field(..., description="Max/mean semantic similarity [0.0, 1.0]")
    technical_overlap: float = Field(..., description="Ratio of overlapping technical features [0.0, 1.0]")
    claim_feature_coverage: float = Field(..., description="Ratio of claims covering core features [0.0, 1.0]")
    evidence_strength: float = Field(..., description="Confidence from retrieval density [0.0, 1.0]")
    factor_rationales: dict = Field(default_factory=dict, description="Explicit technical rationale for each factor")


class NoveltyAssessmentResult(BaseModel):
    novelty_level: str = Field(..., description="Low | Moderate | High | Unassessed")
    novelty_score: Optional[float] = Field(None, description="Calculated prototype heuristic score [0.0, 1.0] or null")
    heuristic_score: Optional[float] = Field(None, description="Explicit alias for prototype prior-art overlap novelty score")
    is_heuristic: bool = Field(True, description="Explicit indicator that this is a research heuristic, not an official novelty determination")
    score_type: str = Field("prototype_prior_art_overlap_heuristic", description="prototype_prior_art_overlap_heuristic | retrieval_overlap_signal | unassessed")
    prior_art_overlap_signal: Optional[float] = Field(None, description="Direct semantic retrieval overlap signal [0.0, 1.0]")
    top_prior_art_similarity: Optional[float] = Field(None, description="Peak cosine similarity with retrieved prior-art chunks [0.0, 1.0]")
    confidence: float = Field(0.0, description="Assessment confidence score [0.0, 1.0]")
    factors: Optional[NoveltyFactorBreakdown] = Field(None, description="Individual factor scores")
    formula_explanation: str = Field(..., description="Documented formula explanation and factor weights")
    supporting_evidence: List[str] = Field(default_factory=list, description="Citations supporting novelty assessment")
    distribution_type: str = Field("UNDETERMINED", description="single_reference_dominated | distributed_across_corpus | INSUFFICIENT_EVIDENCE")
    limitations: List[str] = Field(default_factory=list, description="Known assessment limitations")
    legal_disclaimer: str = Field(
        "Research prototype only. Results represent AI-assisted technical overlap analysis against the retrieved local corpus. "
        "They are not a legal patentability opinion, exhaustive prior-art search, or guarantee of novelty.",
        description="Mandatory global research disclaimer"
    )


class GapDiscoveryResult(BaseModel):
    well_covered_areas: List[str] = Field(default_factory=list, description="Features thoroughly addressed in literature")
    partially_covered_areas: List[str] = Field(default_factory=list, description="Features with partial coverage")
    underrepresented_features: List[str] = Field(default_factory=list, description="Features with sparse or no coverage in retrieved corpus")
    unexplored_combinations: List[str] = Field(default_factory=list, description="Less-explored intersections of features verified against evidence")
    patent_vs_paper_differences: str = Field("", description="Divergence between retrieved patents and research papers")
    potential_research_directions: List[str] = Field(default_factory=list, description="Recommended gap opportunities")
    evidence_citations: List[str] = Field(default_factory=list, description="Evidence citations grounding the gap analysis")
    gap_type: str = Field("seed_corpus_gap", description="potential_research_gap | potential_patent_whitespace | combination_gap | insufficient_evidence")
    limitation_statement: str = Field(
        "Research prototype limitation: Findings reflect underrepresentation within the retrieved local 40-document seed corpus, NOT global patent or scientific absence.",
        description="Corpus boundary limitation statement"
    )

