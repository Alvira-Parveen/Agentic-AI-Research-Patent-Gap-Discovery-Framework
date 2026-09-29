from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from src.schemas.ideas import UserIdeaInput, ExtractedFeatures
from src.schemas.retrieval import RetrievalResult
from src.schemas.analysis import (
    PatentAnalysisItem,
    PaperAnalysisItem,
    FeatureComparisonResult,
    NoveltyAssessmentResult,
    GapDiscoveryResult,
)


class FinalAnalysisReport(BaseModel):
    idea: UserIdeaInput = Field(..., description="Original user idea input")
    extracted_features: ExtractedFeatures = Field(..., description="Structured feature representation")
    mode: str = Field(..., description="single_llm | rag | multi_agent_rag")
    retrieval: Optional[RetrievalResult] = Field(None, description="Retrieval evidence if applicable")
    patent_analysis: List[PatentAnalysisItem] = Field(default_factory=list, description="Patent analysis results")
    research_analysis: List[PaperAnalysisItem] = Field(default_factory=list, description="Research paper analysis results")
    feature_comparison: Optional[FeatureComparisonResult] = Field(None, description="Detailed feature comparison matrix")
    novelty: Optional[NoveltyAssessmentResult] = Field(None, description="Preliminary novelty assessment")
    gaps: Optional[GapDiscoveryResult] = Field(None, description="Identified patent and research gaps")
    final_summary: str = Field(..., description="Executive summary and conclusions")
    limitations: List[str] = Field(default_factory=list, description="Explicit system limitations")
    confidence: float = Field(0.0, description="Overall confidence score [0.0, 1.0]")
    processing_time_seconds: float = Field(0.0, description="Elapsed execution time")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ComparativeAnalysisResult(BaseModel):
    idea: UserIdeaInput
    single_llm: FinalAnalysisReport
    rag: FinalAnalysisReport
    multi_agent_rag: FinalAnalysisReport
    comparison_summary: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
