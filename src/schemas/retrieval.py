from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.schemas.documents import DocumentType


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Document identifier e.g. PAT-001 or PAP-001")
    document_type: DocumentType = Field(..., description="patent or paper")
    document_title: str = Field(..., description="Title of parent document")
    section: str = Field("general", description="Section (e.g. claim, abstract, intro)")
    text: str = Field(..., description="Text content of the retrieved chunk")
    similarity_score: float = Field(..., description="Normalized cosine similarity [0.0, 1.0]")
    rank: int = Field(1, description="Rank within retrieved set")
    source: str = Field("", description="Source filename or identifier")
    page: Optional[int] = Field(None, description="Page number if available")
    citation: str = Field("", description="Explainable citation tag, e.g. [PAT-003, Claim 1]")
    extraction_method: str = Field("native", description="native | ocr")
    is_ocr: bool = Field(False, description="Whether chunk was extracted via OCR")
    ocr_quality: str = Field("UNKNOWN", description="HIGH | MEDIUM | LOW | UNKNOWN")
    metadata: Dict[str, Any] = Field(default_factory=dict)



class RetrievalResult(BaseModel):
    query: str = Field(..., description="Original user query or search phrase")
    patents: List[RetrievedChunk] = Field(default_factory=list, description="Top retrieved patent chunks")
    papers: List[RetrievedChunk] = Field(default_factory=list, description="Top retrieved research paper chunks")
    all_results: List[RetrievedChunk] = Field(default_factory=list, description="Combined ranked chunks")
    retrieval_summary: str = Field("", description="Overview summary of retrieved prior-art")
    retrieval_time_seconds: float = Field(0.0, description="Elapsed time for retrieval")
