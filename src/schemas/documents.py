from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PATENT = "patent"
    PAPER = "paper"


class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Unique stable ID, e.g. PAT-001 or PAP-001")
    document_type: DocumentType = Field(..., description="patent or paper")
    title: str = Field(..., description="Title of patent or research paper")
    publication_number: Optional[str] = Field(None, description="e.g. US10902042B2 or arXiv ID")
    authors: List[str] = Field(default_factory=list, description="Inventors or authors")
    assignee: Optional[str] = Field(None, description="Applicant / assignee / university")
    publication_date: Optional[str] = Field(None, description="Publication or grant date")
    source: str = Field(..., description="File source or origin")
    filename: str = Field(..., description="Original filename")
    abstract: str = Field("", description="Document abstract")
    num_pages: int = Field(0, description="Total pages in document")
    raw_char_count: int = Field(0, description="Total characters extracted")
    extraction_method: str = Field("native", description="native | ocr")
    is_ocr: bool = Field(False, description="Whether document text was produced via OCR")
    ocr_quality: str = Field("UNKNOWN", description="HIGH | MEDIUM | LOW | UNKNOWN")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique ID: doc_id-chunk_index")
    document_id: str = Field(..., description="Reference to parent DocumentMetadata")
    document_type: DocumentType = Field(..., description="patent or paper")
    document_title: str = Field(..., description="Title of parent document")
    section: str = Field("general", description="Section type: abstract, claim, description, intro, methodology, conclusion")
    text: str = Field(..., description="Cleaned chunk text")
    page: Optional[int] = Field(None, description="Source page number if available")
    chunk_index: int = Field(0, description="0-indexed position within document")
    source: str = Field("", description="Source filename or origin")
    extraction_method: str = Field("native", description="native | ocr")
    is_ocr: bool = Field(False, description="Whether chunk was extracted via OCR")
    ocr_quality: str = Field("UNKNOWN", description="HIGH | MEDIUM | LOW | UNKNOWN")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

