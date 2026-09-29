from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.schemas.ideas import UserIdeaInput
from src.schemas.reports import FinalAnalysisReport, ComparativeAnalysisResult
from src.schemas.retrieval import RetrievedChunk
from src.workflows.orchestrator import IdeaAnalysisOrchestrator
from src.retrieval.patent_retriever import PatentRetriever
from src.retrieval.paper_retriever import PaperRetriever
from src.evaluation.benchmark import BenchmarkRunner
from src.config.settings import settings
from src.config.constants import LEGAL_DISCLAIMER
from src.utils.logging import logger

app = FastAPI(
    title="AI-Based Patent & Research Gap Discovery API",
    description="PBL-3 Academic Research Prototype for Novelty Assessment and White-Space Gap Discovery using RAG and Multi-Agent LLMs.",
    version="1.0.0"
)

# Global instances
orchestrator = IdeaAnalysisOrchestrator()
patent_retriever = PatentRetriever()
paper_retriever = PaperRetriever()
benchmark_runner = BenchmarkRunner(orchestrator)


class AnalyzeRequest(BaseModel):
    idea: str = Field(..., min_length=5, description="Natural language invention or research idea")
    mode: str = Field("multi_agent_rag", description="single_llm | rag | multi_agent_rag")
    domain: Optional[str] = Field(None, description="Optional domain identifier")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Top-k items to retrieve")


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "PBL-3 Patent & Research Gap Discovery",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "vector_store": settings.vector_store
    }


@app.get("/config", tags=["System"])
def get_public_config():
    """Returns runtime parameters without exposing sensitive API keys."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "top_k_patents": settings.top_k_patents,
        "top_k_papers": settings.top_k_papers,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "temperature": settings.temperature,
        "novelty_weights": {
            "semantic_similarity": settings.weight_semantic,
            "technical_overlap": settings.weight_overlap,
            "claim_coverage": settings.weight_claim
        },
        "disclaimer": LEGAL_DISCLAIMER
    }


@app.post("/analyze", response_model=FinalAnalysisReport, tags=["Analysis"])
def analyze(req: AnalyzeRequest):
    try:
        idea_input = UserIdeaInput(idea=req.idea, domain=req.domain)
        result = orchestrator.analyze(idea_input, mode=req.mode)
        return result
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparativeAnalysisResult, tags=["Analysis"])
def compare_modes(req: AnalyzeRequest):
    try:
        idea_input = UserIdeaInput(idea=req.idea, domain=req.domain)
        result = orchestrator.compare(idea_input)
        return result
    except Exception as e:
        logger.error(f"Error in /compare endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/patents", response_model=List[RetrievedChunk], tags=["Search"])
def search_patents(req: SearchRequest):
    try:
        return patent_retriever.retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Error in /search/patents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/papers", response_model=List[RetrievedChunk], tags=["Search"])
def search_papers(req: SearchRequest):
    try:
        return paper_retriever.retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Error in /search/papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", tags=["Evaluation"])
def run_evaluation(max_cases: int = Query(5, ge=1, le=20)):
    try:
        return benchmark_runner.run_benchmark(max_cases=max_cases)
    except Exception as e:
        logger.error(f"Error in /evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))
