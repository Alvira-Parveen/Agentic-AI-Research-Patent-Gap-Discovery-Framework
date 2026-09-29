from typing import Optional, List
import time
from src.retrieval.patent_retriever import PatentRetriever
from src.retrieval.paper_retriever import PaperRetriever
from src.schemas.retrieval import RetrievalResult, RetrievedChunk
from src.config.settings import settings
from src.utils.logging import logger


class RAGRetriever:
    def __init__(
        self,
        patent_retriever: Optional[PatentRetriever] = None,
        paper_retriever: Optional[PaperRetriever] = None
    ):
        self.patent_retriever = patent_retriever or PatentRetriever()
        self.paper_retriever = paper_retriever or PaperRetriever()

    def retrieve(
        self,
        query: str,
        top_k_patents: Optional[int] = None,
        top_k_papers: Optional[int] = None
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        k_pat = top_k_patents or settings.top_k_patents
        k_pap = top_k_papers or settings.top_k_papers

        patents = self.patent_retriever.retrieve(query, top_k=k_pat)
        papers = self.paper_retriever.retrieve(query, top_k=k_pap)

        all_chunks: List[RetrievedChunk] = patents + papers
        all_chunks = sorted(all_chunks, key=lambda x: x.similarity_score, reverse=True)

        elapsed = round(time.perf_counter() - t0, 4)
        summary = (
            f"Retrieved {len(patents)} patent chunks and {len(papers)} research paper chunks "
            f"in {elapsed:.3f}s. Max similarity: {all_chunks[0].similarity_score if all_chunks else 0.0:.4f}"
        )

        return RetrievalResult(
            query=query,
            patents=patents,
            papers=papers,
            all_results=all_chunks,
            retrieval_summary=summary,
            retrieval_time_seconds=elapsed
        )
