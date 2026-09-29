from typing import Dict, Any, Optional
from src.rag.retriever import RAGRetriever
from src.schemas.retrieval import RetrievalResult
from src.utils.logging import logger


class RetrievalAgent:
    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever or RAGRetriever()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes dense semantic retrieval against patent and paper corpora.
        Updates state with 'retrieval_result'.
        """
        idea_text = state.get("user_idea", "")
        features = state.get("extracted_features")
        
        # Build query enriched with extracted technical features if present
        query = idea_text
        if features and features.technical_features:
            query = f"{idea_text} {' '.join(features.technical_features[:3])}"

        logger.info(f"RetrievalAgent querying vector stores with: '{query[:80]}...'")
        result: RetrievalResult = self.retriever.retrieve(query)
        
        state["retrieval_result"] = result
        state["retrieved_patents"] = result.patents
        state["retrieved_papers"] = result.papers
        return state
