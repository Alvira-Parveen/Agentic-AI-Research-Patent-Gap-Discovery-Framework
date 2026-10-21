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
        idea_input = state.get("idea_input")
        title = idea_input.title if (idea_input and hasattr(idea_input, "title")) else ""

        # Build query enriched with title and extracted technical features if present
        query = f"{title}. {idea_text}" if title else idea_text
        if features and features.technical_features:
            query = f"{query} {' '.join(features.technical_features[:3])}"

        logger.info(f"RetrievalAgent querying vector stores with: '{query[:80]}...'")
        result: RetrievalResult = self.retriever.retrieve(query)
        
        state["retrieval_result"] = result
        state["retrieved_patents"] = result.patents
        state["retrieved_papers"] = result.papers
        return state
