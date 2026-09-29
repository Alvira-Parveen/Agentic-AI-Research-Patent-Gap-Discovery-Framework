import numpy as np
from typing import List, Dict, Set, Optional
from src.schemas.analysis import (
    PatentAnalysisItem,
    PaperAnalysisItem,
    FeatureComparisonItem,
    FeatureComparisonResult,
)
from src.schemas.retrieval import RetrievedChunk
from src.embeddings.embedding_service import EmbeddingService
from src.config.settings import settings
from src.config.constants import (
    MATCH_TYPE_MATCH,
    MATCH_TYPE_PARTIAL,
    MATCH_TYPE_UNCERTAIN,
    MATCH_TYPE_NO_MATCH,
)
from src.utils.logging import logger


class FeatureComparator:
    """
    Performs semantic feature comparison between user idea features and candidate prior-art evidence.
    Classifies feature overlap into MATCH, PARTIAL_MATCH, UNCERTAIN, or NO_MATCH using dense embeddings.
    """
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.match_threshold = settings.feature_match_threshold
        self.partial_threshold = settings.feature_partial_threshold
        self.uncertain_threshold = settings.feature_uncertain_threshold

    def compare(
        self,
        idea_features: List[str],
        patent_analyses: List[PatentAnalysisItem],
        paper_analyses: List[PaperAnalysisItem],
        candidate_chunks: Optional[List[RetrievedChunk]] = None
    ) -> FeatureComparisonResult:
        matrix: List[FeatureComparisonItem] = []
        common_elements: List[str] = []
        underrepresented_elements: List[str] = []

        if not idea_features:
            return FeatureComparisonResult(
                idea_features=[],
                comparison_matrix=[],
                common_elements=[],
                underrepresented_elements=[],
                cross_domain_insights="No technical features provided for comparison."
            )

        # Prepare candidate evidence texts
        evidence_pool: List[Dict[str, Any]] = []
        if candidate_chunks:
            for c in candidate_chunks:
                evidence_pool.append({
                    "text": c.text,
                    "document_id": c.document_id,
                    "document_type": str(c.document_type),
                    "chunk_id": c.chunk_id,
                    "citation": c.citation,
                    "is_ocr": getattr(c, "is_ocr", False),
                    "ocr_quality": getattr(c, "ocr_quality", "UNKNOWN"),
                })
        else:
            # Fall back to evidence excerpts stored in patent/paper analyses
            for p in patent_analyses:
                text_content = f"{p.proposed_solution} {' '.join(p.matching_features)} {p.analysis}"
                evidence_pool.append({
                    "text": text_content,
                    "document_id": p.patent_id,
                    "document_type": "patent",
                    "chunk_id": f"{p.patent_id}-EXCERPT",
                    "citation": f"[{p.patent_id}, Claims]",
                    "is_ocr": False,
                    "ocr_quality": "UNKNOWN",
                })
            for r in paper_analyses:
                text_content = f"{r.methodology} {' '.join(r.overlap_with_idea)} {r.analysis}"
                evidence_pool.append({
                    "text": text_content,
                    "document_id": r.paper_id,
                    "document_type": "paper",
                    "chunk_id": f"{r.paper_id}-EXCERPT",
                    "citation": f"[{r.paper_id}, Methodology]",
                    "is_ocr": False,
                    "ocr_quality": "UNKNOWN",
                })

        # Precompute embeddings if pool has items
        if evidence_pool:
            feat_embeddings = self.embedding_service.embed_texts(idea_features, normalize=True)
            chunk_texts = [e["text"][:800] for e in evidence_pool]
            chunk_embeddings = self.embedding_service.embed_texts(chunk_texts, normalize=True)
            # Dot products for normalized vectors = cosine similarity
            similarity_matrix = np.dot(feat_embeddings, chunk_embeddings.T)
        else:
            similarity_matrix = None

        for idx, feat in enumerate(idea_features):
            best_score = 0.0
            best_evidence = None

            if similarity_matrix is not None and len(evidence_pool) > 0:
                scores = similarity_matrix[idx]
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                best_evidence = evidence_pool[best_idx]

            # Classify match based on configurable prototype thresholds
            if best_score >= self.match_threshold:
                match_type = MATCH_TYPE_MATCH
                gap_status = "Well Covered"
                common_elements.append(feat)
            elif best_score >= self.partial_threshold:
                match_type = MATCH_TYPE_PARTIAL
                gap_status = "Partially Covered"
                common_elements.append(feat)
            elif best_score >= self.uncertain_threshold:
                match_type = MATCH_TYPE_UNCERTAIN
                gap_status = "Potential Gap"
                underrepresented_elements.append(feat)
            else:
                match_type = MATCH_TYPE_NO_MATCH
                gap_status = "Potential Gap"
                underrepresented_elements.append(feat)

            doc_id = best_evidence["document_id"] if best_evidence else ""
            doc_type = best_evidence["document_type"] if best_evidence else ""
            chunk_id = best_evidence["chunk_id"] if best_evidence else ""
            excerpt = best_evidence["text"][:240].strip() if best_evidence else "No relevant prior-art text found."
            is_ocr = best_evidence.get("is_ocr", False) if best_evidence else False
            ev_quality = "ocr_derived" if is_ocr else "native"

            explanation = (
                f"Semantic embedding similarity score: {best_score:.4f} against {doc_id}. "
                f"Classified as {match_type} using configurable prototype similarity threshold ({self.match_threshold:.2f}; unvalidated research parameter)."
            )
            if is_ocr:
                explanation += " Note: Evidence is OCR-derived and may contain transcription noise."

            matched_docs = [doc_id] if doc_id and match_type in (MATCH_TYPE_MATCH, MATCH_TYPE_PARTIAL) else []

            # Determine patent and paper coverage
            pat_cov = "Fully Covered" if match_type == MATCH_TYPE_MATCH and doc_type == "patent" else (
                "Partially Covered" if match_type == MATCH_TYPE_PARTIAL and doc_type == "patent" else "Uncovered"
            )
            pap_cov = "Fully Covered" if match_type == MATCH_TYPE_MATCH and doc_type == "paper" else (
                "Partially Covered" if match_type == MATCH_TYPE_PARTIAL and doc_type == "paper" else "Uncovered"
            )

            matrix.append(
                FeatureComparisonItem(
                    feature=feat,
                    user_feature=feat,
                    presence_in_idea=True,
                    patent_coverage=pat_cov,
                    paper_coverage=pap_cov,
                    matched_prior_art=matched_docs,
                    gap_status=gap_status,
                    match_type=match_type,
                    similarity_or_match_score=round(best_score, 4),
                    document_id=doc_id,
                    document_type=doc_type,
                    source_chunk_id=chunk_id,
                    evidence_excerpt=excerpt,
                    explanation=explanation,
                    evidence_quality=ev_quality,
                )
            )

        insights = (
            f"Semantic vector comparison: {len(common_elements)} features show evidence of prior-art alignment "
            f"(score >= {self.partial_threshold:.2f}) across the seed corpus, while {len(underrepresented_elements)} features "
            f"have low or uncertain semantic match within the retrieved local documents."
        )

        return FeatureComparisonResult(
            idea_features=idea_features,
            comparison_matrix=matrix,
            common_elements=common_elements,
            underrepresented_elements=underrepresented_elements,
            cross_domain_insights=insights
        )

    def assess_feature_presence(
        self,
        feature: str,
        candidate_chunks: List[RetrievedChunk]
    ) -> tuple:
        """
        Assesses presence of a single feature against candidate chunks using dense vector cosine similarity.
        Returns (match_type, similarity_score).
        """
        if not candidate_chunks:
            return MATCH_TYPE_NO_MATCH, 0.0

        feat_emb = self.embedding_service.embed_query(feature, normalize=True)
        chunk_texts = [c.text[:800] for c in candidate_chunks]
        chunk_embs = self.embedding_service.embed_texts(chunk_texts, normalize=True)
        scores = np.dot(chunk_embs, feat_emb)
        best_score = float(np.max(scores)) if len(scores) > 0 else 0.0

        if best_score >= self.match_threshold:
            return MATCH_TYPE_MATCH, best_score
        elif best_score >= self.partial_threshold:
            return MATCH_TYPE_PARTIAL, best_score
        elif best_score >= self.uncertain_threshold:
            return MATCH_TYPE_UNCERTAIN, best_score
        else:
            return MATCH_TYPE_NO_MATCH, best_score

    _assess_feature_presence = assess_feature_presence


