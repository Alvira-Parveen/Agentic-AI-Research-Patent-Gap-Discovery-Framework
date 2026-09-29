from typing import List, Optional, Tuple
from src.schemas.analysis import (
    NoveltyAssessmentResult,
    NoveltyFactorBreakdown,
    PatentAnalysisItem,
    PaperAnalysisItem,
    FeatureComparisonResult,
)
from src.schemas.retrieval import RetrievalResult, RetrievedChunk
from src.config.settings import settings
from src.config.constants import LEGAL_DISCLAIMER, GLOBAL_RESEARCH_DISCLAIMER


class NoveltyScoringEngine:
    def __init__(
        self,
        weight_semantic: Optional[float] = None,
        weight_overlap: Optional[float] = None,
        weight_claim: Optional[float] = None,
    ):
        self.w_semantic = weight_semantic if weight_semantic is not None else settings.weight_semantic
        self.w_overlap = weight_overlap if weight_overlap is not None else settings.weight_overlap
        self.w_claim = weight_claim if weight_claim is not None else settings.weight_claim

        # Ensure normalized weights
        total_w = self.w_semantic + self.w_overlap + self.w_claim
        if total_w > 0:
            self.w_semantic /= total_w
            self.w_overlap /= total_w
            self.w_claim /= total_w

    def _compute_claim_evidence_level(
        self,
        doc_chunks: List[RetrievedChunk],
        matching_features: List[str]
    ) -> Tuple[float, str]:
        """
        Computes structured claim evidence level:
        - 0.90: Disclosed in explicit claim section AND matching features identified.
        - 0.70: Non-claim section (description/abstract) with matching features.
        - 0.25: Claim section retrieved BUT no identified matching features.
        - 0.0:  No claims and no matching features.
        """
        has_claim_chunk = any(c.section.lower().startswith("claim") for c in doc_chunks)
        has_features = len(matching_features) > 0

        if has_claim_chunk and has_features:
            return 0.90, "Disclosed in explicit claim section with technical feature match (0.90)"
        elif not has_claim_chunk and has_features:
            return 0.70, "Non-claim specification disclosure with technical feature match (0.70)"
        elif has_claim_chunk and not has_features:
            return 0.25, "Claim section retrieved but no identified feature match (0.25)"
        else:
            return 0.0, "No claims and no matching features (0.0)"

    def _assess_reference_distribution(
        self,
        patent_analyses: List[PatentAnalysisItem],
        total_feats: int
    ) -> Tuple[str, str]:
        """
        Differentiates single-reference dominance vs. distributed feature overlap across references.
        """
        if total_feats <= 0 or not patent_analyses:
            return "INSUFFICIENT_EVIDENCE", "Insufficient features or references to assess distribution."

        max_matches = max((len(p.matching_features) for p in patent_analyses), default=0)
        ratio = max_matches / max(1, total_feats)

        if ratio >= 0.70:
            return "single_reference_dominated", f"Single reference dominates prior art ({max_matches}/{total_feats} features)."
        elif sum(len(p.matching_features) for p in patent_analyses) > 0:
            return "distributed_across_corpus", "Features distributed across multiple distinct references."
        else:
            return "INSUFFICIENT_EVIDENCE", "Insufficient prior-art feature matches to assess distribution."

    def calculate_novelty(
        self,
        retrieval_result: RetrievalResult,
        patent_analyses: List[PatentAnalysisItem],
        paper_analyses: List[PaperAnalysisItem],
        feature_comparison: FeatureComparisonResult
    ) -> NoveltyAssessmentResult:
        """
        Calculates a transparent prototype multi-factor prior-art overlap heuristic.
        NOTE: This is an automated engineering heuristic and NOT an empirically validated legal patentability metric.
        """
        all_chunks = retrieval_result.all_results if retrieval_result else []
        all_scores = [c.similarity_score for c in all_chunks]
        top_semantic = max(all_scores) if all_scores else 0.0

        # Factor 1: Semantic Similarity Rationale
        top_chunk = max(all_chunks, key=lambda c: c.similarity_score) if all_chunks else None
        sem_rationale = (
            f"Peak cosine similarity: {top_semantic:.4f} against {top_chunk.citation if top_chunk else 'N/A'}"
        )

        # Factor 2: Technical Feature Overlap (from semantic comparison matrix)
        total_feats = len(feature_comparison.comparison_matrix) if feature_comparison else 0
        matched_feats = sum(
            1 for item in (feature_comparison.comparison_matrix if feature_comparison else [])
            if getattr(item, "match_type", "") in ("MATCH", "PARTIAL_MATCH") or item.gap_status in ("Well Covered", "Partially Covered")
        )
        technical_overlap = (matched_feats / max(1, total_feats)) if total_feats > 0 else 0.0
        overlap_rationale = (
            f"{matched_feats}/{total_feats} proposed technical features matched or partially matched candidate prior-art."
        )

        # Factor 3: Structured Claim Evidence Factor
        claim_scores = []
        claim_notes = []
        patent_chunks = [c for c in all_chunks if str(c.document_type) == "patent" or "PAT" in c.document_id]

        for p in patent_analyses:
            doc_chunks = [c for c in patent_chunks if c.document_id == p.patent_id]
            score, note = self._compute_claim_evidence_level(doc_chunks, p.matching_features)
            claim_scores.append(score)
            claim_notes.append(f"{p.patent_id}: {note}")

        claim_feature_coverage = sum(claim_scores) / max(1, len(claim_scores)) if claim_scores else 0.0
        claim_rationale = "; ".join(claim_notes) if claim_notes else "No patent references evaluated."

        # Factor 4: Evidence Quality & Strength
        retrieval_density = min(1.0, len(all_chunks) / 6.0)
        mean_score = sum(all_scores) / max(1, len(all_scores)) if all_scores else 0.0
        evidence_strength = round(0.5 * retrieval_density + 0.5 * mean_score, 4)
        evidence_rationale = f"Retrieval density: {len(all_chunks)} chunks retrieved (mean sim: {mean_score:.4f})."

        # Check OCR evidence presence
        ocr_chunks = [c for c in all_chunks if getattr(c, "is_ocr", False)]
        if ocr_chunks:
            evidence_rationale += f" Note: {len(ocr_chunks)} chunk(s) derived from OCR text."

        # Factor Breakdown Rationales
        rationales = {
            "semantic_similarity": sem_rationale,
            "technical_overlap": overlap_rationale,
            "claim_feature_coverage": claim_rationale,
            "evidence_strength": evidence_rationale,
        }

        # Prior-Art Overlap Index (POI) - Heuristic weighted combination
        prior_art_overlap = (
            self.w_semantic * top_semantic +
            self.w_overlap * technical_overlap +
            self.w_claim * claim_feature_coverage
        )

        # Prototype Novelty Score
        novelty_score = max(0.0, min(1.0, 1.0 - (prior_art_overlap * (0.6 + 0.4 * evidence_strength))))
        novelty_score = round(novelty_score, 4)

        # Categorize Novelty Level (Prototype heuristic thresholds)
        if novelty_score >= 0.70:
            novelty_level = "High"
        elif novelty_score >= 0.40:
            novelty_level = "Moderate"
        else:
            novelty_level = "Low"

        confidence = round(0.4 * evidence_strength + 0.3 * (1.0 - abs(0.5 - novelty_score)) + 0.3 * (len(all_scores) / 6.0), 4)
        confidence = max(0.2, min(0.95, confidence))

        # Single-Reference Dominance vs Distributed Feature Overlap
        distribution_type, _ = self._assess_reference_distribution(patent_analyses, total_feats)

        factors = NoveltyFactorBreakdown(
            semantic_similarity=round(top_semantic, 4),
            technical_overlap=round(technical_overlap, 4),
            claim_feature_coverage=round(claim_feature_coverage, 4),
            evidence_strength=round(evidence_strength, 4),
            factor_rationales=rationales,
        )

        formula_explanation = (
            f"Prototype Multi-Factor Prior-Art Overlap Heuristic = 1.0 - (POI * Evidence_Scaling), where "
            f"POI = ({self.w_semantic:.2f} * SemanticSim[{top_semantic:.2f}] + "
            f"{self.w_overlap:.2f} * TechnicalOverlap[{technical_overlap:.2f}] + "
            f"{self.w_claim:.2f} * ClaimEvidence[{claim_feature_coverage:.2f}]). "
            f"Reference Distribution: {distribution_type}. (Heuristic parameters, NOT legally validated)."
        )

        # Supporting Evidence list
        supporting_evidence = []
        for p in patent_analyses[:2]:
            if p.evidence:
                supporting_evidence.extend(p.evidence)
        for r in paper_analyses[:2]:
            if r.evidence:
                supporting_evidence.extend(r.evidence)

        limitations = [
            "Assessment is based exclusively on the local 40-document PBL-3 seed corpus (20 patents, 20 research papers).",
            "This prototype heuristic estimates textual prior-art overlap; it does NOT determine statutory patent novelty or obviousness.",
            "Features distributed across multiple references do not constitute legal anticipation under 35 U.S.C. § 102.",
            "Weights are configurable research prototype parameters and have not been empirically calibrated against independent ground truth."
        ]
        if ocr_chunks:
            limitations.append("Contains OCR-derived patent evidence which may introduce transcription inaccuracies.")

        return NoveltyAssessmentResult(
            novelty_level=novelty_level,
            novelty_score=novelty_score,
            heuristic_score=novelty_score,
            is_heuristic=True,
            score_type="prototype_prior_art_overlap_heuristic",
            prior_art_overlap_signal=round(prior_art_overlap, 4),
            top_prior_art_similarity=round(top_semantic, 4),
            qualitative_preliminary_assessment=(
                f"Preliminary technical overlap assessment: {novelty_level} divergence from retrieved corpus "
                f"(prior art overlap signal: {prior_art_overlap:.4f}, top similarity: {top_semantic:.4f})."
            ),
            confidence=confidence,
            factors=factors,
            formula_explanation=formula_explanation,
            supporting_evidence=list(set(supporting_evidence)),
            distribution_type=distribution_type,
            limitations=limitations,
            legal_disclaimer=GLOBAL_RESEARCH_DISCLAIMER
        )
