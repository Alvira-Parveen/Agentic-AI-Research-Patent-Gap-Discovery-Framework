from typing import List, Optional
from src.schemas.analysis import (
    GapDiscoveryResult,
    PatentAnalysisItem,
    PaperAnalysisItem,
    FeatureComparisonResult,
)
from src.schemas.ideas import ExtractedFeatures
from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.rag.prompts import SYSTEM_STRICT_EVIDENCE, GAP_DISCOVERY_PROMPT
from src.utils.logging import logger


class GapDiscoveryEngine:
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def discover_gaps(
        self,
        extracted_features: ExtractedFeatures,
        patent_analyses: List[PatentAnalysisItem],
        paper_analyses: List[PaperAnalysisItem],
        feature_comparison: FeatureComparisonResult
    ) -> GapDiscoveryResult:
        """
        Synthesizes well-covered vs underrepresented technical areas
        and discovers white-space gaps across patent and paper corpora.
        """
        idea_feats_str = ", ".join(extracted_features.technical_features)
        pat_summary = "\n".join([f"- [{p.patent_id}] Matches: {', '.join(p.matching_features)} | Diff: {', '.join(p.different_features)}" for p in patent_analyses])
        pap_summary = "\n".join([f"- [{p.paper_id}] Overlap: {', '.join(p.overlap_with_idea)} | Diff: {', '.join(p.differences)}" for p in paper_analyses])

        prompt = GAP_DISCOVERY_PROMPT.format(
            idea_features=idea_feats_str,
            patent_summary=pat_summary or "No patent prior art retrieved.",
            paper_summary=pap_summary or "No research papers retrieved."
        )

        try:
            structured = self.llm.generate_structured(
                prompt=prompt,
                schema_cls=GapDiscoveryResult,
                system_prompt=SYSTEM_STRICT_EVIDENCE
            )
            # Merge deterministic findings
            for underrep in feature_comparison.underrepresented_elements:
                if underrep not in structured.underrepresented_features:
                    structured.underrepresented_features.append(underrep)
            return structured
        except Exception as e:
            logger.warning(f"Structured gap discovery failed: {e}. Falling back to deterministic discovery.")
            return self._deterministic_gap_discovery(extracted_features, patent_analyses, paper_analyses, feature_comparison)

    def _deterministic_gap_discovery(
        self,
        extracted_features: ExtractedFeatures,
        patent_analyses: List[PatentAnalysisItem],
        paper_analyses: List[PaperAnalysisItem],
        feature_comparison: FeatureComparisonResult
    ) -> GapDiscoveryResult:
        if not patent_analyses and not paper_analyses:
            return GapDiscoveryResult(
                well_covered_areas=[],
                partially_covered_areas=[],
                underrepresented_features=extracted_features.technical_features if extracted_features else [],
                unexplored_combinations=[],
                patent_vs_paper_differences="Insufficient evidence in retrieved seed corpus to identify a reliable gap.",
                potential_research_directions=["Expand corpus query scope beyond current 40 seed documents."],
                evidence_citations=[],
                gap_type="insufficient_evidence",
                limitation_statement="No relevant prior-art documents were retrieved to establish a grounded gap analysis."
            )

        matrix = feature_comparison.comparison_matrix if feature_comparison else []
        well_covered = []
        partially_covered = []
        underrepresented = []
        research_gaps = []
        patent_whitespaces = []

        for item in matrix:
            pat_cov = getattr(item, "patent_coverage", "Uncovered")
            pap_cov = getattr(item, "paper_coverage", "Uncovered")
            feat_name = getattr(item, "user_feature", "") or item.feature

            if pat_cov in ("Fully Covered", "Partially Covered") and pap_cov in ("Fully Covered", "Partially Covered"):
                well_covered.append(feat_name)
            elif pat_cov in ("Fully Covered", "Partially Covered") and pap_cov == "Uncovered":
                partially_covered.append(feat_name)
                research_gaps.append(
                    f"Potential Research Gap: '{feat_name}' was identified in retrieved patent evidence ({', '.join(item.matched_prior_art)}) "
                    f"but was not identified in retrieved research-paper evidence. This indicates underrepresentation in the retrieved corpus and does not establish global absence."
                )
            elif pap_cov in ("Fully Covered", "Partially Covered") and pat_cov == "Uncovered":
                partially_covered.append(feat_name)
                patent_whitespaces.append(
                    f"Potential Patent White-Space Signal: '{feat_name}' was identified in retrieved research-paper evidence ({', '.join(item.matched_prior_art)}) "
                    f"but was not identified in retrieved patent claims. This indicates underrepresentation in the retrieved corpus and does not establish global absence."
                )
            else:
                underrepresented.append(feat_name)

        # Combination gaps grounded strictly in verified disjoint sets
        unexplored_combos = []
        if research_gaps and patent_whitespaces:
            unexplored_combos.append(
                f"Cross-domain white space: Intersection of academic concept '{patent_whitespaces[0].split(chr(39))[1]}' with patent-disclosed mechanism '{research_gaps[0].split(chr(39))[1]}' is underrepresented in the retrieved local corpus (does not establish global absence)."
            )
        elif underrepresented and well_covered:
            unexplored_combos.append(
                f"Candidate white space: Coupling underrepresented feature '{underrepresented[0]}' with baseline '{well_covered[0]}' shows no joint precedent in the retrieved seed documents (does not establish global absence)."
            )
        elif underrepresented and partially_covered:
            unexplored_combos.append(
                f"Candidate white space: Coupling underrepresented feature '{underrepresented[0]}' with prior-art element '{partially_covered[0]}' shows no joint precedent in the retrieved seed documents (does not establish global absence)."
            )

        # Evidence-grounded differences
        pat_ids = [p.patent_id for p in patent_analyses if p.patent_id]
        pap_ids = [p.paper_id for p in paper_analyses if p.paper_id]
        if pat_ids and pap_ids:
            differences = (
                f"Retrieved patents ({', '.join(pat_ids)}) disclose technical implementations and claim boundaries, "
                f"whereas retrieved research papers ({', '.join(pap_ids)}) focus on algorithmic formulations and benchmark evaluations. "
                f"Identified {len(research_gaps)} potential research gap(s) and {len(patent_whitespaces)} patent white-space signal(s)."
            )
        else:
            differences = "Insufficient cross-domain evidence to contrast patent claims against academic research."

        # Research directions
        directions = []
        for rg in research_gaps:
            directions.append(f"Empirical validation: Conduct formal benchmark study on {rg.split(chr(39))[1]}.")
        for pw in patent_whitespaces:
            directions.append(f"Commercial opportunity: Investigate patent claim formulation for {pw.split(chr(39))[1]}.")
        for ur in underrepresented:
            directions.append(f"Foundational research: Investigate alternative architectural implementations for {ur}.")

        if not directions:
            directions = ["Investigate cross-domain benchmarking within seed corpus constraints."]

        citations = []
        for p in patent_analyses[:2]:
            citations.extend(p.evidence)
        for r in paper_analyses[:2]:
            citations.extend(r.evidence)

        primary_gap_type = (
            "potential_research_gap" if research_gaps and not patent_whitespaces else (
                "potential_patent_whitespace" if patent_whitespaces and not research_gaps else (
                    "combination_gap" if unexplored_combos else "underrepresented_feature_gap"
                )
            )
        )

        return GapDiscoveryResult(
            well_covered_areas=well_covered,
            partially_covered_areas=partially_covered,
            underrepresented_features=underrepresented,
            unexplored_combinations=unexplored_combos,
            patent_vs_paper_differences=differences,
            potential_research_directions=directions[:4],
            evidence_citations=list(set(citations)),
            gap_type=primary_gap_type,
            limitation_statement="Findings reflect technical underrepresentation within the retrieved local corpus (40 seed documents) and do not establish global patent or scientific absence."
        )
