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
            # Merge deterministic findings into LLM output
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
        research_gaps = []       # In patents but not papers
        patent_whitespaces = []  # In papers but not patents

        for item in matrix:
            pat_cov = getattr(item, "patent_coverage", "Uncovered")
            pap_cov = getattr(item, "paper_coverage", "Uncovered")
            feat_name = getattr(item, "user_feature", "") or item.feature

            if pat_cov in ("Fully Covered", "Partially Covered") and pap_cov in ("Fully Covered", "Partially Covered"):
                well_covered.append(feat_name)
            elif pat_cov in ("Fully Covered", "Partially Covered") and pap_cov == "Uncovered":
                partially_covered.append(feat_name)
                research_gaps.append(
                    f"Potential Research Gap: '{feat_name}' was identified in retrieved patent evidence "
                    f"({', '.join(item.matched_prior_art)}) but was not identified in retrieved research-paper "
                    f"evidence. This indicates underrepresentation in the retrieved corpus and does not establish global absence."
                )
            elif pap_cov in ("Fully Covered", "Partially Covered") and pat_cov == "Uncovered":
                partially_covered.append(feat_name)
                patent_whitespaces.append(
                    f"Potential Patent White-Space Signal: '{feat_name}' was identified in retrieved "
                    f"research-paper evidence ({', '.join(item.matched_prior_art)}) but was not identified in "
                    f"retrieved patent claims. This indicates underrepresentation in the retrieved corpus and does not establish global absence."
                )
            else:
                underrepresented.append(feat_name)

        # ----------------------------------------------------------------
        # Enriched combination gap generation — produce multiple diverse
        # gaps instead of a single entry, covering all gap categories.
        # ----------------------------------------------------------------
        unexplored_combos = []

        # (a) Research gaps: each patent-covered / paper-absent feature
        for rg in research_gaps[:2]:
            feat = rg.split("'")[1] if "'" in rg else "feature"
            unexplored_combos.append(
                f"Corpus-level research gap: '{feat}' is disclosed in retrieved patent claims but has no "
                f"corresponding academic publication in the retrieved corpus, suggesting an opportunity for "
                f"empirical validation research (scope limited to 40 seed documents)."
            )

        # (b) Patent white-spaces: each paper-covered / patent-absent feature
        for pw in patent_whitespaces[:2]:
            feat = pw.split("'")[1] if "'" in pw else "feature"
            unexplored_combos.append(
                f"Patent white-space signal: '{feat}' has academic research coverage in the retrieved corpus "
                f"but no corresponding patent claim boundary, indicating a potential IP filing opportunity "
                f"(scope limited to 40 seed documents)."
            )

        # (c) Fully underrepresented features: absent from both corpora
        for ur in underrepresented[:2]:
            unexplored_combos.append(
                f"Dual-corpus white space: '{ur}' was not identified in either retrieved patent or "
                f"research-paper evidence, representing a technically unexplored direction within the "
                f"local seed corpus (does not establish global absence)."
            )

        # (d) Cross-domain combinations: pair research-gaps with patent-whitespaces
        if research_gaps and patent_whitespaces:
            rg_feat = research_gaps[0].split("'")[1] if "'" in research_gaps[0] else "patent-covered feature"
            pw_feat = patent_whitespaces[0].split("'")[1] if "'" in patent_whitespaces[0] else "research-covered feature"
            unexplored_combos.append(
                f"Cross-domain combination gap: Joint implementation of '{rg_feat}' (patent-covered) "
                f"and '{pw_feat}' (research-covered) shows no precedent in the retrieved 40-document "
                f"seed corpus, indicating a potential integration opportunity."
            )

        # (e) Feature-pair combination within underrepresented set
        if len(underrepresented) >= 2:
            unexplored_combos.append(
                f"Feature combination gap: Coupling '{underrepresented[0]}' with '{underrepresented[1]}' "
                f"has no joint prior-art precedent in the retrieved corpus, representing a candidate for "
                f"novel system design."
            )

        # Fallback if no gaps identified at all
        if not unexplored_combos:
            unexplored_combos.append(
                "General corpus gap: The retrieved 40-document seed corpus provides broad coverage of the "
                "submitted idea's features. Extending retrieval to larger patent databases (e.g., USPTO, EPO) "
                "is recommended to identify more specific prior-art boundaries."
            )

        # ----------------------------------------------------------------
        # Evidence-grounded cross-corpus differences summary
        # ----------------------------------------------------------------
        pat_ids = [p.patent_id for p in patent_analyses if p.patent_id]
        pap_ids = [p.paper_id for p in paper_analyses if p.paper_id]
        if pat_ids and pap_ids:
            differences = (
                f"Retrieved patents ({', '.join(pat_ids)}) disclose technical implementations and claim "
                f"boundaries, whereas retrieved research papers ({', '.join(pap_ids)}) focus on algorithmic "
                f"formulations and benchmark evaluations. Identified {len(research_gaps)} potential research "
                f"gap(s) and {len(patent_whitespaces)} patent white-space signal(s)."
            )
        else:
            differences = "Insufficient cross-domain evidence to contrast patent claims against academic research."

        # ----------------------------------------------------------------
        # Research directions — one per gap type, enriched set
        # ----------------------------------------------------------------
        directions = []
        for rg in research_gaps[:2]:
            feat = rg.split("'")[1] if "'" in rg else "identified feature"
            directions.append(f"Empirical validation: Conduct a formal benchmark study on '{feat}' to bridge the patent-to-research gap.")
        for pw in patent_whitespaces[:2]:
            feat = pw.split("'")[1] if "'" in pw else "identified feature"
            directions.append(f"IP opportunity: Investigate patent claim formulation for '{feat}' given its academic research coverage.")
        for ur in underrepresented[:2]:
            directions.append(f"Foundational research: Investigate alternative architectural implementations for '{ur}'.")
        if research_gaps and patent_whitespaces:
            directions.append("Integration study: Design a system combining the identified patent-covered and research-covered features.")
        if not directions:
            directions = ["Investigate cross-domain benchmarking within seed corpus constraints."]

        citations = []
        for p in patent_analyses[:3]:
            citations.extend(p.evidence)
        for r in paper_analyses[:3]:
            citations.extend(r.evidence)

        primary_gap_type = (
            "potential_research_gap" if research_gaps and not patent_whitespaces else (
                "potential_patent_whitespace" if patent_whitespaces and not research_gaps else (
                    "combination_gap" if (research_gaps and patent_whitespaces) else (
                        "underrepresented_feature_gap" if underrepresented else "broad_coverage"
                    )
                )
            )
        )

        return GapDiscoveryResult(
            well_covered_areas=well_covered,
            partially_covered_areas=partially_covered,
            underrepresented_features=underrepresented,
            unexplored_combinations=unexplored_combos,
            patent_vs_paper_differences=differences,
            potential_research_directions=directions[:6],
            evidence_citations=list(set(citations)),
            gap_type=primary_gap_type,
            limitation_statement=(
                "Findings reflect technical underrepresentation within the retrieved local corpus "
                "(40 seed documents) and do not establish global patent or scientific absence."
            )
        )
