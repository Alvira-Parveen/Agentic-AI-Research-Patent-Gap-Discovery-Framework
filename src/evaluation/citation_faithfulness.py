import re
from enum import Enum
from typing import List, Set, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


COMMON_ABBREVIATIONS = [
    "e.g.", "i.e.", "et al.", "fig.", "dr.", "u.s.", "no.", "vs.", "approx.", "etc.", "vol.", "pp."
]

# ---------------------------------------------------------------------------
# Claim Classification
# ---------------------------------------------------------------------------

# Patterns used to partition claims into four mutually exclusive categories.
# These are applied in priority order: disclaimers first, then gap hypotheses,
# then uncited factuals, then prior-art factual claims.
_DISCLAIMER_PATTERNS = [
    r"not constitute", r"legal opinion", r"legal patentability",
    r"research prototype screening", r"additional verification",
    r"external patent databases", r"preliminary assessment",
    r"does not constitute", r"formal legal opinion"
]
_GAP_HYPOTHESIS_PATTERNS = [
    r"remains unaddressed", r"only partially documented",
    r"not yet addressed", r"not demonstrated", r"unresolved",
    r"white[\s\-]space", r"underrepresented"
]


class ClaimCategory(str, Enum):
    """Mutually exclusive categories that determine which denominator a claim falls under.

    Correct denominator selection is mandatory to avoid distorting faithfulness rates:

      - PRIOR_ART_FACTUAL  : Claims about prior art disclosed in retrieved passages.
                             Entailment labels (SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED)
                             are evaluated over this category only for the primary support rate.
      - GAP_HYPOTHESIS     : Claims that a feature or combination is *absent* or underexplored.
                             Rule-based matching cannot verify absence; label as NEEDS_EXPERT_REVIEW.
      - PROCEDURAL_DISCLAIMER : Mandatory legal / scientific caveats (e.g., "not a legal opinion").
                             These require no citation; they must be *excluded* from support-rate denominator.
      - UNCITED_FACTUAL    : Factual-sounding statements with no bracket citation anchor.
                             Reported separately; not included in the cited-claim support rate.
    """
    PRIOR_ART_FACTUAL = "prior_art_factual"
    GAP_HYPOTHESIS = "gap_hypothesis"
    PROCEDURAL_DISCLAIMER = "procedural_disclaimer"
    UNCITED_FACTUAL = "uncited_factual"


def classify_claim_category(claim_text: str, is_uncited: bool) -> ClaimCategory:
    """Assigns one of four mutually exclusive claim categories based on content heuristics.

    Priority order:
      1. Procedural disclaimers (no citation expected or useful).
      2. Gap hypotheses (absence claims — rule-based matching cannot verify).
      3. Uncited factual claims.
      4. Prior-art factual claims (primary denominator for citation support rates).
    """
    t = claim_text.lower()

    if any(re.search(p, t) for p in _DISCLAIMER_PATTERNS):
        return ClaimCategory.PROCEDURAL_DISCLAIMER

    if any(re.search(p, t) for p in _GAP_HYPOTHESIS_PATTERNS):
        return ClaimCategory.GAP_HYPOTHESIS

    if is_uncited:
        return ClaimCategory.UNCITED_FACTUAL

    return ClaimCategory.PRIOR_ART_FACTUAL


# ---------------------------------------------------------------------------
# Entailment Labels
# ---------------------------------------------------------------------------

class EntailmentLabel(str, Enum):
    """Mutually exclusive citation-support screening labels.

    IMPORTANT SCOPE NOTICE:
    These labels are produced by a *rule-based citation-support screening module*
    that uses keyword token overlap with morphological stemming and negation polarity
    checks. This is NOT semantic natural-language inference (NLI). Matching tokens
    between a claim and a passage does not guarantee logical entailment: a passage
    may mention the same technology in a different experimental setting or with a
    different outcome.

    The module provides a reproducible first-pass screening signal. Its labels must
    be compared against independent human judgments (using a confusion matrix and
    per-category precision/recall) before any claim of semantic entailment capability
    is made. Integration of an NLI model (e.g., DeBERTa-v3 fine-tuned on e-SNLI or
    SciNLI) should follow the same human-validation protocol.
    """
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class AnnotationWorkflowStatus(str, Enum):
    """Workflow status flags for human annotation and provenance auditing.

    Distinct from substantive EntailmentLabel values evaluated in model benchmarks.
    """
    UNCERTAIN = "UNCERTAIN"
    UNASSESSABLE = "UNASSESSABLE"
    NEEDS_ADJUDICATION = "NEEDS_ADJUDICATION"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClaimAuditResult:
    """Individual assertion audit record against a cited chunk passage."""
    claim_text: str
    cited_doc_or_chunk_id: str
    is_valid_provenance: bool
    is_uncited: bool
    claim_category: ClaimCategory
    entailment_label: EntailmentLabel
    evidence_rationale: str
    cited_passage_excerpt: Optional[str] = None


@dataclass
class CitationProvenance:
    """Provenance counts for a set of audited claims.

    Units are *citation occurrences* (one claim citing one document ID counts once).
    Each unique (claim_text, citation_id) pair is a separate occurrence. This is
    distinct from unique document IDs cited, which is reported separately.
    """
    total_cited_occurrences: int          # all claims with a bracket citation
    valid_provenance_count: int           # IDs present in the retrieved candidate pool
    invalid_provenance_count: int         # IDs absent from the retrieved candidate pool
    validity_rate: float                  # valid / total_cited_occurrences
    unique_cited_doc_ids: List[str]       # set of distinct document IDs cited (deduped)
    unique_invalid_doc_ids: List[str]     # distinct IDs that failed provenance check


@dataclass
class FaithfulnessSummary:
    """Aggregated citation faithfulness metrics with correctly partitioned denominators.

    METHODOLOGY NOTE:
    ─────────────────
    The support rate is computed **only over prior-art factual claims** — assertions
    that describe prior-art content and carry a bracket citation to a retrieved chunk.

    The following claim types are excluded from the support-rate denominator:
      • Procedural disclaimers  (e.g., "this is not a legal opinion") — no citation expected.
      • Gap hypotheses          (absence claims) — absence cannot be verified by token overlap.
      • Uncited factual claims  — reported separately; no citation to evaluate.

    This partitioning prevents disclaimers and gap statements from inflating the
    UNSUPPORTED count and misrepresenting the system's citation grounding quality.
    """
    # --- Total claim inventory ---
    total_claims: int
    prior_art_factual_count: int
    gap_hypothesis_count: int
    procedural_disclaimer_count: int
    uncited_factual_count: int

    # --- Provenance (over cited claims only; cited = not uncited) ---
    provenance: CitationProvenance

    # --- Entailment over PRIOR-ART FACTUAL claims only ---
    prior_art_supported: int
    prior_art_partially_supported: int
    prior_art_unsupported: int
    prior_art_contradicted: int
    prior_art_support_rate: float
    prior_art_partial_rate: float
    prior_art_unsupported_rate: float
    prior_art_contradiction_rate: float

    # --- Entailment over GAP HYPOTHESES (informational only) ---
    gap_label_distribution: Dict[str, int]

    # --- Full per-claim records ---
    claims: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Sentence / claim extractor
# ---------------------------------------------------------------------------

def extract_claims_with_citations(text: str) -> List[Dict[str, Any]]:
    """Segments text into sentences while protecting abbreviations and extracts
    all associated citation anchors.

    Handles:
      - Common technical abbreviations (e.g., i.e., et al., Fig., U.S.) — original
        casing is restored after segmentation; abbreviations are not lowercased.
      - Multiple citations in a single bracket: [PAT-001, PAP-019]
      - Multiple separate brackets: [PAT-001] [PAP-019]
      - Uncited claims: flagged with citation_ids=[] and is_uncited=True
      - Semicolon-delimited sub-assertions within a sentence

    Returns a list of dicts with keys:
        claim_text  : cleaned text with citation brackets removed
        citation_id : a single document/chunk ID string (one dict per citation)
        is_uncited  : True if no bracket citation was found
        raw_sentence: the original sentence string (for debugging)
        claim_category: ClaimCategory enum value
    """
    if not text or not text.strip():
        return []

    # 1. Mask common abbreviations to prevent incorrect sentence segmentation
    #    while preserving original casing using a per-occurrence token dictionary.
    masked_text = text.strip()
    abbr_tokens: Dict[str, str] = {}
    tok_counter = 0

    for abbr in COMMON_ABBREVIATIONS:
        pattern = re.compile(r'(?<!\w)' + re.escape(abbr), re.IGNORECASE)
        def repl(match, _tc=[tok_counter]):
            nonlocal tok_counter
            token = f"__ABBR_TOK_{tok_counter}__"
            abbr_tokens[token] = match.group(0)
            tok_counter += 1
            return token
        masked_text = pattern.sub(repl, masked_text)

    # 2. Split on sentence-terminal punctuation followed by whitespace
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', masked_text) if s.strip()]

    results = []
    for raw_s in raw_sentences:
        # Restore original abbreviation casing
        s_restored = raw_s
        for token, orig_str in abbr_tokens.items():
            s_restored = s_restored.replace(token, orig_str)

        # Split on semicolons to handle multi-assertion sentences
        sub_assertions = [sub.strip() for sub in re.split(r';\s*', s_restored) if sub.strip()]

        for assertion in sub_assertions:
            # Extract all bracketed citation IDs
            bracket_contents = re.findall(r'\[([^\]]+)\]', assertion)
            cited_ids: List[str] = []
            for content in bracket_contents:
                found_ids = re.findall(r'(PAT-\d+|PAP-\d+)', content)
                cited_ids.extend(found_ids)

            # Strip citation brackets from claim text
            clean_claim = re.sub(r'\[[^\]]*\]', '', assertion).strip()
            clean_claim = re.sub(r'\s+([,.:;])', r'\1', clean_claim)

            is_uncited = len(cited_ids) == 0
            category = classify_claim_category(clean_claim, is_uncited)

            if cited_ids:
                # Deduplicate citation IDs while preserving order
                seen: Set[str] = set()
                unique_cits = [cid for cid in cited_ids if not (cid in seen or seen.add(cid))]  # type: ignore[func-returns-value]
                for cit_id in unique_cits:
                    results.append({
                        "claim_text": clean_claim,
                        "citation_id": cit_id,
                        "is_uncited": False,
                        "raw_sentence": s_restored,
                        "claim_category": category.value,
                        "sub_assertion": clean_claim if len(sub_assertions) > 1 else None,
                    })
            else:
                results.append({
                    "claim_text": clean_claim,
                    "citation_id": "",
                    "is_uncited": True,
                    "raw_sentence": s_restored,
                    "claim_category": category.value,
                    "sub_assertion": clean_claim if len(sub_assertions) > 1 else None,
                })

    return results


# ---------------------------------------------------------------------------
# Rule-based citation-support screening
# ---------------------------------------------------------------------------

def _normalize_stem(word: str) -> str:
    """Basic morphological stemming to align inflected forms (e.g. network/networks)."""
    w = word.lower()
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            if suffix == "ies":
                return w[:-3] + "y"
            return w[:-len(suffix)]
    return w


def screen_citation_support_rule_based(
    claim_text: str,
    passage_text: str
) -> Tuple[EntailmentLabel, str]:
    """Rule-based citation-support screening module (NOT semantic NLI).

    Produces a reproducible first-pass signal by computing keyword token overlap
    with morphological stemming and negation polarity checks.

    LIMITATIONS (must be disclosed in any report):
      1. Token overlap does not capture logical implication. A passage may share
         vocabulary with a claim while describing a different result or context.
      2. Negation detection is pattern-based and will miss compositional negations,
         scope ambiguities, or implicit hedges.
      3. Thresholds (≥ 60% → SUPPORTED, ≥ 30% → PARTIALLY_SUPPORTED) are manually
         set heuristics; they have not been calibrated against a labelled dataset.
      4. Comparison with independent human labels via confusion matrix is mandatory
         before treating screening labels as validated entailment judgments.

    For production use, replace or supplement with a fine-tuned NLI model
    (e.g., DeBERTa-v3-large on e-SNLI or SciNLI) and validate its outputs
    against human-labelled examples.
    """
    if not passage_text or not passage_text.strip():
        return EntailmentLabel.UNSUPPORTED, "Cited passage is empty or unretrieved."

    claim_lower = claim_text.lower()
    passage_lower = passage_text.lower()

    # Extract substantive tokens (≥ 4 characters)
    claim_raw_tokens = re.findall(r'\b[a-z]{4,}\b', claim_lower)
    if not claim_raw_tokens:
        return EntailmentLabel.UNSUPPORTED, "Claim contains no substantive technical keywords."

    passage_raw_tokens = re.findall(r'\b[a-z]{4,}\b', passage_lower)
    passage_stems = {_normalize_stem(w) for w in passage_raw_tokens}
    passage_words = set(passage_raw_tokens)

    matched_tokens: Set[str] = set()
    for tok in set(claim_raw_tokens):
        stem = _normalize_stem(tok)
        if tok in passage_words or stem in passage_stems or tok in passage_lower:
            matched_tokens.add(tok)

    unique_claim_tokens = set(claim_raw_tokens)
    overlap_ratio = len(matched_tokens) / len(unique_claim_tokens)

    # Negation polarity check (pattern-based; see limitations above)
    negation_patterns = [r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bfails to\b', r'\bcannot\b', r'\binvalid\b']
    has_neg_claim = any(re.search(p, claim_lower) for p in negation_patterns)
    has_neg_passage = any(re.search(p, passage_lower) for p in negation_patterns)

    if has_neg_claim != has_neg_passage and overlap_ratio >= 0.60:
        return (
            EntailmentLabel.CONTRADICTED,
            f"Polarity conflict detected despite token alignment ({overlap_ratio:.1%}). "
            f"Note: pattern-based negation check; human review recommended."
        )

    if overlap_ratio >= 0.60:
        return (
            EntailmentLabel.SUPPORTED,
            f"Rule-based screening: token overlap {overlap_ratio:.1%} >= 60% threshold "
            f"(matched: {sorted(list(matched_tokens))[:4]}). Not semantic entailment — human review required."
        )
    elif overlap_ratio >= 0.30:
        return (
            EntailmentLabel.PARTIALLY_SUPPORTED,
            f"Rule-based screening: partial token overlap {overlap_ratio:.1%} (30–60%) "
            f"(matched: {sorted(list(matched_tokens))[:4]}). Not semantic entailment — human review required."
        )
    else:
        return (
            EntailmentLabel.UNSUPPORTED,
            f"Rule-based screening: insufficient token overlap ({overlap_ratio:.1%} < 30%)."
        )


# Keep the old name as an alias so existing callers do not break
evaluate_claim_entailment_rule_based = screen_citation_support_rule_based


# ---------------------------------------------------------------------------
# Claim auditor
# ---------------------------------------------------------------------------

def decompose_compound_claim(text: str) -> List[str]:
    """Decomposes a compound sentence into atomic sub-assertions where feasible.

    Splits on coordinating conjunctions (', and ', ', while ', ', whereas ', ', but ', '; ')
    or 'as well as' while preserving standalone grammatical clauses.
    """
    if not text or not text.strip():
        return []

    split_pattern = r'(?:;\s*|,\s*(?:and|while|whereas|but)\s+|\s+as well as\s+)'
    raw_clauses = re.split(split_pattern, text.strip())
    clauses = [c.strip() for c in raw_clauses if len(c.strip()) > 5]
    return clauses if len(clauses) > 1 else [text.strip()]


def audit_claim_against_retrieved_pool(
    claim: Dict[str, Any],
    retrieved_pool_ids: Set[str],
    doc_chunk_texts: Dict[str, str],
    use_atomic_decomposition: bool = True
) -> ClaimAuditResult:
    """Audits a single claim against the query's specific retrieved candidate pool.

    Decouples:
      1. Category assignment: which denominator bucket the claim belongs to.
      2. Provenance check: is the cited ID in the retrieved candidate pool?
      3. Entailment screening: does the cited passage support the claim?
         When use_atomic_decomposition=True, splits compound claims into atomic
         assertions and evaluates each against the passage. Conservative aggregation
         ensures a compound claim is NEVER labeled SUPPORTED merely because one
         part matches.

    The unit of provenance is a citation occurrence: one claim citing one document ID.
    """
    claim_text = claim["claim_text"]
    cit_id = claim.get("citation_id", "")
    is_uncited = claim.get("is_uncited", False)
    category_str = claim.get("claim_category", "")
    try:
        category = ClaimCategory(category_str)
    except ValueError:
        category = classify_claim_category(claim_text, is_uncited)

    if is_uncited or not cit_id:
        return ClaimAuditResult(
            claim_text=claim_text,
            cited_doc_or_chunk_id="",
            is_valid_provenance=False,
            is_uncited=True,
            claim_category=category,
            entailment_label=EntailmentLabel.UNSUPPORTED,
            evidence_rationale="Claim has no citation anchor (uncited statement)."
        )

    # 1. Provenance check: exact candidate pool membership
    is_valid_prov = cit_id in retrieved_pool_ids

    # 2. Retrieve passage text
    passage_text = doc_chunk_texts.get(cit_id, "")

    # 3. Citation-support screening
    if not is_valid_prov:
        if passage_text:
            label, rationale = screen_citation_support_rule_based(claim_text, passage_text)
            rationale = f"[Invalid Provenance: ID {cit_id} not in retrieved pool] {rationale}"
        else:
            label = EntailmentLabel.UNSUPPORTED
            rationale = f"[Invalid Provenance] Document {cit_id} was not in retrieved candidate pool."
    else:
        # Atomic sub-claim decomposition for compound sentences
        sub_clauses = decompose_compound_claim(claim_text) if use_atomic_decomposition else [claim_text]
        if len(sub_clauses) > 1 and passage_text:
            sub_results = [screen_citation_support_rule_based(sc, passage_text) for sc in sub_clauses]
            sub_labels = [r[0] for r in sub_results]

            # Conservative aggregation:
            # - If any sub-claim is CONTRADICTED -> CONTRADICTED
            # - If all sub-claims are SUPPORTED -> SUPPORTED
            # - If some are SUPPORTED and some are UNSUPPORTED/PARTIAL -> PARTIALLY_SUPPORTED
            #   (prevents marking a claim supported merely because one part matches)
            # - If all are UNSUPPORTED -> UNSUPPORTED
            if EntailmentLabel.CONTRADICTED in sub_labels:
                label = EntailmentLabel.CONTRADICTED
                rationale = "Contradiction detected in atomic sub-claim: " + "; ".join(r[1] for r in sub_results if r[0] == EntailmentLabel.CONTRADICTED)
            elif all(l == EntailmentLabel.SUPPORTED for l in sub_labels):
                label = EntailmentLabel.SUPPORTED
                rationale = f"All {len(sub_clauses)} atomic sub-claims supported by cited passage."
            elif any(l in (EntailmentLabel.SUPPORTED, EntailmentLabel.PARTIALLY_SUPPORTED) for l in sub_labels):
                label = EntailmentLabel.PARTIALLY_SUPPORTED
                n_supp = sum(1 for l in sub_labels if l in (EntailmentLabel.SUPPORTED, EntailmentLabel.PARTIALLY_SUPPORTED))
                rationale = f"Atomic claim decomposition: only {n_supp}/{len(sub_clauses)} sub-claims supported by passage. Conservative partial support assigned."
            else:
                label = EntailmentLabel.UNSUPPORTED
                rationale = f"Atomic claim decomposition: 0/{len(sub_clauses)} sub-claims supported by passage."
        else:
            label, rationale = screen_citation_support_rule_based(claim_text, passage_text)

    return ClaimAuditResult(
        claim_text=claim_text,
        cited_doc_or_chunk_id=cit_id,
        is_valid_provenance=is_valid_prov,
        is_uncited=False,
        claim_category=category,
        entailment_label=label,
        evidence_rationale=rationale,
        cited_passage_excerpt=passage_text[:300] if passage_text else None
    )


# ---------------------------------------------------------------------------
# Aggregator with corrected denominators
# ---------------------------------------------------------------------------

def audit_citation_faithfulness(
    audited_claims: List[ClaimAuditResult]
) -> FaithfulnessSummary:
    """Aggregates claim audit results with correctly partitioned denominators.

    Support rates are computed over PRIOR-ART FACTUAL claims only.
    Procedural disclaimers, gap hypotheses, and uncited factuals are
    reported separately and excluded from the primary support-rate denominator.
    """
    total = len(audited_claims)

    if total == 0:
        empty_prov = CitationProvenance(0, 0, 0, 0.0, [], [])
        return FaithfulnessSummary(
            total_claims=0,
            prior_art_factual_count=0,
            gap_hypothesis_count=0,
            procedural_disclaimer_count=0,
            uncited_factual_count=0,
            provenance=empty_prov,
            prior_art_supported=0,
            prior_art_partially_supported=0,
            prior_art_unsupported=0,
            prior_art_contradicted=0,
            prior_art_support_rate=0.0,
            prior_art_partial_rate=0.0,
            prior_art_unsupported_rate=0.0,
            prior_art_contradiction_rate=0.0,
            gap_label_distribution={},
            claims=[]
        )

    # --- Category counts ---
    def _cat(c: ClaimAuditResult) -> ClaimCategory:
        return c.claim_category if hasattr(c, 'claim_category') else classify_claim_category(c.claim_text, c.is_uncited)

    pa_claims = [c for c in audited_claims if _cat(c) == ClaimCategory.PRIOR_ART_FACTUAL]
    gap_claims = [c for c in audited_claims if _cat(c) == ClaimCategory.GAP_HYPOTHESIS]
    disc_claims = [c for c in audited_claims if _cat(c) == ClaimCategory.PROCEDURAL_DISCLAIMER]
    unc_claims = [c for c in audited_claims if _cat(c) == ClaimCategory.UNCITED_FACTUAL]

    # --- Provenance (cited = not uncited; unit = citation occurrence) ---
    cited_claims = [c for c in audited_claims if not c.is_uncited]
    valid_cited = [c for c in cited_claims if c.is_valid_provenance]
    invalid_cited = [c for c in cited_claims if not c.is_valid_provenance]

    unique_cited_ids = sorted(set(c.cited_doc_or_chunk_id for c in cited_claims if c.cited_doc_or_chunk_id))
    unique_invalid_ids = sorted(set(c.cited_doc_or_chunk_id for c in invalid_cited if c.cited_doc_or_chunk_id))

    prov_rate = round(len(valid_cited) / len(cited_claims), 4) if cited_claims else 0.0
    provenance = CitationProvenance(
        total_cited_occurrences=len(cited_claims),
        valid_provenance_count=len(valid_cited),
        invalid_provenance_count=len(invalid_cited),
        validity_rate=prov_rate,
        unique_cited_doc_ids=unique_cited_ids,
        unique_invalid_doc_ids=unique_invalid_ids,
    )

    # --- Entailment over prior-art factual claims ---
    n_pa = len(pa_claims)
    pa_supported = sum(1 for c in pa_claims if c.entailment_label == EntailmentLabel.SUPPORTED)
    pa_partial = sum(1 for c in pa_claims if c.entailment_label == EntailmentLabel.PARTIALLY_SUPPORTED)
    pa_unsupported = sum(1 for c in pa_claims if c.entailment_label == EntailmentLabel.UNSUPPORTED)
    pa_contradicted = sum(1 for c in pa_claims if c.entailment_label == EntailmentLabel.CONTRADICTED)

    def _rate(n: int) -> float:
        return round(n / n_pa, 4) if n_pa > 0 else 0.0

    # --- Gap hypothesis label distribution ---
    gap_label_dist: Dict[str, int] = {}
    for c in gap_claims:
        k = c.entailment_label.value
        gap_label_dist[k] = gap_label_dist.get(k, 0) + 1

    return FaithfulnessSummary(
        total_claims=total,
        prior_art_factual_count=len(pa_claims),
        gap_hypothesis_count=len(gap_claims),
        procedural_disclaimer_count=len(disc_claims),
        uncited_factual_count=len(unc_claims),
        provenance=provenance,
        prior_art_supported=pa_supported,
        prior_art_partially_supported=pa_partial,
        prior_art_unsupported=pa_unsupported,
        prior_art_contradicted=pa_contradicted,
        prior_art_support_rate=_rate(pa_supported),
        prior_art_partial_rate=_rate(pa_partial),
        prior_art_unsupported_rate=_rate(pa_unsupported),
        prior_art_contradiction_rate=_rate(pa_contradicted),
        gap_label_distribution=gap_label_dist,
        claims=[asdict(c) for c in audited_claims],
    )


# ---------------------------------------------------------------------------
# Multi-Passage & Compound Claim Aggregation Logic (Section 5.3)
# ---------------------------------------------------------------------------

def aggregate_atomic_evidence(
    passage_evaluations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregates multiple passage/citation evaluations for a single atomic proposition.

    Decision rules for single proposition:
      1. Refutation priority: If ANY passage contradicts -> CONTRADICTED
      2. Affirmation: Else if ANY passage supports -> SUPPORTED
      3. Partial: Else if ANY passage partially supports -> PARTIALLY_SUPPORTED
      4. Missing evidence: Else if all passages are unassessable -> UNASSESSABLE
      5. Failure to support: Else (all assessable passages fail to support) -> UNSUPPORTED
    """
    if not passage_evaluations:
        return {
            "label": AnnotationWorkflowStatus.UNASSESSABLE.value,
            "rationale": "No passage evaluations provided.",
            "assessable_passages": 0,
            "unassessable_passages": 0
        }

    labels = []
    unassessable_count = 0
    assessable_count = 0

    for ev in passage_evaluations:
        lbl = ev.get("label") or ev.get("entailment_label")
        if lbl in (AnnotationWorkflowStatus.UNASSESSABLE.value, "UNASSESSABLE") or ev.get("source_passage_recovered") is False:
            unassessable_count += 1
            labels.append(AnnotationWorkflowStatus.UNASSESSABLE.value)
        else:
            assessable_count += 1
            labels.append(lbl)

    if any(l == EntailmentLabel.CONTRADICTED.value for l in labels):
        return {
            "label": EntailmentLabel.CONTRADICTED.value,
            "rationale": "Refuted by at least one cited passage.",
            "assessable_passages": assessable_count,
            "unassessable_passages": unassessable_count
        }
    if any(l == EntailmentLabel.SUPPORTED.value for l in labels):
        return {
            "label": EntailmentLabel.SUPPORTED.value,
            "rationale": "Confirmed by at least one cited passage.",
            "assessable_passages": assessable_count,
            "unassessable_passages": unassessable_count
        }
    if any(l == EntailmentLabel.PARTIALLY_SUPPORTED.value for l in labels):
        return {
            "label": EntailmentLabel.PARTIALLY_SUPPORTED.value,
            "rationale": "Partially confirmed by cited passage(s).",
            "assessable_passages": assessable_count,
            "unassessable_passages": unassessable_count
        }
    if assessable_count == 0:
        return {
            "label": AnnotationWorkflowStatus.UNASSESSABLE.value,
            "rationale": "All cited passages could not be recovered (unassessable).",
            "assessable_passages": 0,
            "unassessable_passages": unassessable_count
        }
    return {
        "label": EntailmentLabel.UNSUPPORTED.value,
        "rationale": "All recovered passages fail to support proposition.",
        "assessable_passages": assessable_count,
        "unassessable_passages": unassessable_count
    }


def aggregate_claim_evidence(
    subclaim_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregates evaluated atomic subclaims into a composite claim-level verdict.

    Adheres strictly to Section 5.3 of docs/EVALUATION_PROTOCOL.md:
      - Rule 1 (CONTRADICTED): If ANY proposition is CONTRADICTED -> composite verdict is CONTRADICTED.
      - Rule 2 (UNASSESSABLE): If ALL propositions are UNASSESSABLE -> composite verdict is UNASSESSABLE.
      - Rule 3 (SUPPORTED): If ALL propositions are SUPPORTED -> composite verdict is SUPPORTED.
      - Rule 4 (PARTIALLY_SUPPORTED): If at least one proposition is SUPPORTED or PARTIALLY_SUPPORTED,
               and no proposition is CONTRADICTED, but some propositions are UNSUPPORTED,
               PARTIALLY_SUPPORTED, or UNASSESSABLE -> composite verdict is PARTIALLY_SUPPORTED.
      - Rule 5 (UNSUPPORTED): If all assessable propositions are UNSUPPORTED (and none supported/contradicted)
               -> composite verdict is UNSUPPORTED.
    """
    if not subclaim_records:
        return {
            "composite_label": AnnotationWorkflowStatus.UNASSESSABLE.value,
            "rationale": "No subclaims provided for aggregation.",
            "per_subclaim_verdicts": []
        }

    resolved_subclaims: List[Dict[str, Any]] = []

    for sc in subclaim_records:
        sub_id = sc.get("subclaim_id") or sc.get("atomic_subclaim_id", "S01")
        sub_text = sc.get("subclaim_text") or sc.get("atomic_subclaim") or sc.get("original_claim", "")

        # If sc already has a single resolved label
        if "label" in sc and "passage_evaluations" not in sc:
            sc_label = sc["label"]
            sc_rationale = sc.get("rationale", "")
            resolved_subclaims.append({
                "subclaim_id": sub_id,
                "subclaim_text": sub_text,
                "label": sc_label,
                "rationale": sc_rationale
            })
        else:
            passages = sc.get("passage_evaluations", [])
            agg_p = aggregate_atomic_evidence(passages)
            resolved_subclaims.append({
                "subclaim_id": sub_id,
                "subclaim_text": sub_text,
                "label": agg_p["label"],
                "rationale": agg_p["rationale"]
            })

    sub_labels = [r["label"] for r in resolved_subclaims]
    n_total = len(sub_labels)

    # Rule 1: Contradiction priority
    if any(l == EntailmentLabel.CONTRADICTED.value for l in sub_labels):
        n_contra = sum(1 for l in sub_labels if l == EntailmentLabel.CONTRADICTED.value)
        return {
            "composite_label": EntailmentLabel.CONTRADICTED.value,
            "rationale": f"Refutation detected: {n_contra}/{n_total} atomic propositions contradicted by evidence.",
            "per_subclaim_verdicts": resolved_subclaims
        }

    # Rule 2: All unassessable
    if all(l in (AnnotationWorkflowStatus.UNASSESSABLE.value, "UNASSESSABLE") for l in sub_labels):
        return {
            "composite_label": AnnotationWorkflowStatus.UNASSESSABLE.value,
            "rationale": f"Unassessable claim: all {n_total} atomic propositions cite unrecoverable passages.",
            "per_subclaim_verdicts": resolved_subclaims
        }

    # Rule 3: All supported
    if all(l == EntailmentLabel.SUPPORTED.value for l in sub_labels):
        return {
            "composite_label": EntailmentLabel.SUPPORTED.value,
            "rationale": f"Full support: all {n_total} atomic propositions confirmed by cited passages.",
            "per_subclaim_verdicts": resolved_subclaims
        }

    # Rule 4: Partial support (at least one supported/partially supported, none contradicted)
    has_positive = any(l in (EntailmentLabel.SUPPORTED.value, EntailmentLabel.PARTIALLY_SUPPORTED.value) for l in sub_labels)
    if has_positive:
        n_supp = sum(1 for l in sub_labels if l == EntailmentLabel.SUPPORTED.value)
        n_part = sum(1 for l in sub_labels if l == EntailmentLabel.PARTIALLY_SUPPORTED.value)
        n_unsupp = sum(1 for l in sub_labels if l == EntailmentLabel.UNSUPPORTED.value)
        n_unassess = sum(1 for l in sub_labels if l in (AnnotationWorkflowStatus.UNASSESSABLE.value, "UNASSESSABLE"))
        details = []
        if n_supp > 0: details.append(f"{n_supp} supported")
        if n_part > 0: details.append(f"{n_part} partially supported")
        if n_unsupp > 0: details.append(f"{n_unsupp} unsupported")
        if n_unassess > 0: details.append(f"{n_unassess} unassessable")
        return {
            "composite_label": EntailmentLabel.PARTIALLY_SUPPORTED.value,
            "rationale": f"Partial claim support ({', '.join(details)}).",
            "per_subclaim_verdicts": resolved_subclaims
        }

    # Rule 5: No proposition supported (all assessable propositions unsupported)
    return {
        "composite_label": EntailmentLabel.UNSUPPORTED.value,
        "rationale": f"Zero support: 0/{n_total} atomic propositions supported by cited evidence.",
        "per_subclaim_verdicts": resolved_subclaims
    }


# ---------------------------------------------------------------------------
# Pre-Adjudication Inter-Annotator Agreement Metrics (Section 5.5)
# ---------------------------------------------------------------------------

def compute_observed_agreement(labels_1: List[str], labels_2: List[str]) -> float:
    """Calculates observed raw agreement Po between two raters.

    Po = sum(1 for y1, y2 in zip(labels_1, labels_2) if y1 == y2) / len(labels_1)
    """
    if not labels_1 or len(labels_1) != len(labels_2):
        raise ValueError("labels_1 and labels_2 must be non-empty lists of identical length.")
    matches = sum(1 for y1, y2 in zip(labels_1, labels_2) if y1 == y2)
    return round(matches / len(labels_1), 4)


def compute_cohens_kappa(
    labels_1: List[str],
    labels_2: List[str],
    categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculates Cohen's kappa (κ) for two independent raters on nominal data,
    including detailed marginal distributions to identify class imbalance skew.
    """
    if not labels_1 or len(labels_1) != len(labels_2):
        raise ValueError("labels_1 and labels_2 must be non-empty lists of identical length.")

    n = len(labels_1)
    if categories is None:
        categories = sorted(list(set(labels_1) | set(labels_2)))

    po = sum(1 for y1, y2 in zip(labels_1, labels_2) if y1 == y2) / n

    # Marginal counts
    dist_1 = {cat: sum(1 for y in labels_1 if y == cat) for cat in categories}
    dist_2 = {cat: sum(1 for y in labels_2 if y == cat) for cat in categories}

    # Expected agreement Pe
    pe = sum((dist_1[cat] / n) * (dist_2[cat] / n) for cat in categories)

    if abs(1.0 - pe) < 1e-9:
        kappa = 1.0 if abs(po - pe) < 1e-9 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "n_items": n,
        "observed_agreement_Po": round(po, 4),
        "expected_agreement_Pe": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
        "categories": categories,
        "rater_1_marginal_distribution": dist_1,
        "rater_2_marginal_distribution": dist_2,
    }


def compute_krippendorff_alpha_nominal(
    labels_1: List[Optional[str]],
    labels_2: List[Optional[str]]
) -> Dict[str, Any]:
    """Calculates Krippendorff's alpha (α) for nominal data across two raters.

    Properly handles missing/unassigned items (None).
    D_o: Observed disagreement
    D_e: Expected disagreement based on pooled marginal frequencies
    alpha = 1 - (D_o / D_e)
    """
    if len(labels_1) != len(labels_2):
        raise ValueError("labels_1 and labels_2 must have identical length.")

    paired = [(y1, y2) for y1, y2 in zip(labels_1, labels_2) if y1 is not None or y2 is not None]
    if not paired:
        return {"n_units": 0, "krippendorff_alpha": 0.0, "status": "NO_VALID_DATA"}

    dual_coded = [(y1, y2) for y1, y2 in paired if y1 is not None and y2 is not None]
    if not dual_coded:
        return {"n_units": len(paired), "krippendorff_alpha": 0.0, "status": "INSUFFICIENT_OVERLAP"}

    all_labels = [y for y1, y2 in paired for y in (y1, y2) if y is not None]
    categories = sorted(list(set(all_labels)))

    n_dual = len(dual_coded)
    d_o = sum(1 for y1, y2 in dual_coded if y1 != y2) / n_dual

    total_annotations = len(all_labels)
    if total_annotations < 2:
        return {"n_units": len(paired), "krippendorff_alpha": 1.0 if d_o == 0.0 else 0.0}

    counts = {cat: all_labels.count(cat) for cat in categories}
    sum_n_k_sq = sum(c * (c - 1) for c in counts.values())
    denom = total_annotations * (total_annotations - 1)
    d_e = 1.0 - (sum_n_k_sq / denom) if denom > 0 else 0.0

    if abs(d_e) < 1e-9:
        alpha = 1.0 if d_o == 0.0 else 0.0
    else:
        alpha = 1.0 - (d_o / d_e)

    return {
        "n_units_total": len(paired),
        "n_dual_coded_units": n_dual,
        "observed_disagreement_Do": round(d_o, 4),
        "expected_disagreement_De": round(d_e, 4),
        "krippendorff_alpha": round(alpha, 4),
        "categories": categories,
        "pooled_marginal_distribution": counts
    }

