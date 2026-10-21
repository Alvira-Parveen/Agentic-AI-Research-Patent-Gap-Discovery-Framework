# Evaluation Protocol: Invention Prior-Art Assessment, Gap Discovery & Citation Faithfulness
## Research Annotation Guidelines & Experimental Validity Framework (PBL-3 Lab CSP391)

> **Important Legal & Scientific Notice**:
> 1. **Reference Framework Only**: System-generated novelty scores and reviewer annotations utilize 35 U.S.C. §§ 102/103 terminology as a structured reference framework. **They do NOT constitute legal determinations of patentability, freedom-to-operate, or patent validity.**
> 2. **Scientific Limitations**: Retrieval benchmarks and automated citation checks do not establish that downstream research gaps are novel, non-obvious, or technically feasible.
> 3. **Provisional Status**: The existing 20 test cases in `data/evaluation/test_cases.json` are developer-curated integration checks. Novelty agreement metrics remain provisional until adjudicated under this multi-reviewer protocol.

---

## 1. Prior-Art Relevance Criteria (Dual-Corpus Standard)

### A. Patent Prior Art (Patents & Published Applications)
A patent document is considered **relevant prior art** if:
- **Direct Technical Overlap**: One or more claims or detailed description embodiments explicitly disclose the specific technical mechanisms of the query idea.
- **Structural / Operational Analogy**: The patent teaches an equivalent sensor, algorithmic, or hardware architecture performing substantially the same function in the same domain.
- **Exclusion (Generic Background)**: Mere mention of broad domain buzzwords (e.g., "artificial intelligence", "drones") without disclosing the structural combination is **not** relevant.

### B. Scientific Research Papers
A research publication is considered **relevant prior art** if:
- **Algorithmic / Methodological Precedent**: The paper implements, formulates, or evaluates an equivalent core algorithm or workflow.
- **Empirical Baseline**: The paper establishes benchmarks on an identical or directly analogous problem setting.
- **Exclusion (Survey Articles)**: Broad surveys lacking specific technical overlap must not be labeled as core relevant prior art.

---

## 2. Decoupled Novelty & Research-Gap Annotation Schema

To prevent conflating distinct legal, technical, and literature constructs, reviewers must evaluate three separate fields independently:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DECOUPLED THREE-DIMENSION EVALUATION SCHEMA                     │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ 1. Section 102 Novelty   │ Single-source anticipation: Does one prior-art document     │
│    (Prior-Art Disclosure)│ disclose all claimed technical elements?                    │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Section 103 Obviousness│ Multi-source combination: Would combining known references  │
│    (Inventive Step)      │ be obvious to a Person Having Ordinary Skill in the Art?   │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Research-Gap Novelty  │ Literature white space: Is the proposed application gap     │
│    (Corpus White Space)  │ unaddressed across the examined patent and paper corpus?   │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### Dimension A: 35 U.S.C. § 102 — Novelty / Prior-Art Anticipation
- **`ANTICIPATED`**: A single retrieved prior-art document explicitly or inherently discloses every core technical feature of the query idea.
- **`NOT_ANTICIPATED`**: No single retrieved document discloses every core technical element in the combination.
- **`UNCERTAIN_102`**: Available evidence in the corpus is insufficient, ambiguous, or obscured by high OCR noise.

### Dimension B: 35 U.S.C. § 103 — Non-Obviousness Inquiry
- **`OBVIOUS_COMBINATION`**: The differences between the query idea and the prior art are such that the combination would have been obvious to a Person Having Ordinary Skill in the Art (PHOSITA) having access to the retrieved references.
- **`NON_OBVIOUS_INVENTION`**: The specific synergy, technical constraint resolution, or cross-domain adaptation provides a non-trivial inventive step not readily suggested by combining the references.
- **`UNCERTAIN_103`**: PHOSITA benchmark or secondary considerations of non-obviousness cannot be determined from the corpus alone.

### Dimension C: Research-Gap Novelty (Corpus & Literature White Space)
- **`GENUINE_GAP`**: The proposed research problem, benchmark, or hybrid architecture is genuinely unaddressed or unresolved in both the academic literature and commercial patents within the corpus.
- **`PARTIALLY_ADDRESSED`**: The gap has been recognized and partially solved in academic preprints or patent applications, but lacks comprehensive industrial validation.
- **`ALREADY_RESOLVED`**: Clear working embodiments or extensive literature benchmarks already address the stated problem.
- **`UNCERTAIN_GAP`**: Literature coverage is inconclusive.

### Mandatory Uncertainty & Confidence Recording
For every annotation, reviewers must record:
- `confidence`: Ordinal scale from 1 (Very Low / Speculative) to 5 (High / Verified against multiple primary sources).
- `uncertainty_rationale`: Required text field whenever `confidence <= 3` or any `UNCERTAIN_*` label is assigned.

---

## 3. Claim-to-Chunk Citation Faithfulness Audit Protocol

To evaluate factual grounding beyond superficial document ID matching, generated summaries are segmented into individual assertions and audited against cited text chunks:

| Entailment Label | Formal Definition | Operational Rule |
|---|---|---|
| **`SUPPORTED`** | Direct Entailment | The cited passage explicitly affirms all factual claims made in the sentence. |
| **`PARTIALLY_SUPPORTED`** | Partial Entailment | The cited passage confirms some elements of the sentence, but omits key quantitative or structural assertions. |
| **`UNSUPPORTED`** | Neutral / No Evidence | The cited passage does not contain sufficient evidence to confirm or refute the claim (claims may be ungrounded or extrapolated). |
| **`CONTRADICTED`** | Inconsistency / Refutation | The cited passage directly contradicts or disproves the claim made in the sentence. |

### Orthogonal Provenance Flag: `is_valid_provenance`
- **`True`**: The cited chunk ID or document ID exists in the system's retrieved top-$k$ candidate pool.
- **`False`**: The citation references a document ID outside the permitted candidate pool.
*Note: A claim may cite an invalid source ID while being factually true, or cite a valid source ID while being unsupported by the passage. Provenance and entailment are evaluated independently.*

---

## 4. Multi-Reviewer Adjudication Protocol

1. **Independent Review**: Two domain reviewers independently annotate each test case without viewing system-generated labels.
2. **Inter-Annotator Agreement**: Computed using Cohen's $\kappa$ across § 102, § 103, and research-gap categories.
3. **Formal Adjudication**: Discrepancies ($\kappa < 0.70$) are resolved by a third senior reviewer, with consensus rationales recorded in `data/evaluation/annotations.json`.

---

## 5. Real-World Citation-Faithfulness Evaluation Protocol & Sampling Manifest

To transition from the frozen 24-example diagnostic benchmark to independent empirical evaluation on actual research outputs, this protocol establishes rigorous standards for claim sampling, human annotation, provenance accounting, and system evaluation.

### 5.1 Sampling Unit, Manifest Specification & Scope

1. **Basic Sampling & Annotation Unit**:
   - The basic atomic unit of audit is the **claim–citation–passage occurrence**:
     $$\text{Unit} = (\text{annotation\_unit\_id}, \text{query\_id}, \text{system\_id}, \text{claim\_id}, \text{atomic\_subclaim\_id}, \text{citation\_id}, \text{passage\_id})$$
   - A single generated sentence citing multiple references or containing multiple atomic propositions is decomposed into distinct audit records, one per (atomic proposition, cited passage) pair.
2. **Sampling Protocol & Manifest**:
   - Prior to annotation, a frozen **Sampling Manifest** (`data/evaluation/real_world_sampling_manifest.json`) is generated from end-to-end rollouts across all 20 benchmark test cases (`TC-001` through `TC-020`).
   - All extracted **Prior-Art Factual Claims** generated by System B (Standard RAG) and System C (Multi-Agent Specialized RAG) are included.
   - Procedural disclaimers and gap hypotheses are partitioned into separate diagnostic buckets and strictly excluded from the support-rate denominator.
   - If System B and System C produce different numbers of substantive claims, the exact counts are reported transparently rather than artificially balanced.
   - Duplicate or near-duplicate claims across systems are preserved with distinct `system_id` values to enable valid comparative system-level grounding comparisons.
3. **Four Distinct Counting Levels (Disambiguation)**:
   To eliminate counting ambiguity across claims, subclaims, and occurrences, every manifest and report must itemize all four distinct levels:

   | Metric / Count | Definition | Real-World Benchmark Dataset ($N=20$) |
   |---|---|---|
   | **Unique Generated Claims** | Distinct `claim_id` strings (unique generated claim sentences) | **80 claims** (System B: 40, System C: 40) |
   | **Atomic Subclaims** | Distinct propositions created by fine-grained clause decomposition | **81 propositions** (79 single-clause + 2 from 1 compound claim) |
   | **Citation Occurrences** | Unique citation instances attached to claims across generation outputs | **230 unique occurrences** (228 valid provenance, 2 out-of-pool) |
   | **Claim–Citation–Passage Records** | Final individual annotation rows (`annotation_unit_id`) | **232 records** |

4. **Mathematical Expansion Reconciliation**:
   - The expansion from 230 citation occurrences to 232 annotation records is governed by the multi-proposition Cartesian product rule:
     $$\text{Annotation Records} = \text{Unique Citation Occurrences} + \sum_{c \in \text{Compound Claims}} (M_c - 1) \times K_c$$
     where $M_c$ is the number of atomic subclaims in compound claim $c$, and $K_c$ is the number of citations attached to claim $c$.
   - In the benchmark dataset, exactly one compound claim exists (`TC-003-SYSC-C01`), which decomposes into $M_c = 2$ atomic propositions (`S01` and `S02`) and cites $K_c = 2$ references (`OCC-0029` and `OCC-0030`).
   - The Cartesian product yields $2 \times 2 = 4$ annotation records (`ANN-0029` through `ANN-0032`), adding $(2 - 1) \times 2 = +2$ records over the unique occurrence count.
   - In earlier exploratory runs prior to passage deduplication, 238 occurrences expanded to 241 records via $K_c = 3$ citations ($238 + (2-1)\times 3 = 241$). Both demonstrate the exact algebraic expansion property.
   - **No Double-Counting Guarantee**: When computing Tier 1 citation-level metrics, records sharing the same `citation_occurrence_id` are grouped, guaranteeing that no citation occurrence is counted multiple times in provenance or citation support calculations.


### 5.2 Decoupled Provenance, Passage Recovery, and Citation Support

Provenance validity, source evidence recovery, and semantic citation support are distinct empirical questions and must never be conflated:

1. **Citation Provenance Validity**:
   Measures whether the retrieval subsystem actually supplied the cited document to the generation prompt:
   $$\text{Provenance Validity Rate} = \frac{\sum \text{Valid Citation Occurrences}}{\sum \text{All Citation Occurrences Audited}}$$
   A document ID cited across 5 different claims represents 5 citation occurrences, each evaluated against that specific query's retrieved candidate pool.
2. **Source Passage Recovery & Handling of Missing Evidence**:
   - If a citation ID is present in the retrieved pool, its text is directly available (`source_passage_recovered = True`).
   - If a citation ID is outside the query's retrieved pool but exists in the broader patent/paper corpus, its passage is recovered from the corpus (`source_passage_recovered = True`) so that annotators and models can evaluate true semantic support.
   - If a cited document ID cannot be recovered anywhere in the corpus (e.g. fabricated or corrupted ID), `source_passage_recovered = False`. In the annotation workflow, such cases are labeled **`UNASSESSABLE`** rather than automatically assumed to be semantically unsupported or contradictory.
3. **Citation Semantic Support (Faithfulness)**:
   Evaluated strictly over assessable cited passages to determine if the cited premise logically confirms, partially confirms, fails to confirm, or refutes the specific proposition.
4. **End-to-End Retrieval-Grounded Support**:
   Reported separately as a composite system metric where out-of-pool or unrecoverable citations cannot count as grounding achieved by that retrieval-generation pipeline.

### 5.3 Compound Claim Decomposition & Executable Aggregation

1. **Claim Preservation**:
   The original, intact claim sentence as generated is permanently preserved in `original_claim` to ensure auditability and prevent loss of causal qualifiers, contrastive markers, or scope conditions. Multi-clause assertions are segmented into `atomic_subclaims` with stable `atomic_subclaim_id` identifiers.
2. **Executable Multi-Passage Decision Rules**:
   When a claim consists of multiple atomic propositions or cites multiple passages, pair-level verdicts are aggregated via the executable function `aggregate_claim_evidence()` adhering to these priority rules:
   - **Step 1: Proposition-Level Evidence Synthesis** (for proposition $S_i$ citing passages $P_{i,1} \dots P_{i,k}$):
     - **Refutation Priority**: If *any* cited passage contradicts $S_i$ $\to$ proposition verdict is `CONTRADICTED`.
     - **Affirmation**: Else if *any* cited passage fully confirms $S_i$ $\to$ proposition verdict is `SUPPORTED`.
     - **Partial Affirmation**: Else if *any* cited passage partially supports $S_i$ $\to$ proposition verdict is `PARTIALLY_SUPPORTED`.
     - **Missing Evidence**: Else if all cited passages are unrecoverable $\to$ proposition verdict is `UNASSESSABLE`.
     - **Failure to Support**: Else (all assessable passages fail to support) $\to$ proposition verdict is `UNSUPPORTED`.
   - **Step 2: Claim-Level Composite Verdict** (across propositions $S_1 \dots S_m$):
     - **Rule 1 (`CONTRADICTED`)**: If **any** atomic proposition is `CONTRADICTED` $\to$ composite verdict is `CONTRADICTED`.
     - **Rule 2 (`UNASSESSABLE`)**: If **all** propositions cite unrecoverable passages $\to$ composite verdict is `UNASSESSABLE`.
     - **Rule 3 (`SUPPORTED`)**: If **all** atomic propositions are `SUPPORTED` $\to$ composite verdict is `SUPPORTED`.
     - **Rule 4 (`PARTIALLY_SUPPORTED`)**: If at least one proposition is `SUPPORTED` or `PARTIALLY_SUPPORTED`, and no proposition is `CONTRADICTED`, while others are `UNSUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNASSESSABLE` $\to$ composite verdict is `PARTIALLY_SUPPORTED`.
     - **Rule 5 (`UNSUPPORTED`)**: If all assessable propositions are `UNSUPPORTED` (and none supported or contradicted) $\to$ composite verdict is `UNSUPPORTED`.

### 5.4 Annotation Workflow with Explicit Uncertainty

Human annotators must never be forced to guess when passage evidence is truncated, highly technical, or ambiguous.

1. **Four Substantive Evaluated Classes**:
   Reviewers evaluate assessable pairs into canonical classes:
   - `SUPPORTED`: Cited passage directly confirms the proposition.
   - `PARTIALLY_SUPPORTED`: Cited passage confirms core concepts but omits specific parameters or bounds.
   - `UNSUPPORTED`: Cited passage mentions related topics but provides no evidence for the proposition.
   - `CONTRADICTED`: Cited passage directly refutes or provides opposite empirical findings.
2. **Annotation Workflow Statuses (Non-Evaluated Classes)**:
   - `UNCERTAIN` / `NEEDS_ADJUDICATION`: Flagged when technical phrasing or context makes support ambiguous.
   - `UNASSESSABLE`: Flagged when cited source passage cannot be recovered from the corpus.
   These are **workflow tracking states**, not substantive model evaluation classes.
3. **Mandatory Reviewer Rationales & Adjudication**:
   - Both Reviewer 1 and Reviewer 2 record independent rationales (`reviewer_1_rationale`, `reviewer_2_rationale`).
   - Discrepancies and `UNCERTAIN` cases are resolved in a joint consensus session, producing `adjudicated_label` and `adjudication_rationale`. The adjudication rate must be reported.

### 5.5 Pre-Adjudication Inter-Annotator Agreement & Query Clustering

To avoid biased consensus reporting, agreement between independent annotators must be calculated **prior to adjudication**:
- **Raw Observed Agreement ($P_o$)**: Proportion of identical labels across all dual-coded items.
- **Cohen's Kappa ($\kappa$)**: Nominal inter-rater agreement accounting for chance, accompanied by full disclosure of marginal distributions to detect class-imbalance effects.
- **Krippendorff's Alpha ($\alpha$)**: Nominal alpha across all units with transparent handling of missing/unassigned observations.
- **Caution on Statistical Clustering**: Multiple claims generated from the same query are not statistically independent. When performing comparative significance testing between System B and System C, tests must cluster or block by `query_id` rather than treating individual rows as independent and identically distributed ($i.i.d.$) samples.

### 5.6 Protection of the Independent Test Set

To maintain scientific integrity and prevent test-set contamination:
- Model predictions must remain completely hidden from annotators during review.
- Model predictions are generated and stored in a decoupled evaluation file (`data/evaluation/real_world_model_predictions.json`).
- All three automated baseline systems (Rule-Based, Standalone NLI, Hybrid B) remain **strictly frozen** with pre-defined thresholds. No post-hoc parameter or threshold tuning is permitted on the real-world dataset.

---

### 5.7 Standard 18-Field Annotation Record Schema

Every audited claim–passage occurrence is stored in the following standardized schema:

| Field | Type | Description |
|---|---|---|
| `annotation_unit_id` | String | Unique record identifier (`ANN-0001` .. `ANN-0213`) |
| `query_id` | String | Benchmark query identifier (`TC-001` .. `TC-020`) |
| `system_id` | String | Generator architecture (`System B: Standard RAG` or `System C: Multi-Agent RAG`) |
| `claim_id` | String | Unique stable claim identifier (e.g., `TC-001-SYSB-C01`) |
| `original_claim` | String | Exact intact claim sentence as generated |
| `atomic_subclaim_id` | String | Proposition identifier (e.g., `TC-001-SYSB-C01-S01`) |
| `atomic_subclaim` | String | Decomposed atomic proposition text |
| `citation_id` | String | Cited document or chunk identifier (e.g., `PAT-016`) |
| `passage_id` | String | Specific passage identifier evaluated (`PAT-016-RETRIEVED`, `CORPUS-RECOVERED`, etc.) |
| `passage_text` | Optional[String] | Text excerpt of cited passage premise (or None if unrecoverable) |
| `provenance_valid` | Boolean | `True` if `citation_id` exists in query's top-$k$ FAISS candidate pool |
| `source_passage_recovered` | Boolean | `True` if cited text was retrieved or recovered from corpus |
| `reviewer_1_label` | Optional[String] | Reviewer 1 label (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `UNCERTAIN`, `UNASSESSABLE`) |
| `reviewer_1_rationale` | Optional[String] | Reviewer 1 textual rationale and evidence |
| `reviewer_2_label` | Optional[String] | Reviewer 2 label (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `UNCERTAIN`, `UNASSESSABLE`) |
| `reviewer_2_rationale` | Optional[String] | Reviewer 2 textual rationale and evidence |
| `adjudicated_label` | Optional[String] | Final consensus label after adjudication |
| `adjudication_rationale` | Optional[String] | Consensus rationale and resolution notes |

---

### 5.8 Standard 8-Section Final Evaluation Report Structure

When presenting real-world evaluation findings, reports must adhere to this uniform 8-section structure:

1. **Dataset Construction & Sampling Manifest**: Detailed description of query selection ($N=20$), 4-count breakdown, duplicate handling, and multi-citation occurrence breakdown.
2. **Annotation Protocol & Reviewer Agreement**: Inter-annotator agreement metrics ($P_o$, Cohen's $\kappa$, Krippendorff's $\alpha$) before adjudication, adjudication rate, and class marginal distributions.
3. **Provenance & Passage Recovery (Decoupled)**: Valid occurrence rate reported separately from faithfulness metrics: $\frac{\text{Valid Occurrences}}{\text{Total Audited Occurrences}}$ and recovery rate.
4. **Citation-Faithfulness Results**: Evaluation of the three frozen systems (Rule-Based, Standalone NLI, Two-Stage Hybrid B) against adjudicated ground truth.
5. **Confusion Matrices & Per-Class Metrics**: 4x4 contingency tables and per-class Precision, Recall, and F1-Scores using canonical `PARTIALLY_SUPPORTED`.
6. **Error Analysis & Failure Taxonomy**: Detailed linguistic inspection of complementary errors (antonym blindness, synonym shifts, partial grounding inflation).
7. **Runtime & Hardware Benchmark**: Standardized warm inference latency (mean, median, range) with decoupled initialization disclosure.
8. **Limitations & Research-Gap Implications**: Clear boundary distinguishing citation factual support from the scientific validity and patentability (§ 102/103) of discovered research gaps.

