# Evaluation Protocol: Invention Prior-Art Assessment & Gap Discovery
## Research Annotation Guidelines & Experimental Validity Framework (PBL-3 Lab CSP391)

> **Important Scientific Notice**:
> This document governs the annotation and evaluation methodology for the AI-Based Patent and Research Gap Discovery project. The existing test cases in `data/evaluation/test_cases.json` are currently classified as **`provisional_developer_curated`**. They serve as software integration consistency checks. Until independent human annotations are completed under this protocol, precision, recall, and novelty agreement metrics **CANNOT be presented as evidence of generalized real-world performance**.

---

## 1. Definition of a Relevant Patent

A patent is considered **relevant prior art** if and only if it satisfies one of the following criteria:
1. **Direct Claim Overlap (High Relevance)**: An independent or dependent claim in the patent explicitly recites one or more core technical mechanisms of the proposed invention idea.
2. **Detailed Specification Disclosure (Moderate Relevance)**: The patent's written description or drawings disclose the technical architecture, sensor fusion pipeline, or algorithm, even if the specific feature is not formally claimed in the patent's claim tree.
3. **Background Art / Generic Precedent (Low Relevance / Non-Target)**: The patent merely mentions the domain (e.g. "artificial intelligence", "drones") without disclosing the specific structural or operational combination.

**Evidence Requirement**: An annotator must record the exact patent identifier (e.g. `PAT-016`), the section (`claims` or `description`), and the verbatim excerpt supporting the relevance call.

---

## 2. Definition of a Relevant Research Paper

A scientific research paper is considered **relevant prior art** if:
1. **Methodological Precedent**: The paper proposes, implements, or mathematically models the same or an equivalent algorithmic technique (e.g. contrastive multi-agent evaluation, tree search over claim representations).
2. **Empirical Benchmark Precedent**: The paper evaluates the specific technical problem on an equivalent dataset (e.g. multispectral crop disease classification).
3. **Domain Survey (Contextual Only)**: Papers providing general overviews without algorithmic overlap must not be labeled as core relevant prior art.

---

## 3. How Technical Feature Overlap Should Be Judged

Annotators must evaluate technical feature overlap using **functional equivalence**, not lexical or substring overlap:
- **MATCH**: The candidate document discloses the same technical mechanism performing the same function on equivalent inputs to produce equivalent outputs (e.g., "convolutional neural network for leaf disease classification" matches "CNN architecture for foliar pathogen identification").
- **PARTIAL_MATCH**: The candidate document discloses a related technical mechanism with significant architectural divergence (e.g., "drone obstacle avoidance sensors" partially matches "drone multispectral imaging sensor payload").
- **UNCERTAIN**: The disclosure is ambiguous or obscured by high OCR noise or broad patent legal jargon.
- **NO_MATCH**: The feature is completely unaddressed in the document.

---

## 4. How Novelty Should Be Annotated

Human annotators must evaluate preliminary novelty on an ordinal scale with documented justification:
- **Low Novelty**: A single retrieved patent or research paper anticipates the core technical mechanisms, or the proposed combination is trivial for a person having ordinary skill in the art (PHOSITA).
- **Moderate Novelty**: The individual components exist in prior art, but their specific combination, cross-domain adaptation, or operational workflow demonstrates a non-trivial distinction.
- **High Novelty**: No direct or closely analogous prior art is found in the literature; the proposed architecture represents a substantial departure from known methods.

**Rule**: Annotators must NOT assign novelty based on abstract idea descriptions; they must evaluate the specific combination of technical features.

---

## 5. How Research & Patent Gaps Should Be Annotated

A gap cannot be asserted in a vacuum. Annotators must substantiate gaps by contrasting two distinct literature bodies:
1. **Potential Research Gap**: A technical mechanism is claimed or disclosed in commercial patent literature, but lacks empirical academic benchmarking, open-source replication, or formal scientific analysis.
2. **Potential Patent White-Space Signal**: An algorithmic technique is studied in academic research literature, but is absent from commercial patent claims within the examined corpus.
3. **Combination White Space**: Two proven techniques whose joint coupling is absent from all retrieved prior-art references.

---

## 6. Inter-Annotator Agreement & Adjudication

To eliminate circular bias and subjective variance:
1. **Double-Blind Annotation**: Each test case must be independently evaluated by at least two domain experts (e.g. patent professionals or postgraduate AI researchers).
2. **Statistical Agreement Metric**: Inter-rater reliability must be quantified using **Cohen's Kappa ($\kappa$)** for categorical novelty levels and **Fleiss' Kappa** if $>2$ annotators participate.
3. **Adjudication Procedure**: Cases where annotators disagree (e.g. Low vs Moderate) must be reviewed in a consensus session with a senior adjudicator, and the consensus rationale recorded in `adjudication_status`.

---

## 7. Evidence Provenance Requirements

Every valid annotation must include:
- `annotator_id`: Anonymized identifier of the human expert.
- `annotation_date`: ISO-8601 timestamp.
- `confidence`: Subjective annotator confidence (0.0 to 1.0).
- `evidence_notes`: Exact citations (`[PAT-003, Claim 1]`, `[PAP-010, Section 3.2]`) and verbatim quotes.

---

## 8. What Metrics Are Valid Before Human Annotation Exists?

Until human annotations are completed, the evaluation suite strictly separates software execution metrics from empirical AI quality metrics:

| Metric | Status Before Human Annotation | Academic Validity |
|---|:---:|---|
| **Pipeline Latency (seconds)** | **VALID** | Measures true computational time across modes. |
| **Citation Grounding Ratio** | **VALID** | Measures whether generated citations strictly exist in retrieved FAISS chunks (hallucination control). |
| **Pydantic Schema Conformance** | **VALID** | Verifies structural integrity of API and workflow outputs. |
| **Precision@3 & Recall@3** | **PROVISIONAL ONLY** | Reflects consistency against developer-curated candidate targets, NOT an independent gold standard. |
| **Novelty Agreement Ratio** | **N/A (WITHHELD)** | Automatically withheld (`N/A`) until human numerical novelty annotations exist. |
| **Gap Agreement Ratio** | **N/A (WITHHELD)** | Automatically withheld (`N/A`) until human gap annotations exist. |

> **Mandatory Academic Disclosure**:
> Any presentation or publication referencing this project must explicitly report whether metrics were calculated on `provisional_developer_curated` test cases or independently validated human benchmarks.
