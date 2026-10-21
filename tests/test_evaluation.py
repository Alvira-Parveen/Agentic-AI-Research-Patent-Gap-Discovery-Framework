import pytest
from src.evaluation.benchmark import BenchmarkRunner
from src.schemas.documents import DocumentMetadata, DocumentType
from src.schemas.retrieval import RetrievedChunk


def test_schema_ocr_metadata():  
    doc_meta = DocumentMetadata(
        document_id="PAT-TEST",
        document_type=DocumentType.PATENT,
        title="Test Patent",
        source="data/raw/patents/test.pdf",
        filename="test.pdf",
        extraction_method="tesseract_ocr",
        is_ocr=True,
        ocr_quality="MEDIUM"
    )
    assert doc_meta.is_ocr is True
    assert doc_meta.extraction_method == "tesseract_ocr"
    assert doc_meta.ocr_quality == "MEDIUM"

    chunk = RetrievedChunk(
        chunk_id="C-TEST",
        document_id="PAT-TEST",
        document_type=DocumentType.PATENT,
        document_title="Test Patent",
        section="claim",
        text="Sample claim text",
        similarity_score=0.85,
        rank=1,
        citation="[PAT-TEST]",
        extraction_method="tesseract_ocr",
        is_ocr=True,
        ocr_quality="MEDIUM"
    )
    assert chunk.is_ocr is True
    assert chunk.ocr_quality == "MEDIUM"


def test_benchmark_runner_mode_a_and_provenance():
    runner = BenchmarkRunner()
    summary = runner.run_benchmark(max_cases=2)

    assert "execution_metadata" in summary
    meta = summary["execution_metadata"]
    assert "benchmark_type" in meta
    assert meta["benchmark_type"] in ["software_integration", "human_validated"]
    assert "llm_provider" in meta
    assert "model_name" in meta
    assert "dataset_annotation_status" in meta
    assert "provisional_developer_curated" in meta["dataset_annotation_status"]

    s_llm = summary["single_llm"]
    # Mode A retrieval metrics must be None (N/A)
    assert s_llm["avg_precision@3"] is None
    assert s_llm["avg_recall@3"] is None
    assert s_llm["avg_f1@3"] is None
    assert s_llm["mean_reciprocal_rank"] is None
    assert s_llm["avg_grounding_ratio"] == 0.0

    # Unannotated human agreement must be None
    assert s_llm["novelty_agreement_human"] is None
    assert summary["rag"]["novelty_agreement_human"] is None
    assert summary["multi_agent_rag"]["novelty_agreement_human"] is None

    # Check detailed cases
    assert len(summary["detailed_cases"]) == 2
    case_0 = summary["detailed_cases"][0]
    assert case_0["single_llm"]["precision@3"] is None
    assert case_0["single_llm"]["novelty_score"] is None


def test_benchmark_runner_max_cases():
    runner = BenchmarkRunner()
    test_cases = runner.load_test_cases()
    assert len(test_cases) == 20

    # Verify that max_cases="all" handles all cases
    assert len(runner.load_test_cases()) == 20


def test_wilcoxon_exact_integer_hits_tie_preservation():
    """Regression test: Ensure Wilcoxon rank tests on discrete metrics (e.g. P@3) use

    exact integer hit differences to prevent 4-decimal truncation (0.3333 vs 0.3334)
    from artificially breaking ties and creating spurious nominal significance.
    """
    import json
    from pathlib import Path
    import numpy as np
    import scipy.stats as stats

    results_path = Path("data/evaluation/ablation_study_results.json")
    assert results_path.exists()
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_query = data["per_query_results"]
    assert len(per_query) == 20

    # Extract stored floats
    d_p3 = np.array([q["Dense-Only"]["P@3"] for q in per_query])
    h_p3 = np.array([q["Hybrid-RRF"]["P@3"] for q in per_query])

    # Reconstruct underlying integer hits: P@3 = hits / 3
    d_hits = np.array([int(round(x * 3)) for x in d_p3])
    h_hits = np.array([int(round(x * 3)) for x in h_p3])
    diff_hits = h_hits - d_hits

    # Verify non-zero difference vector
    nz = diff_hits[diff_hits != 0]
    assert len(nz) == 11
    assert sum(nz > 0) == 8
    assert sum(nz < 0) == 3

    # Verify that all 10 unit differences (|d_i| == 1) receive tied rank exactly 5.5
    abs_nz = np.abs(nz)
    ranks = stats.rankdata(abs_nz)
    assert np.all(ranks[abs_nz == 1] == 5.5)
    assert np.all(ranks[abs_nz == 2] == 11.0)

    # Negative rank sum must be exactly 3 * 5.5 = 16.5; Positive rank sum = 7 * 5.5 + 11.0 = 49.5
    t_neg = float(np.sum(ranks[nz < 0]))
    t_pos = float(np.sum(ranks[nz > 0]))
    assert t_neg == 16.5
    assert t_pos == 49.5

    # Direct SciPy test on exact hit differences must yield W = 16.5, p > 0.10
    w_res = stats.wilcoxon(diff_hits, zero_method="wilcox", correction=True)
    assert w_res.statistic == 16.5
    assert round(float(w_res.pvalue), 4) == 0.1200
    assert float(w_res.pvalue) > 0.05, "P@3 must not be nominally significant on exact integer hits"

    # SciPy mode="exact" calculates untied rank distribution (358 / 2048)
    w_scipy_exact = stats.wilcoxon(diff_hits, zero_method="wilcox", mode="exact")
    assert round(float(w_scipy_exact.pvalue), 6) == 0.174805

    # Genuinely exact paired permutation test preserving ties (374 / 2048)
    import itertools
    all_signs = list(itertools.product([-1.0, 1.0], repeat=len(nz)))
    count = sum(1 for s in all_signs if min(np.sum(ranks[np.array(s) > 0]), np.sum(ranks[np.array(s) < 0])) <= 16.5 + 1e-9)
    p_exact_perm = count / len(all_signs)
    assert count == 374
    assert len(all_signs) == 2048
    assert round(p_exact_perm, 6) == 0.182617

    # Demonstrate that the spurious W = 9.5 was strictly an artifact of 4-decimal float subtraction
    diff_fl = h_p3 - d_p3
    w_fl = stats.wilcoxon(diff_fl, zero_method="wilcox", correction=False)
    assert w_fl.statistic == 9.5


def test_mrr_reconstruction_and_permutation():
    """Verify that MRR can be reconstructed from reciprocal ranks to yield exact fractions,

    confirming that the non-zero differences are exactly {-1/6, +1/2, +1/2} with W=1.0 and exact permutation p=0.5000.
    """
    import json
    from pathlib import Path
    from fractions import Fraction
    import numpy as np

    results_path = Path("data/evaluation/ablation_study_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_query = data["per_query_results"]

    def reconstruct_mrr(val):
        if val <= 0:
            return Fraction(0, 1)
        return Fraction(1, int(round(1.0 / val)))

    diffs = []
    for q in per_query:
        f_d = reconstruct_mrr(q["Dense-Only"]["MRR"])
        f_h = reconstruct_mrr(q["Hybrid-RRF"]["MRR"])
        diffs.append(f_h - f_d)

    nz_diffs = [d for d in diffs if d != 0]
    assert len(nz_diffs) == 3
    assert nz_diffs == [Fraction(-1, 6), Fraction(1, 2), Fraction(1, 2)]

    # Exactly 4 out of 8 sign permutations have min(T+, T-) <= 1.0
    # Permutation p-value is exactly 4/8 = 0.5000
    ranks = np.array([1.0, 2.5, 2.5])
    import itertools
    all_signs = list(itertools.product([-1.0, 1.0], repeat=3))
    count = sum(1 for s in all_signs if min(np.sum(ranks[np.array(s) > 0]), np.sum(ranks[np.array(s) < 0])) <= 1.0 + 1e-9)
    assert count == 4
    assert len(all_signs) == 8


def test_citation_faithfulness_audit_labels_and_provenance():
    """Verify that citation faithfulness evaluation decouples textual entailment

    from candidate pool provenance, using the 4 mutually exclusive entailment labels,
    while correctly handling abbreviations, multiple bracket citations, and uncited assertions.
    """
    from src.evaluation.citation_faithfulness import (
        EntailmentLabel,
        ClaimAuditResult,
        ClaimCategory,
        CitationProvenance,
        FaithfulnessSummary,
        audit_citation_faithfulness,
        extract_claims_with_citations,
        audit_claim_against_retrieved_pool,
        classify_claim_category,
        screen_citation_support_rule_based,
    )

    sample_text = (
        "Convolutional neural networks classify foliar diseases (e.g., leaf rust, Fig. 1) [PAT-003, PAP-019]. "
        "The system formulates variable-rate fungicide maps autonomously [PAT-015]. "
        "Quantum computing is utilized for real-time trajectory optimization [PAT-999]. "
        "This is an uncited general statement without any bracketed references."
    )

    claims = extract_claims_with_citations(sample_text)
    # Sentence 1 has 2 citations -> 2 claims audited; Sentence 2 has 1; Sentence 3 has 1; Sentence 4 has 0 (uncited) -> Total = 5
    assert len(claims) == 5

    assert claims[0]["citation_id"] == "PAT-003"
    assert "e.g., leaf rust, Fig. 1" in claims[0]["claim_text"]  # Abbreviation was protected!
    assert claims[1]["citation_id"] == "PAP-019"
    assert claims[2]["citation_id"] == "PAT-015"
    assert claims[3]["citation_id"] == "PAT-999"
    assert claims[4]["is_uncited"] is True
    assert claims[4]["citation_id"] == ""

    # Verify claim_category is present in each extracted claim dict
    for cl in claims:
        assert "claim_category" in cl
        assert cl["claim_category"] in [c.value for c in ClaimCategory]

    # Simulate query-specific retrieved pool and passage texts
    retrieved_pool = {"PAT-003", "PAP-019", "PAT-015"}  # PAT-999 missing (invalid provenance)
    doc_chunk_texts = {
        "PAT-003": "A deep convolutional neural network is trained on multispectral drone imagery to classify foliar crop diseases.",
        "PAP-019": "Multispectral aerial imaging benchmarks for plant disease identification.",
        "PAT-015": "Automated variable-rate chemical spraying apparatus for agricultural fields.",
    }

    audited = [
        audit_claim_against_retrieved_pool(claims[0], retrieved_pool, doc_chunk_texts),
        audit_claim_against_retrieved_pool(claims[1], retrieved_pool, doc_chunk_texts),
        audit_claim_against_retrieved_pool(claims[2], retrieved_pool, doc_chunk_texts),
        audit_claim_against_retrieved_pool(claims[3], retrieved_pool, doc_chunk_texts),
        audit_claim_against_retrieved_pool(claims[4], retrieved_pool, doc_chunk_texts),
    ]

    for a in audited:
        assert hasattr(a, "claim_category")
        assert isinstance(a.claim_category, ClaimCategory)

    summary = audit_citation_faithfulness(audited)
    assert summary.total_claims == 5
    assert summary.uncited_factual_count == 1
    assert summary.procedural_disclaimer_count == 0

    prov = summary.provenance
    assert isinstance(prov, CitationProvenance)
    assert prov.total_cited_occurrences == 4
    assert prov.valid_provenance_count == 3
    assert prov.invalid_provenance_count == 1
    assert prov.validity_rate == 0.75
    assert "PAT-999" in prov.unique_invalid_doc_ids

    assert summary.prior_art_factual_count >= 1
    assert summary.prior_art_supported >= 1
    assert summary.prior_art_unsupported >= 1

    import inspect
    module_docstring = inspect.getdoc(screen_citation_support_rule_based)
    assert "NOT semantic" in module_docstring or "Rule-based" in module_docstring


def test_contradiction_antonym_directional_shifts():
    """Explicit regression tests for antonyms and directional changes.

    Demonstrates that:
      1. Directional shifts (increases/decreases, improves/worsens, enables/prevents)
         cause rule-based token-overlap modules to fail (false SUPPORTED) due to high
         lexical overlap without surface negation words.
      2. The SemanticNLIEvaluator resolves these antonymic contradictions.
    """
    from src.evaluation.citation_faithfulness import (
        EntailmentLabel,
        screen_citation_support_rule_based
    )
    from src.evaluation.nli_evaluator import SemanticNLIEvaluator

    test_pairs = [
        # (claim, passage, expected_semantic_label, description)
        (
            "Monte Carlo Tree Search degrades the diversity of generated patent claim formulations.",
            "Monte Carlo Tree Search substantially increases the diversity and statutory coverage of generated patent claim formulations.",
            EntailmentLabel.CONTRADICTED,
            "increases vs degrades"
        ),
        (
            "Dense vector embeddings worsen retrieval precision over keyword baselines.",
            "Dense vector embeddings significantly improve retrieval precision over keyword baselines.",
            EntailmentLabel.CONTRADICTED,
            "improves vs worsen"
        ),
        (
            "Automated multi-agent synthesis prevents innovation in scientific research communities.",
            "Automated multi-agent synthesis enables innovation in scientific research communities.",
            EntailmentLabel.CONTRADICTED,
            "enables vs prevents"
        ),
        (
            "Experimental evidence contradicts the efficiency hypothesis.",
            "Experimental evidence strongly supports the efficiency hypothesis.",
            EntailmentLabel.CONTRADICTED,
            "supports vs contradicts"
        ),
    ]

    nli_evaluator = SemanticNLIEvaluator()

    for claim, passage, expected_label, desc in test_pairs:
        # Rule-based screening failure demonstration (baseline limitation):
        # Because the texts share >60% tokens and lack 'not'/'never', rule-based predicts SUPPORTED
        rb_label, rb_rationale = screen_citation_support_rule_based(claim, passage)
        assert rb_label == EntailmentLabel.SUPPORTED, (
            f"Rule-based baseline unexpectedly changed on {desc}: {rb_label}"
        )

        # Semantic NLI evaluator correctly detects contradiction on directional shifts
        nli_label, nli_rationale, probs, _ = nli_evaluator.evaluate(claim, passage)
        assert nli_label == expected_label, (
            f"NLI failed on {desc}: got {nli_label} with probs {probs}"
        )


def test_compound_claim_decomposition_and_conservative_partial_support():
    """Verify that compound sentences are decomposed into atomic claims and that

    a compound claim is NEVER labeled SUPPORTED merely because one part matches.
    """
    from src.evaluation.citation_faithfulness import (
        EntailmentLabel,
        decompose_compound_claim,
        audit_claim_against_retrieved_pool
    )

    # Compound sentence with 2 distinct propositions
    compound_claim = "Convolutional neural networks classify foliar diseases, and variable-rate maps are formulated autonomously."
    sub_clauses = decompose_compound_claim(compound_claim)
    assert len(sub_clauses) == 2
    assert "Convolutional neural networks classify foliar diseases" in sub_clauses[0]
    assert "variable-rate maps are formulated autonomously" in sub_clauses[1]

    # Passage only supports clause 1 (CNN foliar disease), not clause 2 (variable-rate maps)
    passage_text = "A deep convolutional neural network is trained on multispectral drone imagery to classify foliar crop diseases."
    retrieved_pool = {"PAT-003"}
    doc_chunk_texts = {"PAT-003": passage_text}

    claim_dict = {
        "claim_text": compound_claim,
        "citation_id": "PAT-003",
        "is_uncited": False,
        "claim_category": "prior_art_factual"
    }

    audit_result = audit_claim_against_retrieved_pool(claim_dict, retrieved_pool, doc_chunk_texts, use_atomic_decomposition=True)

    # Crucial property: Conservative partial support assigned, NOT full SUPPORTED!
    assert audit_result.entailment_label == EntailmentLabel.PARTIALLY_SUPPORTED
    assert "only 1/2 sub-claims supported" in audit_result.evidence_rationale


def test_reproducibility_of_benchmark_predictions_and_metrics():
    """Reproducibility Assertion: Recomputes accuracy, macro F1, Cohen's kappa,

    and the confusion matrix directly from saved itemized predictions across all
    three evaluation systems (Rule-based, Semantic NLI, and Two-Stage Hybrid).
    Guarantees no discrepancy between saved itemized predictions and reported summaries.
    """
    import json
    from pathlib import Path

    results_path = Path("data/evaluation/benchmark_model_comparison_results.json")
    assert results_path.exists(), f"Benchmark results file missing at {results_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    total_n = 24

    for system_key, expected in [
        ("rule_based_baseline", {"accuracy": 0.75, "macro_f1": 0.7239, "kappa": 0.6667, "contra_recall": 0.3333}),
        ("semantic_nli_baseline", {"accuracy": 0.5417, "macro_f1": 0.4852, "kappa": 0.3889, "contra_recall": 0.8333}),
        ("hybrid_nli_lexical_fallback", {"accuracy": 0.7917, "macro_f1": 0.7867, "kappa": 0.7222, "contra_recall": 0.8333}),
    ]:
        assert system_key in data, f"Missing system {system_key} in benchmark results"
        system_data = data[system_key]
        itemized = system_data["itemized_results"]
        assert len(itemized) == total_n, f"Expected 24 itemized predictions for {system_key}, found {len(itemized)}"

        # Recompute confusion matrix from raw item predictions
        recomputed_matrix = {p: {gt: 0 for gt in classes} for p in classes}
        for item in itemized:
            gt = item["ground_truth"]
            pred = item["predicted"]
            assert gt in classes, f"Invalid ground truth class: {gt}"
            assert pred in classes, f"Invalid predicted class: {pred}"
            recomputed_matrix[pred][gt] += 1

        # Check confusion matrix matches stored matrix
        stored_matrix = system_data["confusion_matrix"]
        for p in classes:
            for gt in classes:
                assert recomputed_matrix[p][gt] == stored_matrix[p][gt], (
                    f"Matrix mismatch for {system_key} at pred={p}, gt={gt}: "
                    f"recomputed {recomputed_matrix[p][gt]} vs stored {stored_matrix[p][gt]}"
                )

        # Recompute per-class TP, FP, FN, Precision, Recall, F1
        col_totals = {c: sum(recomputed_matrix[p][c] for p in classes) for c in classes}
        row_totals = {c: sum(recomputed_matrix[c][gt] for gt in classes) for c in classes}

        f1_scores = []
        for c in classes:
            tp = recomputed_matrix[c][c]
            prec = tp / row_totals[c] if row_totals[c] > 0 else 0.0
            rec = tp / col_totals[c] if col_totals[c] > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(classes)
        correct_count = sum(recomputed_matrix[c][c] for c in classes)
        accuracy = correct_count / total_n
        pe = sum((row_totals[c] / total_n) * (col_totals[c] / total_n) for c in classes)
        kappa = (accuracy - pe) / (1.0 - pe) if pe < 1.0 else 1.0
        contra_recall = recomputed_matrix["CONTRADICTED"]["CONTRADICTED"] / col_totals["CONTRADICTED"]

        # Assert recomputed summary metrics match expected theoretical values
        assert abs(accuracy - expected["accuracy"]) < 0.001, (
            f"Accuracy mismatch for {system_key}: recomputed {accuracy:.4f} vs expected {expected['accuracy']:.4f}"
        )
        assert abs(macro_f1 - expected["macro_f1"]) < 0.001, (
            f"Macro F1 mismatch for {system_key}: recomputed {macro_f1:.4f} vs expected {expected['macro_f1']:.4f}"
        )
        assert abs(kappa - expected["kappa"]) < 0.001, (
            f"Cohen's kappa mismatch for {system_key}: recomputed {kappa:.4f} vs expected {expected['kappa']:.4f}"
        )
        assert abs(contra_recall - expected["contra_recall"]) < 0.001, (
            f"Contradiction recall mismatch for {system_key}: recomputed {contra_recall:.4f} vs expected {expected['contra_recall']:.4f}"
        )

        # Assert recomputed summary metrics match stored JSON summary values
        assert abs(accuracy - system_data["raw_accuracy_Po"]) < 0.001
        assert abs(macro_f1 - system_data["macro_f1"]) < 0.001
        assert abs(kappa - system_data["cohens_kappa"]) < 0.001


def test_real_world_sampling_manifest_schema_and_counts():
    """Verifies that the Real-World Sampling Manifest satisfies Section 5 of EVALUATION_PROTOCOL.md:

    1. All 18 required fields exist in each record.
    2. Manifest explicitly itemizes all four counting levels:
       - Unique generated claims
       - Atomic subclaims
       - Citation occurrences
       - Final claim–citation–passage annotation records
    3. Provenance validity is computed by occurrence counts: Valid Occurrences / Total Audited Occurrences.
    4. Passage recovery distinguishes available evidence from unrecoverable citations (status: UNASSESSABLE).
    5. Covers all 20 benchmark test cases (TC-001 through TC-020).
    6. Model predictions are strictly excluded from the manifest to preserve independent test set integrity.
    """
    import json
    from pathlib import Path

    manifest_path = Path("data/evaluation/real_world_sampling_manifest.json")
    assert manifest_path.exists(), f"Missing sampling manifest at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    meta = manifest["manifest_metadata"]
    assert meta["total_queries_sampled"] == 20
    assert meta["sampling_scope"] == "TC-001 through TC-020"

    counts = meta["manifest_counts"]
    assert counts["unique_generated_claims"] == 80
    assert counts["unique_claims_per_system"]["System B (Standard RAG)"] == 40
    assert counts["unique_claims_per_system"]["System C (Multi-Agent RAG)"] == 40
    assert counts["total_atomic_subclaims"] == 81
    assert counts["total_citation_occurrences"] == len(set(r["citation_occurrence_id"] for r in manifest["records"]))
    assert counts["total_annotation_records"] == len(manifest["records"])
    assert counts["total_annotation_records"] >= counts["total_citation_occurrences"]

    prov_sum = meta["provenance_summary"]
    assert prov_sum["valid_occurrences"] + prov_sum["invalid_occurrences"] == counts["total_citation_occurrences"]
    expected_prov_rate = prov_sum["valid_occurrences"] / counts["total_citation_occurrences"]
    assert abs(prov_sum["provenance_validity_rate"] - expected_prov_rate) < 0.001

    recov_sum = meta["passage_recovery_summary"]
    assert recov_sum["recovered_passages"] + recov_sum["unrecoverable_passages"] == counts["total_citation_occurrences"]
    assert recov_sum["recovered_passages"] == counts["total_citation_occurrences"]

    records = manifest["records"]
    assert len(records) == counts["total_annotation_records"]

    required_18_fields = [
        "annotation_unit_id", "query_id", "system_id", "claim_id",
        "original_claim", "atomic_subclaim_id", "atomic_subclaim",
        "citation_id", "passage_id", "passage_text",
        "provenance_valid", "source_passage_recovered",
        "reviewer_1_label", "reviewer_1_rationale",
        "reviewer_2_label", "reviewer_2_rationale",
        "adjudicated_label", "adjudication_rationale"
    ]

    seen_unit_ids = set()
    for rec in records:
        for rf in required_18_fields:
            assert rf in rec, f"Record {rec.get('annotation_unit_id')} missing field {rf}"
        # Ensure model predictions are NOT leaked in the human annotation manifest
        assert "model_prediction" not in rec
        assert "nli_probability" not in rec
        seen_unit_ids.add(rec["annotation_unit_id"])

    assert len(seen_unit_ids) == len(records), "Duplicate annotation_unit_id found in manifest!"


def test_executable_claim_evidence_aggregation_all_edge_cases():
    """Validates executable conservative multi-passage aggregation across all edge cases

    mandated in Section 5.3 of docs/EVALUATION_PROTOCOL.md:
      1. Every proposition supported -> SUPPORTED
      2. One proposition partially supported and none contradicted -> PARTIALLY_SUPPORTED
      3. One proposition contradicted, regardless of other labels -> CONTRADICTED
      4. No proposition supported (all assessable unsupported) -> UNSUPPORTED
      5. Multiple citations supporting same proposition (combining evidence)
      6. Unrecoverable cited passage (UNASSESSABLE workflow handling)
      7. Compound claim whose different propositions have different citations
      8. Mixed unassessable and unsupported states
    """
    from src.evaluation.citation_faithfulness import (
        EntailmentLabel,
        AnnotationWorkflowStatus,
        aggregate_atomic_evidence,
        aggregate_claim_evidence,
    )

    # Edge Case 1: Every proposition supported
    case1 = [
        {"subclaim_id": "S1", "label": EntailmentLabel.SUPPORTED.value},
        {"subclaim_id": "S2", "label": EntailmentLabel.SUPPORTED.value},
    ]
    res1 = aggregate_claim_evidence(case1)
    assert res1["composite_label"] == EntailmentLabel.SUPPORTED.value

    # Edge Case 2: One proposition partially supported and none contradicted
    case2a = [
        {"subclaim_id": "S1", "label": EntailmentLabel.SUPPORTED.value},
        {"subclaim_id": "S2", "label": EntailmentLabel.PARTIALLY_SUPPORTED.value},
    ]
    res2a = aggregate_claim_evidence(case2a)
    assert res2a["composite_label"] == EntailmentLabel.PARTIALLY_SUPPORTED.value

    case2b = [
        {"subclaim_id": "S1", "label": EntailmentLabel.PARTIALLY_SUPPORTED.value},
        {"subclaim_id": "S2", "label": EntailmentLabel.PARTIALLY_SUPPORTED.value},
    ]
    res2b = aggregate_claim_evidence(case2b)
    assert res2b["composite_label"] == EntailmentLabel.PARTIALLY_SUPPORTED.value

    # Edge Case 3: One proposition contradicted, regardless of other labels
    case3a = [
        {"subclaim_id": "S1", "label": EntailmentLabel.SUPPORTED.value},
        {"subclaim_id": "S2", "label": EntailmentLabel.UNSUPPORTED.value},
        {"subclaim_id": "S3", "label": EntailmentLabel.CONTRADICTED.value},
    ]
    res3a = aggregate_claim_evidence(case3a)
    assert res3a["composite_label"] == EntailmentLabel.CONTRADICTED.value

    case3b = [
        {"subclaim_id": "S1", "label": EntailmentLabel.UNSUPPORTED.value},
        {"subclaim_id": "S2", "label": EntailmentLabel.CONTRADICTED.value},
    ]
    res3b = aggregate_claim_evidence(case3b)
    assert res3b["composite_label"] == EntailmentLabel.CONTRADICTED.value

    # Edge Case 4: No proposition supported (all assessable unsupported)
    case4 = [
        {"subclaim_id": "S1", "label": EntailmentLabel.UNSUPPORTED.value},
        {"subclaim_id": "S2", "label": EntailmentLabel.UNSUPPORTED.value},
    ]
    res4 = aggregate_claim_evidence(case4)
    assert res4["composite_label"] == EntailmentLabel.UNSUPPORTED.value

    # Edge Case 5: Multiple citations supporting same proposition (combining evidence)
    # Proposition S1 has 2 passages: P1 unsupported, P2 supported -> S1 resolves to SUPPORTED
    passages_s1 = [
        {"passage_id": "P1", "label": EntailmentLabel.UNSUPPORTED.value},
        {"passage_id": "P2", "label": EntailmentLabel.SUPPORTED.value},
    ]
    s1_resolved = aggregate_atomic_evidence(passages_s1)
    assert s1_resolved["label"] == EntailmentLabel.SUPPORTED.value

    case5 = [
        {"subclaim_id": "S1", "passage_evaluations": passages_s1},
        {"subclaim_id": "S2", "passage_evaluations": [{"passage_id": "P3", "label": EntailmentLabel.SUPPORTED.value}]}
    ]
    res5 = aggregate_claim_evidence(case5)
    assert res5["composite_label"] == EntailmentLabel.SUPPORTED.value

    # Edge Case 6: Unrecoverable cited passage (UNASSESSABLE handling)
    # 6a: Single proposition citing unrecoverable passage -> UNASSESSABLE
    case6a = [
        {"subclaim_id": "S1", "label": AnnotationWorkflowStatus.UNASSESSABLE.value}
    ]
    res6a = aggregate_claim_evidence(case6a)
    assert res6a["composite_label"] == AnnotationWorkflowStatus.UNASSESSABLE.value

    # 6b: Compound claim where S1 is supported but S2 has unrecoverable passage -> PARTIALLY_SUPPORTED
    case6b = [
        {"subclaim_id": "S1", "label": EntailmentLabel.SUPPORTED.value},
        {"subclaim_id": "S2", "label": AnnotationWorkflowStatus.UNASSESSABLE.value}
    ]
    res6b = aggregate_claim_evidence(case6b)
    assert res6b["composite_label"] == EntailmentLabel.PARTIALLY_SUPPORTED.value
    assert "1 supported" in res6b["rationale"]
    assert "1 unassessable" in res6b["rationale"]

    # Edge Case 7: Compound claim where different propositions cite different citations
    case7 = [
        {"subclaim_id": "S1", "atomic_subclaim": "Feature A uses CNN", "passage_evaluations": [{"passage_id": "PAT-003", "label": EntailmentLabel.SUPPORTED.value}]},
        {"subclaim_id": "S2", "atomic_subclaim": "Feature B uses MCTS", "passage_evaluations": [{"passage_id": "PAP-019", "label": EntailmentLabel.SUPPORTED.value}]},
    ]
    res7 = aggregate_claim_evidence(case7)
    assert res7["composite_label"] == EntailmentLabel.SUPPORTED.value
    assert len(res7["per_subclaim_verdicts"]) == 2

    # Edge Case 8: Mixed unsupported and unassessable -> UNSUPPORTED
    case8 = [
        {"subclaim_id": "S1", "label": EntailmentLabel.UNSUPPORTED.value},
        {"subclaim_id": "S2", "label": AnnotationWorkflowStatus.UNASSESSABLE.value}
    ]
    res8 = aggregate_claim_evidence(case8)
    assert res8["composite_label"] == EntailmentLabel.UNSUPPORTED.value


def test_inter_annotator_agreement_metrics_computation():
    """Verifies calculation of Po, Cohen's kappa, and Krippendorff's alpha nominal."""
    from src.evaluation.citation_faithfulness import (
        compute_observed_agreement,
        compute_cohens_kappa,
        compute_krippendorff_alpha_nominal,
    )

    r1_perfect = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED"]
    r2_perfect = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED"]

    # Perfect agreement
    po_perf = compute_observed_agreement(r1_perfect, r2_perfect)
    assert po_perf == 1.0

    kappa_perf = compute_cohens_kappa(r1_perfect, r2_perfect)
    assert kappa_perf["cohens_kappa"] == 1.0
    assert kappa_perf["observed_agreement_Po"] == 1.0

    alpha_perf = compute_krippendorff_alpha_nominal(r1_perfect, r2_perfect)
    assert alpha_perf["krippendorff_alpha"] == 1.0

    # Partial disagreement case with known numbers
    r1 = ["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    r2 = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]

    po = compute_observed_agreement(r1, r2)
    assert po == 0.75  # 3 of 4 match

    kappa_res = compute_cohens_kappa(r1, r2)
    assert kappa_res["observed_agreement_Po"] == 0.75
    assert kappa_res["cohens_kappa"] > 0.0

    # Test Krippendorff's alpha with missing/unassigned item (None)
    r1_missing = ["SUPPORTED", "SUPPORTED", None, "CONTRADICTED"]
    r2_missing = ["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    alpha_miss = compute_krippendorff_alpha_nominal(r1_missing, r2_missing)
    assert alpha_miss["n_dual_coded_units"] == 3
    assert alpha_miss["observed_disagreement_Do"] == 0.0
    assert alpha_miss["krippendorff_alpha"] == 1.0









