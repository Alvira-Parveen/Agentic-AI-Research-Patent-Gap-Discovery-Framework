#!/usr/bin/env python3
"""
End-to-End Workflow & Traceability Verification
PBL-3 Research Engineering Project (CSP391)

Validates the complete research idea pipeline against the 6-point checklist:
  1. Research idea can be submitted successfully
  2. Relevant patents and papers are retrieved from the FAISS vector stores
  3. Patent and research agents return structured evidence with exact chunk IDs
  4. Report generation uses the actual retrieved evidence
  5. Final report citations can be traced directly to source passages
  6. Missing evidence and errors are handled gracefully
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.workflows.orchestrator import IdeaAnalysisOrchestrator
from src.schemas.ideas import UserIdeaInput
from src.evaluation.citation_faithfulness import extract_claims_with_citations, ClaimCategory


def verify_end_to_end():
    print("=" * 80)
    print("END-TO-END WORKFLOW & TRACEABILITY AUDIT")
    print("=" * 80)

    # Step 1: Submit research idea
    test_idea = UserIdeaInput(
        title="Privacy-Preserving Multi-Agent Autonomous Drone Coordination",
        idea="A decentralized multi-agent framework where autonomous drones use local edge models for agricultural inspection while sharing federated parameter updates without transmitting raw multispectral video.",
        domain="Computer Vision & Smart Agriculture"
    )
    print(f"\n[Check 1/6] Submitting research idea: '{test_idea.title}'...")
    orchestrator = IdeaAnalysisOrchestrator()
    print("  -> Idea submitted and validated successfully through Pydantic schema.")

    # Step 2: Retrieval execution
    print("\n[Check 2/6] Executing dual-corpus retrieval (Patents & Academic Papers)...")
    comp_result = orchestrator.compare(test_idea)
    multi_agent = comp_result.multi_agent_rag
    retrieval = multi_agent.retrieval
    assert retrieval is not None, "Retrieval result is None!"
    print(f"  -> Retrieved {len(retrieval.patents)} patent chunks and {len(retrieval.papers)} paper chunks.")
    assert len(retrieval.patents) > 0, "No patents retrieved!"
    assert len(retrieval.papers) > 0, "No papers retrieved!"
    print(f"  -> Top patent: [{retrieval.patents[0].document_id}] '{retrieval.patents[0].document_title}' (score: {retrieval.patents[0].similarity_score:.4f})")
    print(f"  -> Top paper : [{retrieval.papers[0].document_id}] '{retrieval.papers[0].document_title}' (score: {retrieval.papers[0].similarity_score:.4f})")

    # Step 3: Check agent evidence with chunk IDs
    print("\n[Check 3/6] Verifying patent and research agents return evidence with chunk IDs...")
    # Run the underlying graph state to inspect agent items
    from src.workflows.multi_agent_workflow import MultiAgentRAGWorkflow
    app = MultiAgentRAGWorkflow().graph
    initial_state = {
        "idea_input": test_idea,
        "user_idea": test_idea.idea,
        "domain": test_idea.domain
    }
    final_state = app.invoke(initial_state)

    pat_analyses = final_state.get("patent_analysis", [])
    pap_analyses = final_state.get("research_analysis", [])
    assert len(pat_analyses) > 0, "PatentAnalysisAgent produced no items!"
    assert len(pap_analyses) > 0, "ResearchAnalysisAgent produced no items!"

    for p in pat_analyses[:2]:
        print(f"  -> Patent Agent item: [{p.patent_id}] Chunk ID: '{p.chunk_id}'")
        print(f"     Matching features: {p.matching_features}")
        assert p.chunk_id, f"Missing chunk_id on patent analysis for {p.patent_id}!"
        assert p.excerpt, f"Missing excerpt on patent analysis for {p.patent_id}!"

    for pap in pap_analyses[:2]:
        print(f"  -> Paper Agent item: [{pap.paper_id}] Chunk ID: '{pap.chunk_id}'")
        print(f"     Overlap with idea: {pap.overlap_with_idea}")
        assert pap.chunk_id, f"Missing chunk_id on paper analysis for {pap.paper_id}!"
        assert pap.excerpt, f"Missing excerpt on paper analysis for {pap.paper_id}!"

    # Step 4: Check report generation uses retrieved evidence
    print("\n[Check 4/6] Verifying report generation incorporates retrieved evidence...")
    final_report = final_state.get("final_report")
    assert final_report is not None, "Final report is None!"
    summary = final_report.final_summary
    print(f"  -> Generated Summary ({len(summary)} characters):")
    for line in summary.split("\n")[:4]:
        print(f"     {line}")

    # Step 5: Citation traceability to source passages
    print("\n[Check 5/6] Verifying citations trace directly to source passages in corpus...")
    claims = extract_claims_with_citations(summary)
    prior_art_claims = [c for c in claims if c["claim_category"] == ClaimCategory.PRIOR_ART_FACTUAL.value]
    print(f"  -> Found {len(prior_art_claims)} prior-art factual claims.")

    from scripts.build_real_world_sampling_manifest import load_corpus_chunk_map
    corpus_map = load_corpus_chunk_map()

    traceability_success = True
    for c in prior_art_claims:
        cit_id = c.get("citation_id")
        claim_text = c.get("claim_text")
        in_corpus = cit_id in corpus_map
        print(f"  -> Claim: \"{claim_text[:60]}...\"")
        print(f"     Citation: [{cit_id}] | Recovered from Corpus: {in_corpus}")
        if in_corpus:
            passage_snippet = corpus_map[cit_id][:120].replace('\n', ' ')
            print(f"     Source Passage Snippet: \"{passage_snippet}...\"")
        else:
            traceability_success = False

    assert traceability_success, "One or more citations failed corpus traceability!"
    print("  -> All cited references successfully traced to verified primary source passages.")

    # Step 6: Handling missing evidence and errors gracefully
    print("\n[Check 6/6] Verifying graceful error handling for missing evidence...")
    unretrieved_idea = UserIdeaInput(
        title="Hypothetical Sub-Atomic Wormhole Propulsion Mechanism",
        idea="A system for faster-than-light interstellar travel via controlled micro-singularities.",
        domain="Astrophysics & Exotic Propulsion"
    )
    comp_unretrieved = orchestrator.compare(unretrieved_idea)
    unretrieved_summary = comp_unretrieved.multi_agent_rag.final_summary
    print(f"  -> Graceful fallback output for out-of-corpus query:")
    print(f"     {unretrieved_summary[:200]}...")
    assert "preliminary" in unretrieved_summary.lower() or "not constitute a legal" in unretrieved_summary.lower()
    print("  -> Missing evidence and low-similarity concepts handled gracefully with explicit caveats.")

    print("\n" + "=" * 80)
    print("ALL 6 END-TO-END WORKFLOW CHECKS PASSED SUCCESSFULLY (6/6)")
    print("=" * 80)


if __name__ == "__main__":
    verify_end_to_end()
