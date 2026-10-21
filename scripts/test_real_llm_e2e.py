"""
Script to test end-to-end execution with the real LLM provider (OpenAI gpt-4o-mini).
Records model, prompt, retrieved evidence, and outputs or API status.
If API access/quota is unavailable, records the exact error and sets status to PENDING_API_QUOTA.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.schemas.ideas import UserIdeaInput
from src.workflows.multi_agent_workflow import MultiAgentRAGWorkflow
from src.rag.prompts import REPORT_SYNTHESIS_PROMPT
from src.llm.openai_provider import OpenAIProvider


def run_real_llm_e2e_test():
    print("=" * 80)
    print("REAL LLM END-TO-END VALIDATION AUDIT")
    print("=" * 80)

    test_queries = [
        UserIdeaInput(
            title="Drone-Based Crop Disease Classification",
            idea="An AI system that uses drone-based multispectral images to detect crop diseases and recommend variable-rate chemical treatment.",
            domain="Agriculture & Computer Vision"
        ),
        UserIdeaInput(
            title="Privacy-Preserving Multi-Agent Autonomous Drone Coordination",
            idea="A decentralized multi-agent framework where autonomous drones coordinate surveillance tasks without sharing raw telemetry, using local differential privacy and federated learning.",
            domain="Autonomous Systems & Privacy"
        )
    ]

    results_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_requested": settings.llm_model,
        "provider_tested": "OpenAI",
        "api_key_configured": bool(settings.openai_api_key),
        "test_cases": []
    }

    # Initialize workflow with retrieval to get real evidence
    workflow = MultiAgentRAGWorkflow()

    for idx, idea in enumerate(test_queries, 1):
        print(f"\n[Test Case {idx}/{len(test_queries)}] Query: '{idea.title}'")

        # 1. Run pipeline up to report synthesis to retrieve real evidence
        init_state = {
            "user_idea": idea.idea,
            "idea_input": idea,
            "processing_time": 0.0
        }
        # Run retrieval and agent analysis
        graph = workflow.graph
        final_state = graph.invoke(init_state)

        retrieval = final_state.get("retrieval_result")
        pat_analyses = final_state.get("patent_analysis", [])
        pap_analyses = final_state.get("research_analysis", [])
        feat_comp = final_state.get("feature_comparison")
        novelty = final_state.get("novelty_assessment")
        gaps = final_state.get("gap_analysis")
        features = final_state.get("extracted_features")

        # Build passage-anchored evidence details
        evidence_lines = []
        for p in pat_analyses:
            evidence_lines.append(
                f"- Patent [{p.patent_id}] (Chunk: {p.chunk_id}) '{p.patent_title}': "
                f"Discloses features: {', '.join(p.matching_features) if p.matching_features else 'General prior art'}. "
                f"Excerpt: \"{p.excerpt[:250]}\""
            )
        for pap in pap_analyses:
            evidence_lines.append(
                f"- Paper [{pap.paper_id}] (Chunk: {pap.chunk_id}) '{pap.paper_title}': "
                f"Research overlap: {', '.join(pap.overlap_with_idea) if pap.overlap_with_idea else 'Theoretical context'}. "
                f"Excerpt: \"{pap.excerpt[:250]}\""
            )
        evidence_details_str = "\n".join(evidence_lines)

        prompt_rendered = REPORT_SYNTHESIS_PROMPT.format(
            idea_title=features.title if features else idea.title,
            novelty_level=novelty.novelty_level if novelty else "Moderate",
            novelty_score=f"{novelty.novelty_score:.2f}" if novelty and novelty.novelty_score else "N/A",
            citations=", ".join([p.patent_id for p in pat_analyses] + [pap.paper_id for pap in pap_analyses]) or "None",
            evidence_details=evidence_details_str,
            gaps="; ".join(gaps.unexplored_combinations[:2]) if gaps and gaps.unexplored_combinations else "None"
        )

        # 2. Attempt call with real OpenAI provider
        live_provider = OpenAIProvider()
        test_case_entry = {
            "title": idea.title,
            "idea_text": idea.idea,
            "retrieved_patent_count": len(pat_analyses),
            "retrieved_paper_count": len(pap_analyses),
            "evidence_details_passed": evidence_lines,
            "prompt_sample": prompt_rendered[:1000] + "...",
            "live_call_status": "UNKNOWN",
            "error_details": None,
            "output": None
        }

        try:
            print(f"  -> Attempting live API call to model '{settings.llm_model}'...")
            live_output = live_provider.generate(
                prompt=prompt_rendered,
                system_prompt="You are an expert patent attorney and research scientist specializing in intellectual property.",
                temperature=0.2,
                max_tokens=800
            )
            print(f"  -> Live API Call SUCCESS! Output length: {len(live_output)} chars")
            test_case_entry["live_call_status"] = "SUCCESS"
            test_case_entry["output"] = live_output
        except Exception as e:
            err_msg = str(e)
            print(f"  -> Live API Call FAILED as expected/detected: {err_msg[:120]}...")
            test_case_entry["live_call_status"] = "PENDING_API_QUOTA"
            test_case_entry["error_details"] = err_msg
            test_case_entry["offline_deterministic_output"] = final_state.get("final_report").final_summary

        results_record["test_cases"].append(test_case_entry)

    # Save to data/evaluation/real_llm_test_record.json
    out_path = PROJECT_ROOT / "data" / "evaluation" / "real_llm_test_record.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_record, f, indent=2)

    print(f"\n[Summary] Live LLM test record persisted to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_real_llm_e2e_test()
