import json
import re
from typing import Optional, Type, TypeVar
from src.llm.base import BaseLLMProvider
from src.utils.logging import logger

T = TypeVar("T")


class MockProvider(BaseLLMProvider):
    """
    Deterministic Mock LLM Provider for offline testing, CI/CD,
    and cost-controlled reproducible evaluation without requiring an API key.
    """
    def __init__(self, model_name: str = "mock-gpt-4o"):
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        prompt_lower = prompt.lower()

        # 1. Patent Analysis Prompt (checked first to avoid matching embedded 'feature' words)
        if "retrieved patent:" in prompt_lower or "analyze the retrieved patent" in prompt_lower:
            pat_match = re.search(r'\[(PAT-\d+)\]', prompt)
            pat_id = pat_match.group(1) if pat_match else "PAT-001"
            sim_match = re.search(r'Similarity:\s*([0-9.]+)', prompt)
            sim = float(sim_match.group(1)) if sim_match else 0.75

            return json.dumps({
                "patent_id": pat_id,
                "patent_title": f"Prior Art Specification for {pat_id}",
                "similarity_score": sim,
                "technical_problem": "Addressing computational latency and high false-positive rates in automated detection pipelines.",
                "proposed_solution": "A multi-stage contextual indexing architecture with feature vector aggregation.",
                "matching_features": ["Vector feature embedding", "Multi-stage data filtering", "Contextual metadata association"],
                "different_features": ["Proprietary hardware acceleration layer", "Static rule-based post-filtering"],
                "claim_relevance": f"Independent Claim 1 of {pat_id} recites vector similarity matching, but does not disclose cross-domain agent orchestration.",
                "evidence": [f"[{pat_id}, Claim 1]", f"[{pat_id}, Description]"],
                "analysis": f"The disclosure in {pat_id} establishes significant prior art regarding feature representations, but lacks the specific agentic gap resolution proposed."
            })

        # 2. Research Analysis Prompt
        if "retrieved research paper:" in prompt_lower or "analyze the retrieved scientific research" in prompt_lower or "retrieved paper excerpt:" in prompt_lower:
            pap_match = re.search(r'\[(PAP-\d+)\]', prompt)
            pap_id = pap_match.group(1) if pap_match else "PAP-001"
            sim_match = re.search(r'Similarity:\s*([0-9.]+)', prompt)
            sim = float(sim_match.group(1)) if sim_match else 0.70

            return json.dumps({
                "paper_id": pap_id,
                "paper_title": f"Academic Research Study for {pap_id}",
                "similarity_score": sim,
                "problem_addressed": "Investigating empirical limits of single-model reasoning on multi-source knowledge integration.",
                "methodology": "Comparative empirical benchmark using dense contrastive embeddings and transformer architectures.",
                "findings": "Dense vector retrieval coupled with structured reasoning yields a 14% improvement in retrieval recall over keyword baselines.",
                "limitations": "The study was restricted to closed academic benchmarks and did not evaluate commercial patent claim boundaries.",
                "overlap_with_idea": ["Semantic embedding alignment", "Retrieval-augmented generation grounding"],
                "differences": ["Focuses purely on scientific papers without patent claim scope mapping"],
                "evidence": [f"[{pap_id}, Abstract]", f"[{pap_id}, Methodology]"],
                "analysis": f"{pap_id} provides solid academic precedent for semantic retrieval, yet leaves open the intersection with patent white-space discovery."
            })

        # 3. Gap Discovery Prompt
        if "identify potential patent and research gaps" in prompt_lower or "analyzed patents summary:" in prompt_lower:
            return json.dumps({
                "well_covered_areas": ["Vector embedding search", "Standard convolutional neural network backbones"],
                "partially_covered_areas": ["Automated drone image preprocessing and geo-spatial normalization"],
                "underrepresented_features": ["End-to-end multi-agent feedback loop between edge detection and prescription generation"],
                "unexplored_combinations": ["Hybrid CNN-transformer architectures combined with decentralized drone task scheduling"],
                "patent_vs_paper_differences": "Patents emphasize data packaging and sensor transmission hardware claims, whereas research papers focus on empirical benchmark accuracy and loss functions.",
                "potential_research_directions": ["Cross-dataset domain adaptation for seasonal crop variations", "Lightweight edge-quantized multi-agent inference pipelines"],
                "evidence_citations": ["[PAT-016, Description]", "[PAP-003, Abstract]"]
            })

        # 4. Feature Extraction Prompt
        if "extract the technical architecture" in prompt_lower or "extract technical features" in prompt_lower or "technical_features" in prompt_lower:
            idea_match = re.search(r'Idea:\s*([^\n]+)', prompt, re.IGNORECASE)
            idea_text = idea_match.group(1) if idea_match else "AI Invention"
            words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', idea_text) if w.lower() not in ["with", "that", "this", "from", "using"]]
            
            return json.dumps({
                "title": f"Novel Concept: {idea_text[:60]}",
                "description": idea_text,
                "domain": "Artificial Intelligence & Applied Machine Learning",
                "technologies": words[:4] or ["Machine Learning", "Neural Networks"],
                "technical_features": [f"{w.capitalize()} processing mechanism" for w in words[:4]] or ["Automated feature analysis", "Real-time decision pipeline"],
                "methods": ["Deep Learning", "Contextual Embedding Search", "Statistical Classification"],
                "inputs": ["Multispectral sensor stream", "Structured contextual metadata"],
                "outputs": ["Anomaly detection score", "Automated recommendation action"],
                "key_components": ["Ingestion Subsystem", "Embedding Engine", "Decision Agent"]
            })

        # 5. Novelty Assessment Prompt
        if "novelty" in prompt_lower:
            return json.dumps({
                "novelty_level": "Moderate",
                "novelty_score": 0.62,
                "confidence": 0.81,
                "factors": {
                    "semantic_similarity": 0.68,
                    "technical_overlap": 0.55,
                    "claim_feature_coverage": 0.45,
                    "evidence_strength": 0.78
                },
                "formula_explanation": "Novelty = 1.0 - (0.40*sim + 0.35*overlap + 0.25*claim_coverage) * confidence_scaling",
                "supporting_evidence": ["[PAT-001, Claim 1]", "[PAP-002, Abstract]"],
                "potential_risks": ["Prior art in PAT-001 closely discloses vector search mechanisms."],
                "limitations": ["Corpus is limited to seed PBL-3 dataset (20 patents, 20 papers)."],
                "legal_disclaimer": "PRELIMINARY AI RESEARCH ASSESSMENT ONLY: Not a legal patentability opinion."
            })

        # 5. Generic Summary
        return (
            "The proposed invention demonstrates moderate preliminary novelty. "
            "While core embedding and retrieval elements are well established in prior art (e.g. PAT-001, PAT-002), "
            "the specialized multi-agent gap discovery pipeline exhibits underrepresented technical coverage in the current corpus."
        )
