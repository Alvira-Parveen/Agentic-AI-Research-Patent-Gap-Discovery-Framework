"""
Hybrid Citation Faithfulness Evaluator: Semantic NLI with Lexical Overlap Fallback
PBL-3 Research Engineering Project (CSP391)

This module implements the two-stage hybrid evaluation method that combines:
  Stage 1: Pretrained Semantic NLI (distilbert-base-uncased-mnli) for directional
           contradictions and strict entailment.
  Stage 2: Rule-Based Lexical Screening Fallback to disambiguate neutral NLI
           predictions into partial grounding versus ungrounded text.

METHODOLOGICAL NOTE:
The partial-support classification is an explicitly defined heuristic decision rule
combining NLI neutrality with >=30% keyword overlap. Lexical overlap does not prove
that a claim is true, and related technical terminology can inflate overlap scores.
This must be reported as a heuristic screening policy, not a semantic guarantee.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.evaluation.citation_faithfulness import (
    EntailmentLabel,
    screen_citation_support_rule_based,
    decompose_compound_claim
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "distilbert-base-uncased-mnli"


class HybridNLILexicalEvaluator:
    """Two-Stage Hybrid Evaluator combining Semantic NLI with Lexical Overlap Fallback."""

    def __init__(self, model_path: Optional[str] = None):
        path = str(model_path or DEFAULT_MODEL_DIR)
        self.model_name = "typeform/distilbert-base-uncased-mnli (with lexical overlap fallback)"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def evaluate(
        self,
        claim_text: str,
        passage_text: str,
        decompose_compound: bool = False
    ) -> Tuple[EntailmentLabel, str, Dict[str, Any], float]:
        """Evaluates a claim against a premise passage using the two-stage hybrid policy.

        Args:
            claim_text: Claim hypothesis string.
            passage_text: Retrieved source passage premise string.
            decompose_compound: If True, decomposes multi-clause claims into atomic propositions.

        Returns:
            Tuple of (EntailmentLabel, evidence_rationale, metadata_dict, latency_ms)
        """
        if not passage_text or not passage_text.strip():
            return EntailmentLabel.UNSUPPORTED, "Cited passage is empty or unretrieved.", {}, 0.0

        t0 = time.perf_counter()

        # Optional atomic decomposition for multi-clause compound sentences
        sub_claims = decompose_compound_claim(claim_text) if decompose_compound else [claim_text]

        # Stage 1: Semantic NLI Inference on the claim
        inputs = self.tokenizer(
            passage_text,
            claim_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        p_entail = probs[0].item()
        p_neutral = probs[1].item()
        p_contra = probs[2].item()

        # Stage 2: Lexical Screening Fallback
        rb_label, rb_rationale = screen_citation_support_rule_based(claim_text, passage_text)

        latency_ms = (time.perf_counter() - t0) * 1000

        meta = {
            "nli_probs": {
                "entailment": round(p_entail, 4),
                "neutral": round(p_neutral, 4),
                "contradiction": round(p_contra, 4)
            },
            "rule_based_label": rb_label.value,
            "rule_based_rationale": rb_rationale,
            "sub_claims_count": len(sub_claims)
        }

        # ---------------------------------------------------------------------
        # Pre-Defined Hybrid Decision Rules
        # ---------------------------------------------------------------------
        # Rule 1: Contradiction Detection via NLI (fixes rule-based antonym blindspot)
        if p_contra > p_entail and p_contra >= 0.35:
            label = EntailmentLabel.CONTRADICTED
            rationale = (
                f"Hybrid Stage 1: Semantic NLI detected directional contradiction "
                f"(P(contra)={p_contra:.1%}, P(entail)={p_entail:.1%})."
            )
        # Rule 2: Strong Entailment via NLI
        elif p_entail >= 0.50:
            label = EntailmentLabel.SUPPORTED
            rationale = (
                f"Hybrid Stage 1: Semantic NLI confirmed entailment (P(entail)={p_entail:.1%})."
            )
        # Rule 3 & 4: NLI is Neutral -> Fallback to Lexical Overlap
        else:
            if rb_label in (EntailmentLabel.SUPPORTED, EntailmentLabel.PARTIALLY_SUPPORTED):
                label = EntailmentLabel.PARTIALLY_SUPPORTED
                rationale = (
                    f"Hybrid Stage 2 Fallback: NLI is neutral (P(neutral)={p_neutral:.1%}) "
                    f"while lexical screening detected partial topical overlap ({rb_label.value}). "
                    f"Note: Assigned via heuristic decision rule, not formal semantic guarantee."
                )
            else:
                label = EntailmentLabel.UNSUPPORTED
                rationale = (
                    f"Hybrid Stage 2 Fallback: NLI is neutral (P(neutral)={p_neutral:.1%}) "
                    f"and lexical screening confirmed insufficient token overlap (< 30%)."
                )

        return label, rationale, meta, latency_ms
