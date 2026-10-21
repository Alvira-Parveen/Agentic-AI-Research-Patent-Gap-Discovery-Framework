"""
Semantic NLI Evaluator for Citation Faithfulness & Factual Verification
PBL-3 Research Engineering Project (CSP391)

This module implements a semantic Natural Language Inference (NLI) baseline
using a pretrained cross-encoder transformer model (distilbert-base-uncased-mnli).
It evaluates whether a retrieved premise passage logically entails, contradicts,
or is neutral toward a generated claim.

Label Mapping (MNLI 3-class -> Citation Evaluation 4-class):
  - ENTAILMENT (p >= 0.50)           -> SUPPORTED
  - CONTRADICTION (p >= 0.35 & > ent) -> CONTRADICTED
  - NEUTRAL (with partial overlap)   -> PARTIALLY_SUPPORTED
  - NEUTRAL (without overlap)        -> UNSUPPORTED
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.evaluation.citation_faithfulness import EntailmentLabel

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "distilbert-base-uncased-mnli"


class SemanticNLIEvaluator:
    """Pretrained NLI evaluator for citation factual support and contradiction detection."""

    def __init__(self, model_path: Optional[str] = None):
        path = str(model_path or DEFAULT_MODEL_DIR)
        self.model_name = "typeform/distilbert-base-uncased-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def evaluate(self, claim_text: str, passage_text: str) -> Tuple[EntailmentLabel, str, Dict[str, float], float]:
        """Evaluates semantic entailment of a claim against a premise passage.

        Args:
            claim_text: Hypothesis statement
            passage_text: Premise passage from retrieved source

        Returns:
            Tuple of (EntailmentLabel, evidence_rationale, probability_dict, latency_ms)
        """
        if not passage_text or not passage_text.strip():
            return EntailmentLabel.UNSUPPORTED, "Cited passage is empty or unretrieved.", {}, 0.0

        t0 = time.perf_counter()
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

        latency_ms = (time.perf_counter() - t0) * 1000

        p_entail = probs[0].item()
        p_neutral = probs[1].item()
        p_contra = probs[2].item()

        prob_dict = {
            "entailment": round(p_entail, 4),
            "neutral": round(p_neutral, 4),
            "contradiction": round(p_contra, 4)
        }

        # Deterministic label mapping
        if p_contra > p_entail and p_contra >= 0.35:
            label = EntailmentLabel.CONTRADICTED
            rationale = f"Semantic NLI contradiction detected (P(contra)={p_contra:.1%}, P(entail)={p_entail:.1%})."
        elif p_entail >= 0.50:
            label = EntailmentLabel.SUPPORTED
            rationale = f"Semantic NLI entailment verified (P(entail)={p_entail:.1%})."
        elif p_entail >= 0.15 or (p_neutral >= 0.50 and p_entail >= 0.05):
            label = EntailmentLabel.PARTIALLY_SUPPORTED
            rationale = f"Semantic NLI partial alignment (P(entail)={p_entail:.1%}, P(neutral)={p_neutral:.1%})."
        else:
            label = EntailmentLabel.UNSUPPORTED
            rationale = f"Semantic NLI unsupported: premise does not entail claim (P(neutral)={p_neutral:.1%})."

        return label, rationale, prob_dict, latency_ms
