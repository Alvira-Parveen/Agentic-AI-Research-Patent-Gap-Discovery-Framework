import re
from typing import List, Set, Dict, Any


def evaluate_explainability(
    cited_evidence: List[str],
    retrieved_doc_ids: Set[str]
) -> Dict[str, Any]:
    """
    Checks citation validity and hallucination rate:
    Verifies if cited document references actually correspond to retrieved prior-art items.
    """
    if not cited_evidence:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "grounding_ratio": 0.0,
            "hallucinated_citations": []
        }

    valid_count = 0
    hallucinated = []

    for cit in cited_evidence:
        # Extract doc ID inside brackets e.g. [PAT-001, Claim 1] -> PAT-001
        match = re.search(r'\[\s*(PAT-\d+|PAP-\d+)', cit)
        if match:
            doc_id = match.group(1)
            if doc_id in retrieved_doc_ids:
                valid_count += 1
            else:
                hallucinated.append(cit)
        else:
            # Citation format missing proper ID
            hallucinated.append(cit)

    grounding_ratio = round(valid_count / len(cited_evidence), 4) if cited_evidence else 0.0

    return {
        "total_citations": len(cited_evidence),
        "valid_citations": valid_count,
        "grounding_ratio": grounding_ratio,
        "hallucinated_citations": hallucinated
    }
