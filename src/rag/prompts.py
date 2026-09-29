"""
Centralized Prompt Library for PBL-3 Patent & Research Gap Discovery.
Strictly enforces evidence grounding, hallucination control, and structured JSON output.
"""

SYSTEM_STRICT_EVIDENCE = """You are a rigorous, academic research AI assistant specializing in patent and scientific literature analysis.
CRITICAL RULES:
1. Ground your reasoning strictly on the provided retrieved excerpts.
2. If information is not in the text, write "Not available in the retrieved document." Do not guess or fabricate patent numbers, authors, or claim terms.
3. Distinguish clearly between facts directly stated in the evidence and your inferences.
4. Always cite specific evidence using standard citation brackets, e.g. [PAT-001, Claim 1] or [PAP-002, Abstract].
5. Never invent numeric similarity scores.
6. Use cautious research language: "Preliminary assessment suggests...", "Potential gap based on retrieved evidence...".
7. NEVER state that an idea is legally patentable, that a patent will be granted, or that no prior art exists.
8. Absence of a feature in retrieved excerpts indicates technical underrepresentation in the local corpus, NOT global absence or proof of novelty.
"""

FEATURE_EXTRACTION_PROMPT = """Extract the technical architecture and features from the following invention/research idea.

Idea: {idea}

Respond with a JSON object strictly adhering to this schema:
{{
  "title": "Concise technical title",
  "description": "Technical description of the proposed concept",
  "domain": "Primary domain (e.g. AI/ML, Computer Vision, Agriculture)",
  "technologies": ["Core technology 1", "Core technology 2"],
  "technical_features": ["Specific feature mechanism 1", "Specific feature mechanism 2"],
  "methods": ["Algorithm or workflow 1", "Algorithm or workflow 2"],
  "inputs": ["Input data type 1", "Input sensor 2"],
  "outputs": ["Output result 1", "Output artifact 2"],
  "key_components": ["Subsystem 1", "Subsystem 2"]
}}
"""

PATENT_ANALYSIS_PROMPT = """Analyze the retrieved patent against the proposed invention idea.

Proposed Idea Features:
{idea_features}

Retrieved Patent:
Identifier: {doc_id}
Title: {title}
Deterministic Cosine Similarity: {similarity_score}
Citation: {citation}
Retrieved Patent Excerpt:
\"\"\"{excerpt}\"\"\"

Analyze the technical overlap. Output a JSON object adhering to this schema:
{{
  "patent_id": "{doc_id}",
  "patent_title": "{title}",
  "similarity_score": {similarity_score},
  "technical_problem": "Problem the patent addresses",
  "proposed_solution": "Solution disclosed in the patent excerpt",
  "matching_features": ["Feature 1 present in both idea and patent", "Feature 2"],
  "different_features": ["Feature in patent not in idea", "Feature in idea not in patent"],
  "claim_relevance": "How the patent claims relate to the proposed concept",
  "evidence": ["{citation}"],
  "analysis": "2-3 sentences of technical comparison grounded in the excerpt."
}}
"""

RESEARCH_ANALYSIS_PROMPT = """Analyze the retrieved scientific research paper against the proposed invention idea.

Proposed Idea Features:
{idea_features}

Retrieved Research Paper:
Identifier: {doc_id}
Title: {title}
Deterministic Cosine Similarity: {similarity_score}
Citation: {citation}
Retrieved Paper Excerpt:
\"\"\"{excerpt}\"\"\"

Analyze the scientific overlap. Output a JSON object adhering to this schema:
{{
  "paper_id": "{doc_id}",
  "paper_title": "{title}",
  "similarity_score": {similarity_score},
  "problem_addressed": "Scientific question addressed",
  "methodology": "Methodology / algorithm used in the paper",
  "findings": "Key findings or empirical results",
  "limitations": "Limitations reported or apparent from excerpt",
  "overlap_with_idea": ["Common scientific concepts"],
  "differences": ["Distinct scientific methodologies"],
  "evidence": ["{citation}"],
  "analysis": "2-3 sentences of academic comparison grounded in the excerpt."
}}
"""

SINGLE_LLM_PROMPT = """Evaluate the following invention/research idea without external prior art retrieval.

Idea: {idea}

Provide a structured preliminary analysis:
1. Technical feature breakdown
2. Likely existing prior art based on your general knowledge
3. Potential novelty assessment (use cautious language: "Preliminary assessment...")
4. Potential gaps or unexplored opportunities
5. Limitations of assessing without retrieval evidence
"""

RAG_GENERATION_PROMPT = """Evaluate the proposed invention idea using the retrieved patent and research paper evidence provided below.

Proposed Idea:
{idea}

Retrieved Prior-Art Evidence:
{context}

Provide a structured, evidence-grounded report covering:
1. Extracted Technical Features
2. Relevant Prior Art Found (cite specific documents e.g. [PAT-001, Claim 1])
3. Overlapping vs Distinct Features
4. Preliminary Novelty Assessment (cautious research language; state that this is not a legal patentability opinion)
5. Potential Patent and Research Gaps based on retrieved evidence
6. Clear statement of any limitations and missing evidence
"""

GAP_DISCOVERY_PROMPT = """Identify potential patent and research gaps by contrasting the proposed idea against the analyzed patents and research papers.

User Idea Features:
{idea_features}

Analyzed Patents Summary:
{patent_summary}

Analyzed Research Papers Summary:
{paper_summary}

Analyze the intersection and white-spaces between patents and academic literature. Output a JSON object adhering to this schema:
{{
  "well_covered_areas": ["Area 1 thoroughly covered in literature"],
  "partially_covered_areas": ["Area 2 with partial coverage"],
  "underrepresented_features": ["Feature with little or no coverage in retrieved corpus"],
  "unexplored_combinations": ["Specific combinations of features that appear less represented"],
  "patent_vs_paper_differences": "Differences observed between what patents claim vs what papers study",
  "potential_research_directions": ["Recommended research opportunities"],
  "evidence_citations": ["Citations supporting this analysis"]
}}
"""

REPORT_SYNTHESIS_PROMPT = """Synthesize the final explainable report for the proposed idea.

Idea: {idea_title}
Novelty Level: {novelty_level} (Score: {novelty_score})
Key Prior Art Evidence: {citations}
Key Gaps Identified: {gaps}

Write a comprehensive executive summary (3-4 paragraphs) suitable for an academic PBL-3 project presentation.
Emphasize evidence grounding, cite specific sources, highlight why the preliminary novelty level was assigned, and clearly state that this is not a legal opinion.
"""
