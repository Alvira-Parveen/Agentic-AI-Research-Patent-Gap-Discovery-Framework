# Explainable Patent and Research Gap Discovery Using Retrieval-Augmented Multi-Agent AI

**Alvira Parveen, Rohit**  
*Department of Computer Science & Engineering, Sharda University, Greater Noida, India*  
*Mentor: Dr. Sandeep Gupta*

---

## Abstract
Identifying relevant prior art and potential research gaps for emerging inventions requires substantial manual effort because technical disclosures are distributed across patent specifications and scientific literature. This paper presents an explainable, retrieval-augmented multi-agent framework for preliminary invention screening and exploratory research-gap discovery across both corpora. Given a natural-language invention concept, the system retrieves relevant patent and research-paper passages using Sentence-Transformer embeddings, FAISS-based dense retrieval, and BM25-based hybrid ranking. A modular LangGraph workflow coordinates specialized agents for patent analysis, scientific literature analysis, technical feature comparison, prior-art overlap assessment, and candidate research-gap identification. To improve evidence traceability, the report-generation stage receives source chunk identifiers, document metadata, and verbatim excerpts, enabling generated claims to be inspected against their cited passages. The framework was evaluated using a curated corpus of 40 documents comprising 20 patents and 20 research papers, with a benchmark of 20 technical queries. Hybrid retrieval achieved Precision@3 of 46.67%, compared with 36.67% for dense-only retrieval, although the observed improvements were not statistically significant after correction for multiple comparisons. An interactive Streamlit prototype demonstrates the end-to-end workflow and source-passage inspection. The framework supports transparent, corpus-bounded exploration of prior art and potential research gaps; its overlap scores are engineering heuristics rather than legal patentability assessments, and further independent evaluation is required to establish semantic citation accuracy and generalizability.

**Keywords**—Patent analysis, research-gap discovery, retrieval-augmented generation, multi-agent systems, hybrid information retrieval, evidence grounding, explainable artificial intelligence.

---

## I. INTRODUCTION

The accelerating pace of technological innovation has generated an overwhelming volume of technical disclosures across proprietary patent registries and academic archives. In modern industrial R&D and scientific research, conducting effective preliminary invention screening is a critical prerequisite to avoid duplicative effort, evaluate technical viability, and identify unresolved challenges. However, early-stage invention assessment remains an exhausting, manual, and error-prone process. Researchers must formulate complex search queries across fragmented databases, navigate diverse classification taxonomies, and manually cross-reference dozens of dense, highly specialized documents [1]–[4].

A fundamental obstacle in automated prior-art analysis is *dual-corpus fragmentation*. Technical knowledge is bifurcated across two distinct repositories with divergent linguistic styles, structural conventions, and communicative goals: patent specifications, which employ legally guarded terminology focused on statutory claim scope [5], [14], [21], [34], and scientific literature, which emphasizes experimental hypotheses, empirical benchmarks, and methodological limitations [3], [7]. Existing discovery tools generally operate in silos, focusing either on patent search or scholarly literature retrieval. Consequently, valuable cross-domain white spaces—such as patented architectures with unaddressed empirical limitations, or published academic methodologies that lack corresponding patent disclosures—remain difficult to identify systematically [4], [11].

While Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) offer promising capabilities for automated technical synthesis [6], [8], [28], [32], conventional RAG pipelines exhibit notable limitations when applied to multi-corpus technical discovery. Conventional RAG frameworks typically concatenate heterogeneous text chunks into a single monolithic generation prompt without domain-specific decomposition. This unconstrained synthesis risks generating ungrounded assertions, conflating patent claims with academic hypotheses, and obscuring source attribution, undermining the reliability required for engineering and intellectual property assessments [1], [5].

To address these challenges, we present an explainable, retrieval-augmented multi-agent framework for preliminary invention screening and exploratory research-gap discovery across both patent and academic corpora. Given a natural-language invention concept, the system extracts structured technical features, retrieves candidate passages through hybrid vector search, and coordinates an ensemble of specialized autonomous agents via LangGraph to produce an evidence-grounded analysis report.

The primary contributions of this paper are fourfold:
1. **Dual-Corpus Hybrid Retrieval**: An information retrieval pipeline combining Sentence-Transformer embeddings (`all-MiniLM-L6-v2`), FAISS dense vector search, and BM25 lexical ranking to concurrently retrieve and rank candidate evidence from partitioned patent and academic literature corpora.
2. **Modular Multi-Agent Architecture**: A graph-based workflow implemented in LangGraph that decouples prior-art analysis across specialized agents: a `PatentAnalysisAgent` to analyze claim structures, a `ResearchAnalysisAgent` to extract scientific methodologies, a `FeatureComparator` to construct alignment matrices, and a `GapDiscoveryAgent` to identify potential cross-literature gaps.
3. **Passage-Anchored Synthesis and Traceability**: A report-generation mechanism that receives explicit source chunk identifiers, document metadata, and verbatim excerpts, reducing unsupported cross-domain claims and enabling users to inspect generated statements directly against cited passages.
4. **Empirical Diagnostic Evaluation & Prototype Deployment**: A quantitative diagnostic benchmark over a curated 40-document corpus (20 patents and 20 research papers) across 20 technical queries, evaluating dense and hybrid retrieval baselines, accompanied by an open-source, interactive Streamlit demonstration prototype.

The remainder of this paper is organized as follows: Section II reviews related work. Section III presents the system architecture and multi-agent methodology. Section IV outlines the experimental setup and evaluation protocol. Section V discusses empirical results and diagnostic findings. Section VI presents system limitations and boundary disclosures. Section VII concludes the paper and outlines future directions.

---

## II. RELATED WORK

Recent research papers addressing patent analysis and AI ideation are summarised in Table I [1]–[20], while relevant patents covering retrieval and claim mapping are analyzed in Table II [21]–[40].

### Table I: Comparative Analysis of 20 Selected Research Papers
| No. | Title & Year | Authors | Method / Key Finding / Limitation |
|---|---|---|---|
| [1] | Patent Intelligence in the Age of AI: Unlocking Strategic Insights through Granular Classification [2026] | Giuntelli et al. | LLM + RAG; patent intelligence; hallucination/cost |
| [2] | Agent Ideate: A Framework for Product Idea Generation from Patents Using Agentic AI [2025] | Kanumolu et al. | Multi-agent AI; patent ideation; domain dependent |
| [3] | Many Heads Are Better Than One: Improved Scientific Idea Generation by a LLM Based Multi Agent System [2026] | Dong et al. | Multi-agent LLM; scientific ideation; no patent analysis |
| [4] | EVOPAT: A MULTI-LLM-BASED PATENTS SUMMARIZATION AND ANALYSIS AGENT [2024] | Wang et al. | Multi-LLM + RAG; patent analysis; patent focused |
| [5] | Can AI Examine Novelty of Patents?: Novelty Evaluation Based on the Correspondence between Patent Claim and Prior Art [2025] | Ikoma et al. | LLM + claim matching; novelty evaluation; domain/year dependent |
| [6] | In-depth Analysis of Graph-based RAG in a Unified Framework [2026] | Zhou et al. | GraphRAG; improved retrieval; model/context limitations |
| [7] | Exploring Design of Multi-Agent LLM Dialogues for Research Ideation [2025] | Ueda et al. | Multi-agent dialogue; idea refinement; no novelty verification |
| [8] | An automatic patent literature retrieval system based on LLM-RAG [2025] | Ding et al. | FAISS + RAG; patent retrieval; no gap analysis |
| [9] | Research on Evaluation Methods for Patent Novelty Search Systems and Empirical Analysis [2025] | Zhang et al. | Semantic search; novelty-search evaluation; incomplete ground truth |
| [10] | ToC: Tree-of-Claims Search with Multi-Agent Language Models [2026] | Yu et al. | Multi-agent + MCTS; claim search; domain specific |
| [11] | IdeaForge: A Knowledge Graph-Grounded Multi-Agent Framework for Cross-Methodology Innovation Analysis and Patent Claim Generation [2026] | Bose | KG + multi-agent; innovation analysis; small use-case set |
| [12] | AgentSwift: Efficient LLM Agent Design via Value-Guided Hierarchical Search [2026] | Li et al. | Hierarchical agent search; efficient agents; not patent-gap focused |
| [13] | TCLMA: A Two-dimension Contrastive Learning based Multiagent Framework for Scientific Novelty Evaluation [2025] | Zheng et al. | Contrastive learning + agents; novelty evaluation; no white-space analysis |
| [14] | Integrated Patent Prior Art Search with Claim Aware Retrieval and Novelty Assessment [2026] | Han et al. | Claim-aware retrieval; prior-art + novelty; no research-gap analysis |
| [15] | A novel patentability detection model based on Siamese network [2025] | Kayaköku et al. | Siamese LSTM; patentability detection; EV domain only |
| [16] | Towards Automated Patent Workflows: AI-Orchestrated Multi-Agent Framework for Intellectual Property Management and Analysis [2024] | Srinivas et al. | Agents + RAG + KG; automated IP workflow; USPTO focused |
| [17] | Enhancing the Patent Matching Capability of Large Language Models via the Memory Graph [2025] | Xiong et al. | Memory Graph + RAG; patent matching; no gap discovery |
| [18] | AI-based novelty detection in crowdsourced idea spaces [2024] | Just et al. | SBERT + similarity; idea novelty detection; moderate human correlation |
| [19] | An extraction and novelty evaluation framework for technology knowledge elements of patents [2024] | Wei et al. | BERT + UIE + KG; technology-element novelty; domain specific |
| [20] | Retrieval-Augmented Generation Systems for Intellectual Property via Synthetic Multi-Angle Fine-tuning [2025] | Ren et al. | RAG + retriever tuning; IP retrieval; Taiwan focused |

### Table II: Comparative Analysis of 20 Selected Patents
| No. | Patent & Year | Technology / Method | Key Capability / Limitation |
|---|---|---|---|
| [21] | Patent Claim Reference Generation [2020] | Claim-limitation extraction + evidence retrieval | Claim evidence generation; claim-focused |
| [22] | Vector-Based Contextual Text Searching [2022] | Vector representations + similarity ranking | Semantic text retrieval; general-purpose |
| [23] | Novelty Detection Using Deep Learning Network [2022] | Autoencoder + representation comparison | Novelty scoring; not patent-specific |
| [24] | Automated System and Methodology for Feature Extraction [2022] | ML-based feature extraction | Automated feature extraction; domain-specific |
| [25] | Construction Method, Device, Computing Device, and Storage Medium for Constructing Patent Knowledge Database [2023] | BERT/NER + knowledge graph | Patent knowledge extraction; KG construction |
| [26] | Machine Learning Architecture for Contextual Data Retrieval [2023] | Query generation + contextual retrieval | Retrieval and confidence checking; general-purpose |
| [27] | Automated Patent Claim Scope Concept Mapping [2023] | Semantic claim analysis + concept mapping | Claim scope mapping; patent-focused |
| [28] | Personalized Retrieval-Augmented Generation System [2023] | Embeddings + vector DB + LLM | RAG-based retrieval; personalized focus |
| [29] | Platform for Semantic Search and Dynamic Reclassification [2024] | Semantic search + graph analysis | Patent/scientific document search; broad platform |
| [30] | Digital Information Infrastructure and Method for Security Designated Data and with Granular Data Stores [2017] | Distributed storage + granular extraction | Structured data management; not patent-gap focused |
| [31] | Retrieval-Augmented Generation (RAG) System Optimization [2025] | Chunking + vector retrieval + ranking | RAG optimization; retrieval-focused |
| [32] | Method and System for Optimizing Use of Retrieval-Augmented Generation Pipelines in Generative AI Applications [2025] | Multi-LLM + RAG orchestration | Specialized RAG pipelines; architectural reference |
| [33] | Methods and Systems of Handling Patent Claims [2017] | Claim parsing + semantic processing | Patent claim handling; claim-focused |
| [34] | Systems and Methods for Analyzing Validity or Infringement of Patent Claims [2020] | NLP + information retrieval | Claim evidence matching; legal analysis |
| [35] | Artificial Intelligence, Machine Learning, and Predictive Analytics for Patent and Non-Patent Documents [2022] | ML + predictive analytics | Patent/non-patent document analysis; legal/predictive focus |
| [36] | Enhanced Search Result Generation Using Multi-Document Summarization [2024] | Multi-document RAG + summarization | Cross-document synthesis; no research-gap analysis |
| [37] | Multi-Segment Text Search Using Machine Learning Model for Text Similarity [2025] | ML-based segmented-text similarity | Patent reference retrieval; similarity-focused |
| [38] | Patent Mapping [2025] | Claims + concepts + ontologies | White-space and technology mapping; patent-centric |
| [39] | Computer Implemented Methods for Automated Analysis/Use of Data, Including LLM [2026] | LLM + structured semantic reasoning | Search augmentation and reasoning; general-purpose |
| [40] | Augmented Question and Answer with Large Language Models [2026] | Embeddings + nearest-segment retrieval + LLM | Contextual document QA; no gap discovery |

---

## III. PROPOSED METHODOLOGY

This section details the architecture, retrieval algorithms, multi-agent coordination protocol, and evidence grounding mechanism of the proposed framework across four core subsections.

---

### 3.1. Overall System Architecture

#### 1. Purpose
Automated discovery across patents and scientific literature requires coordinating diverse screening tasks without losing domain context or generating ungrounded assertions. Monolithic language models conflate legal claims with empirical findings and lack intermediate verification checkpoints. A modular architecture provides explicit boundaries, intermediate audit points, and reproducible execution across both corpora.

#### 2. Method
The framework models discovery as a state-directed acyclic graph (DAG) coordinated via LangGraph. Execution maintains an explicit, typed global execution state $\mathcal{S}$:

$$\mathcal{S} = \langle \mathcal{I}, \mathcal{F}_{\mathrm{idea}}, \mathcal{R}_{\mathrm{pat}}, \mathcal{R}_{\mathrm{pap}}, \mathcal{A}_{\mathrm{pat}}, \mathcal{A}_{\mathrm{pap}}, \mathcal{M}_{\mathrm{comp}}, \mathcal{N}, \mathcal{G}, \mathcal{T}_{\mathrm{rep}} \rangle$$

Execution flows through four sequential phases:
- *Phase 1 (Input & Feature Ingestion)*: The input concept $\mathcal{I}$ is parsed into a structured feature schema $\mathcal{F}_{\mathrm{idea}} = \langle \mathrm{Tech}, \mathrm{Feat}, \mathrm{Method}, \mathrm{In}, \mathrm{Out}, \mathrm{Comp} \rangle$.
- *Phase 2 (Partitioned Dual Retrieval)*: Top candidate passages are fetched independently from the patent corpus ($\mathcal{R}_{\mathrm{pat}}$) and the research literature corpus ($\mathcal{R}_{\mathrm{pap}}$).
- *Phase 3 (Multi-Agent Analytical Decomposition)*: Domain agents evaluate candidates to produce patent claim analyses $\mathcal{A}_{\mathrm{pat}}$, literature analyses $\mathcal{A}_{\mathrm{pap}}$, feature alignment matrix $\mathcal{M}_{\mathrm{comp}}$, prior-art overlap metrics $\mathcal{N}$, and candidate gaps $\mathcal{G}$.
- *Phase 4 (Passage-Anchored Report Synthesis)*: Upstream analytical outputs and verbatim excerpts are composed into the final auditable discovery report $\mathcal{T}_{\mathrm{rep}}$.

#### 3. Implementation
The state machine is built in Python 3.11 using LangGraph (`StateGraph`), Pydantic v2 schemas for deterministic state transitions, and a dual-engine LLM backend (Google Gemini 2.5 Flash with an automated deterministic fallback). The presentation layer is implemented in Streamlit.

---

### 3.2. Dual-Corpus Hybrid Retrieval

#### 1. Purpose
Patent specifications rely on stylized legal phraseology, whereas academic articles use domain-specific empirical vocabulary. Dense semantic vectors excel at capturing conceptual meaning but often miss verbatim keywords, alphanumeric classification codes, and claim numbers. Lexical retrieval captures exact technical nomenclature but fails on synonymous phrasing, necessitating a hybrid retrieval strategy.

#### 2. Method
A partitioned hybrid retrieval strategy runs parallel dense and lexical pipelines across separate patent and literature indices. 

For dense retrieval, each document chunk $c$ and query $q$ is mapped to a 384-dimensional dense vector $\mathbf{e} \in \mathbb{R}^{384}$. Semantic relevance is computed using cosine similarity:

$$S_{\mathrm{dense}}(q, c) = \cos(\mathbf{e}_q, \mathbf{e}_c) = \frac{\mathbf{e}_q \cdot \mathbf{e}_c}{\|\mathbf{e}_q\| \|\mathbf{e}_c\|}$$

In parallel, BM25 calculates keyword relevance:

$$S_{\mathrm{lex}}(q, c) = \mathrm{BM25}(q, c)$$

Candidate lists are fused using Reciprocal Rank Fusion (RRF), eliminating the need for arbitrary score normalization:

$$\mathrm{RRF}(c) = \frac{1}{k + r_{\mathrm{dense}}(c)} + \frac{1}{k + r_{\mathrm{lex}}(c)}$$

where $r(c)$ is the ordinal rank in each result set and $k = 60$ is the smoothing constant. The top $K=5$ chunks per corpus populate $\mathcal{R}_{\mathrm{pat}}$ and $\mathcal{R}_{\mathrm{pap}}$.

#### 3. Implementation
Text extraction uses PyMuPDF for digital PDFs and Tesseract OCR for scans. Chunks are sized at $L=600$ tokens with $\Delta=80$ overlap. Dense embeddings are generated with Sentence-Transformers (`all-MiniLM-L6-v2`) and stored in L2-normalized FAISS indices. Lexical retrieval is executed via `rank_bm25`.

---

### 3.3. Multi-Agent Orchestration

#### 1. Purpose
Invention assessment requires multi-perspective technical analysis that cannot be effectively accomplished within a single generative prompt. Evaluating patent novelty demands strict boundary deconstruction of claims, while academic analysis focuses on empirical performance, benchmarks, and limitations. Decoupled agents ensure each domain is analyzed under its native analytical standards before synthesizing cross-domain overlap.

#### 2. Method
The framework deploys specialized LangGraph nodes communicating through typed state transformations:
- **`PatentAnalysisAgent`**: Parses candidate patent chunks $\mathcal{R}_{\mathrm{pat}}$ to extract independent claim scopes, technical problems, and disclosed implementations ($\mathcal{A}_{\mathrm{pat}}$).
- **`ResearchAnalysisAgent`**: Analyzes candidate literature chunks $\mathcal{R}_{\mathrm{pap}}$ to isolate experimental methodologies, empirical conclusions, and documented edge cases ($\mathcal{A}_{\mathrm{pap}}$).
- **`FeatureComparator`**: Performs Cartesian cross-matching between $\mathcal{F}_{\mathrm{idea}}$ and retrieved evidence, producing alignment matrix $\mathcal{M}_{\mathrm{comp}}$ with states $\{\mathrm{Match}, \mathrm{PartialMatch}, \mathrm{Uncovered}\}$.
- **`NoveltyAgent`**: Calculates the multi-factor Prior-Art Overlap Index ($POI$):
  $$POI = 0.40 \cdot S_{\mathrm{sem}} + 0.35 \cdot O_{\mathrm{tech}} + 0.25 \cdot C_{\mathrm{claim}}$$
  where $S_{\mathrm{sem}}$ is peak cosine similarity, $O_{\mathrm{tech}}$ is technical feature overlap, and $C_{\mathrm{claim}}$ is patent claim coverage. Modulated by evidence strength $E_{\mathrm{str}}$, the engineering heuristic score is:
  $$\mathrm{OverlapScore} = 1.0 - \left(POI \times (0.60 + 0.40 \cdot E_{\mathrm{str}})\right)$$
- **`GapDiscoveryAgent`**: Identifies unaddressed features and unexplored cross-domain combinations ($f_{\mathrm{pat}} \wedge f_{\mathrm{paper}}$ absent in the seed corpus).

#### 3. Implementation
Nodes are implemented as pure Python functions receiving `AgentState` and returning state deltas. Prompts enforce structured JSON outputs conforming to Pydantic schemas, guaranteeing type safety and deterministic graph traversal.

---

### 3.4. Passage-Anchored Evidence Grounding

#### 1. Purpose
Generative models frequently suffer from hallucinations, fabricated citations, and conflated technical claims when summarizing complex technical literature. Reliable preliminary screening requires verifiable evidence provenance where every assertion can be audited against original texts. Passage-level anchoring guarantees transparency and enables rapid human verification.

#### 2. Method
Upstream agents pass an enriched evidence payload to the synthesis node. Rather than receiving arbitrary document titles, the synthesis node receives:
- Canonical document identifiers (`PAT-xxx`, `PAP-xxx`).
- Discrete chunk identifiers (e.g., `PAT-004-C138`).
- Verbatim source excerpts ($200\text{--}800$ characters).
- Sectional metadata (Claims, Detailed Description, Experimental Evaluation).

The synthesis prompt enforces that any statement of prior art must cite the explicit chunk ID and restrict technical descriptions to the mechanisms present in the excerpt.

#### 3. Implementation
The final report binds cited tags directly to an interactive Citation & Source Passage Inspector in Streamlit. Users can expand any cited reference to view its parent metadata, exact similarity score, and verbatim text, allowing immediate human auditing.
