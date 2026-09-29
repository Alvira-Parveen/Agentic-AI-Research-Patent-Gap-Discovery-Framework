import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.schemas.documents import DocumentMetadata, DocumentType
from src.ingestion.pdf_loader import PDFLoader
from src.utils.ids import generate_document_id
from src.utils.logging import logger


CURATED_PAPER_METADATA = {
    "1.pdf": {"title": "Patent intelligence in the age of AI: Unlocking strategic insights through granular classification", "citation": "Giuntelli et al., 2026", "authors": ["Giuntelli", "et al."], "year": "2026"},
    "2.pdf": {"title": "Agent Ideate: A Framework for Product Idea Generation from Patents Using Agentic AI", "citation": "Kanumolu et al., 2025", "authors": ["Kanumolu", "et al."], "year": "2025"},
    "3.pdf": {"title": "Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System", "citation": "Su et al., 2025", "authors": ["Su", "et al."], "year": "2025"},
    "4.pdf": {"title": "EVOPAT: A Multi-LLM-Based Patents Summarization and Analysis Framework", "citation": "Wang et al., 2024", "authors": ["Wang", "et al."], "year": "2024"},
    "5.pdf": {"title": "Can AI Examine Novelty of Patents?: Novelty Evaluation Based on Correspondence between Patent Claim and Prior Art", "citation": "Ikoma & Mitamura, 2025", "authors": ["Ikoma", "Mitamura"], "year": "2025"},
    "6.pdf": {"title": "In-depth Analysis of Graph-based RAG in a Unified Framework", "citation": "Zhou et al., 2025", "authors": ["Zhou", "et al."], "year": "2025"},
    "7.pdf": {"title": "Exploring Design of Multi-Agent LLM Dialogues for Research Ideation", "citation": "Ueda et al., 2025", "authors": ["Ueda", "et al."], "year": "2025"},
    "8.pdf": {"title": "An automatic patent literature retrieval system based on LLM-RAG", "citation": "Ding et al., 2025", "authors": ["Ding", "et al."], "year": "2025"},
    "9.pdf": {"title": "Research on Evaluation Methods for Patent Novelty Search Systems and Empirical Analysis", "citation": "Zhang et al., 2025", "authors": ["Zhang", "et al."], "year": "2025"},
    "10.pdf": {"title": "ToC: Tree-of-Claims Search with Multi-Agent Language Models", "citation": "Yu et al., 2026", "authors": ["Yu", "et al."], "year": "2026"},
    "11.pdf": {"title": "IdeaForge: A Knowledge Graph-Grounded Multi-Agent Framework for Cross-Methodology Innovation Analysis", "citation": "Bose, 2026", "authors": ["Bose"], "year": "2026"},
    "12.pdf": {"title": "AgentSwift: Efficient LLM Agent Design via Value-Guided Hierarchical Search", "citation": "Li et al., 2026", "authors": ["Li", "et al."], "year": "2026"},
    "13.pdf": {"title": "TCLMA: A Two-dimension Contrastive Learning based Multiagent Framework for Scientific Novelty Evaluation", "citation": "Zheng et al., 2025", "authors": ["Zheng", "et al."], "year": "2025"},
    "14.pdf": {"title": "Integrated Patent Prior Art Search with Claim-Aware Retrieval and Novelty Assessment", "citation": "Han & Qu, 2026", "authors": ["Han", "Qu"], "year": "2026"},
    "15.pdf": {"title": "A novel patentability detection model based on Siamese network", "citation": "Kayakökü & Tüfekci, 2025", "authors": ["Kayakökü", "Tüfekci"], "year": "2025"},
    "16.pdf": {"title": "Towards Automated Patent Workflows: Multi-agent optimization patterns", "citation": "Li et al., 2026", "authors": ["Li", "et al."], "year": "2026"},
    "17.pdf": {"title": "Enhancing the Patent Matching Capability of Large Language Models via Memory Graph", "citation": "Xiong et al., 2025", "authors": ["Xiong", "et al."], "year": "2025"},
    "18.pdf": {"title": "Innovation Organization & Management: AI-based novelty detection in crowdsourced idea spaces", "citation": "Just et al., 2024", "authors": ["Just", "et al."], "year": "2024"},
    "19.pdf": {"title": "An extraction and novelty evaluation framework for technology knowledge elements of patents", "citation": "Wei et al., 2024", "authors": ["Wei", "et al."], "year": "2024"},
    "20.pdf": {"title": "Research Paper Retrieval-Augmented Generation Systems for Intellectual Property", "citation": "Ren et al., 2025", "authors": ["Ren", "et al."], "year": "2025"}
}


class PaperLoader:
    def __init__(self, loader: Optional[PDFLoader] = None):
        self.loader = loader or PDFLoader()

    def parse_paper(self, file_path: str, index: int) -> Tuple[DocumentMetadata, Dict[str, str]]:
        path = Path(file_path)
        pages, is_scanned = self.loader.load_pdf(str(path))
        full_text = "\n\n".join(pages)
        front_page = pages[0] if pages else ""

        doc_id = generate_document_id("paper", index)
        curated = CURATED_PAPER_METADATA.get(path.name, {})

        title = curated.get("title") or self._extract_title(front_page, path.stem)
        authors = curated.get("authors") or self._extract_authors(front_page)
        date = curated.get("year") or self._extract_date(full_text)
        pub_citation = curated.get("citation") or f"arXiv-{path.stem}"
        sections = self._extract_sections(pages, full_text)

        metadata = DocumentMetadata(
            document_id=doc_id,
            document_type=DocumentType.PAPER,
            title=title,
            publication_number=pub_citation,
            authors=authors,
            assignee=authors[0] if authors else "Academic Institution",
            publication_date=date,
            source=path.name,
            filename=path.name,
            abstract=sections.get("abstract", "")[:1200],
            num_pages=len(pages),
            raw_char_count=len(full_text),
            extra={"is_scanned": is_scanned}
        )

        return metadata, sections

    def _extract_title(self, text: str, fallback: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        candidate = []
        for line in lines[:10]:
            # Skip journal headers or vol lines
            if re.match(r'^(Vol\.|arXiv|ISSN|Article|http|www\.)', line, re.IGNORECASE):
                continue
            if line.lower().startswith("abstract"):
                break
            candidate.append(line)
            if len(" ".join(candidate)) > 30 and (len(line) < 40 or line.endswith(":")):
                break
            if len(candidate) >= 3:
                break
        title = " ".join(candidate).strip()
        return title[:180] if len(title) > 5 else f"Research Paper {fallback}"

    def _extract_authors(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        authors = []
        for line in lines[1:8]:
            if "abstract" in line.lower() or "introduction" in line.lower():
                break
            if any(char in line for char in ["@", "university", "institute", "department", "school", "laboratory"]):
                continue
            # Check if line looks like author names
            if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', line):
                parts = re.split(r'[,;*†1234567890]+', line)
                for p in parts:
                    p_clean = p.strip()
                    if 2 < len(p_clean) < 35 and " " in p_clean:
                        authors.append(p_clean)
        return authors[:6]

    def _extract_date(self, text: str) -> Optional[str]:
        match = re.search(r'(?:20[12][0-9]|199[0-9])', text[:1500])
        return match.group(0) if match else "2025"

    def _extract_sections(self, pages: List[str], full_text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "conclusion": ""
        }

        # Abstract
        abs_match = re.search(r'(?:Abstract|ABSTRACT)[:\s—\-\.](.*?)(?=\n\s*(?:1\.?\s+Introduction|Key\s?words|Categories|Index Terms|I\.\s+INTRODUCTION|\n\n\n))', full_text, re.DOTALL | re.IGNORECASE)
        if abs_match:
            sections["abstract"] = abs_match.group(1).strip()
        else:
            sections["abstract"] = full_text[:600].strip()

        # Introduction
        intro_match = re.search(r'(?:1\.?\s+Introduction|I\.\s+INTRODUCTION)(.*?)(?=(?:2\.?\s+|II\.\s+|Related Work|Methodology|Proposed Method))', full_text, re.DOTALL | re.IGNORECASE)
        if intro_match:
            sections["introduction"] = intro_match.group(1)[:5000].strip()

        # Methodology / Proposed Method
        method_match = re.search(r'(?:Methodology|Proposed (?:Method|Framework|Approach)|System Architecture|Technical Approach)(.*?)(?=(?:Experiments|Results|Evaluation|Discussion|Conclusion))', full_text, re.DOTALL | re.IGNORECASE)
        if method_match:
            sections["methodology"] = method_match.group(1)[:8000].strip()
        else:
            # Fallback to middle chunk of document
            mid = len(full_text) // 2
            sections["methodology"] = full_text[mid:mid+5000].strip()

        # Results / Evaluation
        res_match = re.search(r'(?:Experiments|Results|Evaluation|Empirical Analysis)(.*?)(?=(?:Conclusion|Discussion|Related Work|\Z))', full_text, re.DOTALL | re.IGNORECASE)
        if res_match:
            sections["results"] = res_match.group(1)[:5000].strip()

        # Conclusion
        conc_match = re.search(r'(?:Conclusion|Concluding Remarks|Summary and Outlook)(.*?)(?=(?:References|Acknowledgments|\Z))', full_text, re.DOTALL | re.IGNORECASE)
        if conc_match:
            sections["conclusion"] = conc_match.group(1)[:3000].strip()

        return sections
