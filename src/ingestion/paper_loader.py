import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.schemas.documents import DocumentMetadata, DocumentType
from src.ingestion.pdf_loader import PDFLoader
from src.utils.ids import generate_document_id
from src.utils.logging import logger


class PaperLoader:
    def __init__(self, loader: Optional[PDFLoader] = None):
        self.loader = loader or PDFLoader()

    def parse_paper(self, file_path: str, index: int) -> Tuple[DocumentMetadata, Dict[str, str]]:
        path = Path(file_path)
        pages, is_scanned = self.loader.load_pdf(str(path))
        full_text = "\n\n".join(pages)
        front_page = pages[0] if pages else ""

        doc_id = generate_document_id("paper", index)

        title = self._extract_title(front_page, path.stem)
        authors = self._extract_authors(front_page)
        date = self._extract_date(full_text)
        sections = self._extract_sections(pages, full_text)

        metadata = DocumentMetadata(
            document_id=doc_id,
            document_type=DocumentType.PAPER,
            title=title,
            publication_number=f"arXiv-{path.stem}",
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
