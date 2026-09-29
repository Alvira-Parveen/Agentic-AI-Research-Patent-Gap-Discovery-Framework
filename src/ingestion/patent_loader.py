import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.schemas.documents import DocumentMetadata, DocumentType
from src.ingestion.pdf_loader import PDFLoader
from src.utils.ids import generate_document_id
from src.utils.logging import logger


class PatentLoader:
    def __init__(self, loader: Optional[PDFLoader] = None):
        self.loader = loader or PDFLoader()

    def parse_patent(self, file_path: str, index: int) -> Tuple[DocumentMetadata, Dict[str, str]]:
        path = Path(file_path)
        pages, is_scanned = self.loader.load_pdf(str(path))
        full_text = "\n\n".join(pages)

        doc_id = generate_document_id("patent", index)
        front_page = pages[0] if pages else ""

        # Extract Publication Number
        pub_num = self._extract_patent_number(front_page, path.stem)

        # Extract Title
        title = self._extract_title(front_page, path.stem)

        # Extract Assignee / Applicant
        assignee = self._extract_assignee(front_page)

        # Extract Inventors
        inventors = self._extract_inventors(front_page)

        # Extract Date
        pub_date = self._extract_date(front_page)

        # Section Extraction: Abstract, Claims, Description
        sections = self._extract_sections(pages, full_text)

        metadata = DocumentMetadata(
            document_id=doc_id,
            document_type=DocumentType.PATENT,
            title=title,
            publication_number=pub_num,
            authors=inventors,
            assignee=assignee,
            publication_date=pub_date,
            source=path.name,
            filename=path.name,
            abstract=sections.get("abstract", "")[:1200],
            num_pages=len(pages),
            raw_char_count=len(full_text),
            extra={"is_scanned": is_scanned}
        )

        return metadata, sections

    def _extract_patent_number(self, text: str, fallback: str) -> str:
        # Regex for US patent numbers e.g. US 10,902,042 B2 or US 2024/0281487 A1
        match = re.search(r'(?:US|US0)?\s*([0-9]{1,2},?[0-9]{3},?[0-9]{3}\s*[A-Z][0-9]?|[0-9]{4}/?[0-9]{7}\s*[A-Z][0-9]?)', text)
        if match:
            clean_num = match.group(0).strip()
            if not clean_num.startswith("US"):
                clean_num = f"US {clean_num}"
            return clean_num
        return f"US-PAT-{fallback}"

    def _extract_title(self, text: str, fallback: str) -> str:
        # Title usually under (54)
        match = re.search(r'\((?:54)\)\s*(.*?)(?=\((?:57|71|72|73|74|65|21|22|51|52|58)\)|ABSTRACT|Applicant|Inventor)', text, re.DOTALL | re.IGNORECASE)
        if match:
            clean = " ".join(match.group(1).split()).strip()
            if len(clean) > 5:
                return clean[:200]
        # Fallback: scan lines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for idx, l in enumerate(lines[:30]):
            if "(54)" in l:
                return " ".join(lines[idx:min(idx+3, len(lines))]).replace("(54)", "").strip()[:200]
        return f"Patent Document {fallback}"

    def _extract_assignee(self, text: str) -> Optional[str]:
        match = re.search(r'(?:Applicant|Assignee):\s*([^\n\(\)]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_inventors(self, text: str) -> List[str]:
        match = re.search(r'Inventor(?:s)?:\s*([^\n\(\)]+)', text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            return [inv.strip() for inv in re.split(r'[;,]', raw) if inv.strip()][:5]
        return []

    def _extract_date(self, text: str) -> Optional[str]:
        match = re.search(r'Date of Patent:\s*([A-Za-z]+\.?\s*\d{1,2},\s*\d{4})', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match_pub = re.search(r'Pub\. Date:\s*([A-Za-z]+\.?\s*\d{1,2},\s*\d{4})', text, re.IGNORECASE)
        if match_pub:
            return match_pub.group(1).strip()
        return None

    def _extract_sections(self, pages: List[str], full_text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {"abstract": "", "claims": "", "description": ""}

        # Abstract is usually on page 0 under (57) or ABSTRACT
        front = pages[0] if pages else ""
        abs_match = re.search(r'(?:\(57\)\s*ABSTRACT|ABSTRACT)\s*(.*?)(?=\n\s*(?:\([0-9]+\)|US\s*\d|Claims|What is claimed|\Z))', front, re.DOTALL | re.IGNORECASE)
        if abs_match:
            sections["abstract"] = abs_match.group(1).strip()
        elif len(front) > 200:
            # First 500 chars as fallback abstract
            sections["abstract"] = front[:600].strip()

        # Claims extraction: "What is claimed is:" or "Claims"
        claims_pattern = re.compile(r'(?:What\s+(?:is|are)\s+claimed\s+(?:is|are)|We\s+claim|Claims:?)\s*(.*?)(?=\Z)', re.DOTALL | re.IGNORECASE)
        claims_match = claims_pattern.search(full_text)
        if claims_match:
            sections["claims"] = claims_match.group(1).strip()
        else:
            # Search last 3 pages for claims
            tail_text = "\n\n".join(pages[-3:]) if len(pages) >= 3 else full_text
            tail_claims = re.findall(r'(\d+\.\s+[A-Z][^\n]+(?:\n[^\n]+){1,10})', tail_text)
            if tail_claims:
                sections["claims"] = "\n\n".join(tail_claims)
            else:
                sections["claims"] = tail_text[-2500:].strip()

        # Description is everything else
        desc_start = re.search(r'(?:FIELD OF THE INVENTION|BACKGROUND|DETAILED DESCRIPTION)', full_text, re.IGNORECASE)
        if desc_start:
            desc_text = full_text[desc_start.start():]
            # Cut off at claims if present
            if claims_match:
                desc_text = full_text[desc_start.start():claims_match.start()]
            sections["description"] = desc_text[:15000].strip()
        else:
            sections["description"] = full_text[:10000].strip()

        return sections
