import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.schemas.documents import DocumentMetadata, DocumentType
from src.ingestion.pdf_loader import PDFLoader
from src.utils.ids import generate_document_id
from src.utils.logging import logger


CURATED_PATENT_METADATA = {
    "Patent 1.pdf": {"title": "Claim Reference Generation & Intrinsic Evidence Hyperlinking", "publication_number": "US 10,902,042 B2", "assignee": "Black Hills IP Holdings, LLC"},
    "Patent 2.pdf": {"title": "Vector-Based Contextual Text Searching", "publication_number": "US 11,321,312 B2", "assignee": "ALEX - Alternative Experts, LLC"},
    "Patent 3.pdf": {"title": "Novelty Detection Using Deep Learning Neural Network", "publication_number": "US 11,816,578 B2", "assignee": "MakinaRocks Co., Ltd."},
    "Patent 4.pdf": {"title": "Computer Implemented Methods for Interacting with Semantic / Question-Answering Systems", "publication_number": "US 11,989,507 B2", "assignee": "Unlikely Artificial Intelligence Limited"},
    "Patent 5.pdf": {"title": "Multi-Segment Text Search Using Machine Learning Model for Text Similarity", "publication_number": "US 12,230,049 B2", "assignee": "Cognition IP Technology Inc."},
    "Patent 6.pdf": {"title": "Machine Learning Architecture for Contextual Data Retrieval", "publication_number": "US 12,339,875 B1", "assignee": "AskTuring.AI Inc."},
    "Patent 7.pdf": {"title": "Automated Patent Claim Scope Concept Mapping", "publication_number": "US 12,339,880 B2", "assignee": "Black Hills IP Holdings, LLC"},
    "Patent 8.pdf": {"title": "Personalized Retrieval-Augmented Generation System", "publication_number": "US 12,373,506 B1", "assignee": "Dropbox, Inc."},
    "Patent 9.pdf": {"title": "Platform for Semantic Search and Dynamic Reclassification", "publication_number": "US 12,461,922 B1", "assignee": "Reveal Networks, Inc."},
    "Patent 10.pdf": {"title": "Patent Mapping & Automated Patent Claim Scope Concept Mapping", "publication_number": "US 12,505,111 B2", "assignee": "Black Hills IP Holdings, LLC"},
    "Patent 11.pdf": {"title": "Retrieval-Augmented Generation (RAG) System Optimization", "publication_number": "US 12,561,314 B2", "assignee": "Goldman Sachs & Co. LLC"},
    "Patent 12.pdf": {"title": "Method and System for Optimizing Use of Retrieval Augmented Generation Pipelines in Generative AI", "publication_number": "US 12,602,412 B2", "assignee": "Vijay Madisetti"},
    "Patent 13.pdf": {"title": "Methods and Systems of Handling Patent Claims", "publication_number": "US 2017/0075877 A1", "assignee": "Marie-Therese Lepeltier"},
    "Patent 14.pdf": {"title": "Systems and Methods for Analyzing the Validity or Infringement of Patent Claims", "publication_number": "US 2020/0050638 A1", "assignee": "Parker Douglas Hancock"},
    "Patent 15.pdf": {"title": "Artificial Intelligence, Machine Learning, and Predictive Analytics for Patent and Non-Patent Documents", "publication_number": "US 2022/0343444 A1", "assignee": "DataNovo, Inc."},
    "Patent 16.pdf": {"title": "Enhanced Search Result Generation Using Multi-Document Summarization", "publication_number": "US 2024/0281487 A1", "assignee": "Snowflake Inc."},
    "Patent 17.pdf": {"title": "Multi-Segment Text Search Using Machine Learning Model for Prior Art", "publication_number": "US 2025/0259470 A1", "assignee": "Cognition IP Technology Inc."},
    "Patent 18.pdf": {"title": "Patent Mapping & White Space Discovery", "publication_number": "US 2025/0335515 A1", "assignee": "Black Hills IP Holdings, LLC"},
    "Patent 19.pdf": {"title": "Computer Implemented Methods for the Automated Analysis or Use of Data, Including Use of Large Language Models", "publication_number": "US 2026/0080164 A1", "assignee": "Unlikely Artificial Intelligence Limited"},
    "Patent 20.pdf": {"title": "Augmented Question and Answer (Q&A) with Large Language Models", "publication_number": "US 2026/0105258 A1", "assignee": "Micro Focus LLC"}
}


class PatentLoader:
    def __init__(self, loader: Optional[PDFLoader] = None):
        self.loader = loader or PDFLoader()

    def parse_patent(self, file_path: str, index: int) -> Tuple[DocumentMetadata, Dict[str, str]]:
        path = Path(file_path)
        pages, is_scanned = self.loader.load_pdf(str(path))
        full_text = "\n\n".join(pages)

        doc_id = generate_document_id("patent", index)
        front_page = pages[0] if pages else ""

        curated = CURATED_PATENT_METADATA.get(path.name, {})

        # Extract Publication Number
        pub_num = curated.get("publication_number") or self._extract_patent_number(front_page, path.stem)

        # Extract Title
        title = curated.get("title") or self._extract_title(front_page, path.stem)

        # Extract Assignee / Applicant
        assignee = curated.get("assignee") or self._extract_assignee(front_page)

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
