import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pymupdf

from src.utils.logging import logger
from src.preprocessing.cleaner import normalize_whitespace, remove_boilerplate


class PDFLoader:
    def __init__(self, cache_dir: Optional[Path] = None, max_ocr_pages: int = 25):
        self.cache_dir = cache_dir or Path("data/processed/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_ocr_pages = max_ocr_pages

    def load_pdf(self, file_path: str) -> Tuple[List[str], bool]:
        """
        Loads PDF text page-by-page.
        Returns (list_of_page_texts, is_scanned_flag).
        Uses disk cache to avoid redundant OCR.
        """
        path = Path(file_path)
        cache_file = self.cache_dir / f"{path.stem}_text.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("pages", []), data.get("is_scanned", False)
            except Exception as e:
                logger.warning(f"Cache read error for {path.name}: {e}. Re-extracting.")

        doc = pymupdf.open(str(path))
        num_pages = len(doc)
        digital_pages = []

        for page in doc:
            p_text = page.get_text("text")
            digital_pages.append(normalize_whitespace(remove_boilerplate(p_text)))

        total_digital_chars = sum(len(p) for p in digital_pages)
        is_scanned = (total_digital_chars / max(1, num_pages)) < 60

        if not is_scanned:
            doc.close()
            self._save_cache(cache_file, digital_pages, False)
            return digital_pages, False

        logger.info(f"Document {path.name} is scanned image PDF. Running OCR via Tesseract...")
        ocr_pages = []
        # For long patents (e.g. 100 pages), prioritize page 0 (front matter/abstract)
        # and specification / claims pages (skip middle drawing-only sheets if many)
        pages_to_process = list(range(min(num_pages, self.max_ocr_pages)))
        # Also ensure last 5 pages are included for claims if num_pages > max_ocr_pages
        if num_pages > self.max_ocr_pages:
            trailing = list(range(max(0, num_pages - 6), num_pages))
            pages_to_process = sorted(list(set(pages_to_process + trailing)))

        for p_idx in pages_to_process:
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=150)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_name = tmp.name
            try:
                pix.save(tmp_name)
                res = subprocess.run(
                    ["tesseract", tmp_name, "stdout", "-l", "eng"],
                    capture_output=True,
                    text=True
                )
                text = res.stdout if res.returncode == 0 else ""
                clean_t = normalize_whitespace(remove_boilerplate(text))
                if len(clean_t) > 30:
                    ocr_pages.append(clean_t)
            except Exception as e:
                logger.error(f"OCR error on {path.name} page {p_idx}: {e}")
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)

        doc.close()
        final_pages = ocr_pages if ocr_pages else digital_pages
        self._save_cache(cache_file, final_pages, True)
        return final_pages, True

    def _save_cache(self, cache_path: Path, pages: List[str], is_scanned: bool):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"pages": pages, "is_scanned": is_scanned}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write cache for {cache_path}: {e}")
