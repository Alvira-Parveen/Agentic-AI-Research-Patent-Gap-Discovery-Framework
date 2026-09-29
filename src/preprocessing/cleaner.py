import re

def normalize_whitespace(text: str) -> str:
    """Normalize excess whitespace and line breaks while preserving paragraph boundaries."""
    if not text:
        return ""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize unicode quotes and dashes
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    # Normalize multiple newlines (max 2 consecutive)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def clean_patent_claim_text(claim_text: str) -> str:
    """Clean patent claim text without removing technical terminology or claim structure."""
    if not claim_text:
        return ""
    # Ensure line breaks between numbered claims
    cleaned = re.sub(r'(\d+\.\s*\([a-zA-Z0-9_-]+\))', r'\n\1', claim_text)
    cleaned = re.sub(r'(\n\s*\d+\.\s+)', r'\n\1', cleaned)
    return normalize_whitespace(cleaned)

def remove_boilerplate(text: str) -> str:
    """Remove common headers/footers like page numbers and download notices."""
    if not text:
        return ""
    lines = text.split('\n')
    filtered = []
    for line in lines:
        l_strip = line.strip()
        # Drop standalone page numbers or headers
        if re.match(r'^\d+$', l_strip):
            continue
        if re.match(r'^page \d+ of \d+$', l_strip, re.IGNORECASE):
            continue
        if "Downloaded from" in l_strip or "All rights reserved" in l_strip:
            continue
        filtered.append(line)
    return '\n'.join(filtered).strip()
