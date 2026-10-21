import math
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any
from src.schemas.documents import DocumentChunk


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and filter stop words for technical retrieval."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class BM25OkapiIndex:
    """
    Self-contained pure-Python BM25Okapi implementation for lexical search
    over patent claims, abstracts, descriptions, and research papers.
    """
    def __init__(self, chunks: List[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        
        self.doc_lens: List[int] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.nd: Dict[str, int] = {}
        
        # Build frequency index
        for c in chunks:
            text = f"{c.document_title or ''} {c.section or ''} {c.text}"
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for word in freq:
                self.nd[word] = self.nd.get(word, 0) + 1

        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)

        # Precompute Robertson-Sparck Jones IDF
        self.idf: Dict[str, float] = {}
        for word, freq in self.nd.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[DocumentChunk, float]]:
        """Scores all chunks against query and returns top_k sorted (chunk, score) tuples."""
        if not self.chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = [0.0] * self.corpus_size
        for word in tokens:
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for i, df in enumerate(self.doc_freqs):
                if word in df:
                    tf = df[word]
                    denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                    scores[i] += idf_val * (tf * (self.k1 + 1.0)) / denom

        # Pair scores with chunks and rank
        scored_pairs = [(self.chunks[i], scores[i]) for i in range(self.corpus_size) if scores[i] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]
