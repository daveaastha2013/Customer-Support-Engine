import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    """BM25Okapi keyword retriever for complement to vector search."""
    
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        corpus = [tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.RETRIEVAL_K
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is empty!")
            return []
            
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by BM25 score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue
            item = dict(self.chunks[idx])
            item["score"] = score
            item["bm25_rank"] = rank + 1
            item["retrieval_source"] = "bm25"
            results.append(item)
            
        return results
