import logging
from typing import List, Dict, Any
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HybridRRFRetriever:
    """Combines vector search and BM25 keyword search using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, rrf_k: int = None):
        self.rrf_k = rrf_k or settings.RRF_K

    def fuse_results(self, vector_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.RETRIEVAL_K
        rrf_scores = defaultdict(float)
        items_map = {}
        
        # Process Vector Results
        for rank, item in enumerate(vector_results, start=1):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            if chunk_id not in items_map:
                items_map[chunk_id] = dict(item)
                items_map[chunk_id]["vector_rank"] = rank
                items_map[chunk_id]["vector_score"] = item.get("score", 0.0)

        # Process BM25 Results
        for rank, item in enumerate(bm25_results, start=1):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            if chunk_id not in items_map:
                items_map[chunk_id] = dict(item)
            items_map[chunk_id]["bm25_rank"] = rank
            items_map[chunk_id]["bm25_score"] = item.get("score", 0.0)

        # Sort items by RRF score descending
        fused_list = []
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]
        
        for rank, chunk_id in enumerate(sorted_ids, start=1):
            item = items_map[chunk_id]
            item["rrf_score"] = round(float(rrf_scores[chunk_id]), 6)
            item["rrf_rank"] = rank
            item["retrieval_strategy"] = "hybrid_rrf"
            fused_list.append(item)
            
        logger.info(f"RRF Hybrid fusion combined {len(vector_results)} vector & {len(bm25_results)} BM25 results into top-{len(fused_list)} chunks.")
        return fused_list
