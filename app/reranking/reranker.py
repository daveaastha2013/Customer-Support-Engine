import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-Encoder reranker to select highest-quality evidence from candidate set."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        logger.info(f"Initializing CrossEncoder reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.RERANK_K
        if not candidates:
            return []
            
        pairs = [[query, item.get("content", item.get("resolution", ""))] for item in candidates]
        scores = self.model.predict(pairs)
        
        reranked = []
        for idx, score in enumerate(scores):
            item = dict(candidates[idx])
            item["reranker_score"] = round(float(score), 4)
            reranked.append(item)
            
        # Sort by reranker_score descending
        reranked = sorted(reranked, key=lambda x: x["reranker_score"], reverse=True)[:k]
        
        for rank, item in enumerate(reranked, start=1):
            item["final_rank"] = rank
            
        logger.info(f"CrossEncoder reranked {len(candidates)} candidate items down to top-{len(reranked)} evidence items.")
        return reranked
