import json
import logging
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from app.retrieval.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HistoricalTicketRetriever:
    """Retrieves semantically similar past tickets and their real resolutions."""
    
    def __init__(self, embedding_manager: EmbeddingManager = None):
        self.embedder = embedding_manager or EmbeddingManager()
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        self.tickets: List[Dict[str, Any]] = []

    def build_index(self, tickets_file: Path = None):
        target_file = tickets_file or (settings.PROCESSED_DATA_DIR / "historical_tickets.json")
        if not target_file.exists():
            logger.warning(f"Historical tickets file not found at {target_file}")
            return
            
        with open(target_file, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)
            
        queries = [t["query"] for t in self.tickets]
        logger.info(f"Indexing {len(queries)} historical support tickets...")
        embeddings = self.embedder.embed_texts(queries)
        
        self.index.reset()
        self.index.add(embeddings)
        logger.info(f"Historical tickets FAISS index built with {self.index.ntotal} records.")

    def search_similar_tickets(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.HISTORICAL_K
        if self.index.ntotal == 0:
            logger.warning("Historical tickets index is empty!")
            return []
            
        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.tickets):
                continue
            item = dict(self.tickets[idx])
            item["similarity_score"] = float(score)
            item["historical_rank"] = rank + 1
            item["retrieval_type"] = "historical_ticket"
            # Format as chunk content for context building
            item["content"] = f"[HISTORICAL TICKET {item['ticket_id']}] Issue: {item['query']} | Resolution: {item['resolution']}"
            item["chunk_id"] = f"HIST-{item['ticket_id']}"
            item["metadata"] = {
                "document_title": f"Historical Ticket {item['ticket_id']}",
                "category": item["category"],
                "source": "historical_tickets.json",
                "section": "Past Resolutions",
                "source_type": item.get("source_type", "REAL_BITEXT_RECORD")
            }
            results.append(item)
            
        return results

    def save(self, save_dir: Path = None):
        target_dir = save_dir or settings.HISTORICAL_FAISS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target_dir / "historical_index.faiss"))
        with open(target_dir / "tickets.pkl", 'wb') as f:
            pickle.dump(self.tickets, f)
        logger.info(f"Saved HistoricalTicketRetriever to {target_dir}")

    @classmethod
    def load(cls, save_dir: Path = None, embedding_manager: EmbeddingManager = None):
        target_dir = save_dir or settings.HISTORICAL_FAISS_DIR
        index_file = target_dir / "historical_index.faiss"
        tickets_file = target_dir / "tickets.pkl"
        
        if not index_file.exists() or not tickets_file.exists():
            raise FileNotFoundError(f"Historical ticket index files not found in {target_dir}")
            
        instance = cls(embedding_manager=embedding_manager)
        instance.index = faiss.read_index(str(index_file))
        with open(tickets_file, 'rb') as f:
            instance.tickets = pickle.load(f)
        logger.info(f"Loaded HistoricalTicketRetriever with {instance.index.ntotal} records from {target_dir}")
        return instance
