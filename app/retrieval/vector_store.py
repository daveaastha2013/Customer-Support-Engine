import os
import pickle
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from app.retrieval.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS vector store using IndexFlatIP for normalized cosine similarity search."""
    
    def __init__(self, embedding_manager: EmbeddingManager = None):
        self.embedder = embedding_manager or EmbeddingManager()
        self.dimension = self.embedder.dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        texts = [c["content"] for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts)
        
        self.index.reset()
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {self.dimension}.")

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.RETRIEVAL_K
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty!")
            return []
            
        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            item["vector_rank"] = rank + 1
            item["retrieval_source"] = "vector"
            results.append(item)
            
        return results

    def save(self, save_dir: Path = None):
        target_dir = save_dir or settings.FAISS_INDEX_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        
        index_file = target_dir / "index.faiss"
        chunks_file = target_dir / "chunks.pkl"
        
        faiss.write_index(self.index, str(index_file))
        with open(chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved FAISS index and chunks to {target_dir}")

    @classmethod
    def load(cls, save_dir: Path = None, embedding_manager: EmbeddingManager = None):
        target_dir = save_dir or settings.FAISS_INDEX_DIR
        index_file = target_dir / "index.faiss"
        chunks_file = target_dir / "chunks.pkl"
        
        if not index_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(f"FAISS index files not found in {target_dir}")
            
        instance = cls(embedding_manager=embedding_manager)
        instance.index = faiss.read_index(str(index_file))
        with open(chunks_file, 'rb') as f:
            instance.chunks = pickle.load(f)
        logger.info(f"Loaded FAISS index with {instance.index.ntotal} vectors from {target_dir}")
        return instance
