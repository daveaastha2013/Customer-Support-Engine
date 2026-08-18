import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Configurable SentenceTransformer embedding manager."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].astype(np.float32)
