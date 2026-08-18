import logging
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from app.rag.ingestion import DocumentIngestion
from app.retrieval.embeddings import EmbeddingManager
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.historical import HistoricalTicketRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_all_indices():
    logger.info("Starting index construction pipeline...")
    
    # 1. Ingest Knowledge Base Docs
    ingestor = DocumentIngestion()
    kb_chunks = ingestor.load_kb_documents()
    
    # 2. Build KB FAISS Vector Index
    embedder = EmbeddingManager()
    vector_store = VectorStore(embedding_manager=embedder)
    vector_store.build_index(kb_chunks)
    vector_store.save(settings.FAISS_INDEX_DIR)
    
    # 3. Build KB BM25 Index
    bm25 = BM25Retriever()
    bm25.build_index(kb_chunks)
    bm25_file = settings.FAISS_INDEX_DIR / "bm25.pkl"
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 retriever index to {bm25_file}")
    
    # 4. Build Historical Tickets Index
    historical_retriever = HistoricalTicketRetriever(embedding_manager=embedder)
    historical_retriever.build_index()
    historical_retriever.save(settings.HISTORICAL_FAISS_DIR)
    
    logger.info("All RAG indices successfully built and saved!")

if __name__ == "__main__":
    build_all_indices()
