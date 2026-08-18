import time
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from app.classification.classifier import TicketClassifier
from app.escalation.escalation_model import EscalationRiskModel
from app.retrieval.embeddings import EmbeddingManager
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid import HybridRRFRetriever
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.historical import HistoricalTicketRetriever
from app.llm.generator import LLMResponseGenerator
from app.cache.redis_cache import RedisCacheManager
from app.observability.tracer import PipelineTracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SupportIntelligencePipeline:
    """End-to-End Customer Support Intelligence Pipeline."""
    
    def __init__(self, load_models: bool = True):
        self.classifier = None
        self.escalation_model = None
        self.embedder = None
        self.vector_store = None
        self.bm25 = None
        self.hybrid_rrf = HybridRRFRetriever()
        self.reranker = None
        self.historical_retriever = None
        self.generator = None
        self.cache = RedisCacheManager()
        self.tracer = PipelineTracer()
        
        if load_models:
            self.load_pipeline()

    def load_pipeline(self):
        logger.info("Initializing and loading Customer Support Intelligence Engine components...")
        
        # Load ML models
        try:
            self.classifier = TicketClassifier.load(settings.CLASSIFIER_PATH)
        except Exception as e:
            logger.warning(f"Could not load TicketClassifier ({e}). Running without classifier.")
            
        try:
            self.escalation_model = EscalationRiskModel.load(settings.ESCALATION_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Could not load EscalationRiskModel ({e}). Running without escalation model.")
            
        # Load Embeddings & RERANKER
        self.embedder = EmbeddingManager()
        try:
            self.reranker = CrossEncoderReranker()
        except Exception as e:
            logger.warning(f"Could not load CrossEncoderReranker ({e}). Will use raw retrieval scores.")

        # Load Knowledge Base Indices
        try:
            self.vector_store = VectorStore.load(settings.FAISS_INDEX_DIR, embedding_manager=self.embedder)
            bm25_file = settings.FAISS_INDEX_DIR / "bm25.pkl"
            if bm25_file.exists():
                with open(bm25_file, 'rb') as f:
                    self.bm25 = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load KB indices ({e}).")

        # Load Historical Tickets Index
        try:
            self.historical_retriever = HistoricalTicketRetriever.load(settings.HISTORICAL_FAISS_DIR, embedding_manager=self.embedder)
        except Exception as e:
            logger.warning(f"Could not load Historical Ticket Index ({e}).")

        # Load LLM Response Generator
        self.generator = LLMResponseGenerator()
        logger.info("SupportIntelligencePipeline components loaded successfully.")

    def run(self, query: str, user_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        # Step 0: Redis Cache Lookup
        cached_res = self.cache.get(query)
        if cached_res:
            return cached_res
            
        start_time = time.time()
        user_meta = user_metadata or {}
        
        # Step 1: Issue Classification
        category = "payment"
        classification_res = {}
        if self.classifier:
            try:
                clf_out = self.classifier.predict_proba(query)[0]
                category = clf_out["category"]
                classification_res = clf_out
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                
        # Step 2: Historical Ticket Retrieval (Real Bitext interactions)
        historical_cases = []
        top_hist_sim = 0.5
        if self.historical_retriever:
            try:
                historical_cases = self.historical_retriever.search_similar_tickets(query, top_k=settings.HISTORICAL_K)
                if historical_cases:
                    top_hist_sim = historical_cases[0].get("similarity_score", 0.5)
            except Exception as e:
                logger.error(f"Historical retrieval failed: {e}")

        # Step 3: Hybrid Retrieval (FAISS + BM25 + RRF)
        vector_res = []
        bm25_res = []
        if self.vector_store:
            vector_res = self.vector_store.search(query, top_k=settings.RETRIEVAL_K)
        if self.bm25:
            bm25_res = self.bm25.search(query, top_k=settings.RETRIEVAL_K)
            
        fused_candidates = self.hybrid_rrf.fuse_results(vector_res, bm25_res, top_k=settings.RETRIEVAL_K)

        # Step 4: Reranking with Cross-Encoder
        reranked_kb = fused_candidates
        if self.reranker and fused_candidates:
            reranked_kb = self.reranker.rerank(query, fused_candidates, top_k=settings.RERANK_K)

        # Step 5: Escalation Risk Score (Experimental Proxy Component)
        ticket_feature_payload = {
            "text": query,
            "category": category,
            "interaction_count": user_meta.get("interaction_count", 1),
            "similarity_score": top_hist_sim
        }
        escalation_info = {
            "escalation_score": 0.2,
            "escalation_level": "LOW",
            "recommendation": "Standard support flow",
            "disclaimer": "Experimental Proxy Escalation Model"
        }
        if self.escalation_model:
            try:
                escalation_info = self.escalation_model.predict_risk(ticket_feature_payload)[0]
            except Exception as e:
                logger.error(f"Escalation model failed: {e}")

        # Step 6: Grounded LLM Response Generation (with Low Evidence Detection)
        gen_output = self.generator.generate_response(
            query=query,
            kb_evidence=reranked_kb,
            historical_evidence=historical_cases[:2],
            escalation_info=escalation_info
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        result_payload = {
            "query": query,
            "predicted_category": classification_res,
            "escalation_prediction": escalation_info,
            "retrieved_kb_evidence": reranked_kb,
            "retrieved_historical_tickets": historical_cases,
            "response": gen_output,
            "latency_ms": latency_ms,
            "cached": False
        }
        
        # Step 7: Write to Cache if enabled
        self.cache.set(query, result_payload)
        
        # Step 8: Log observability metrics to MLflow / LangSmith
        self.tracer.log_pipeline_run(result_payload)
        
        return result_payload
