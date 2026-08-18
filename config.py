import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Customer Support Intelligence Engine"
    VERSION: str = "1.0.0"
    BASE_DIR: Path = BASE_DIR
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "customer-support-intelligence")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR}/mlruns")
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    REDIS_TTL_SECONDS: int = int(os.getenv("REDIS_TTL_SECONDS", 3600))
    
    # Model Names
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
    
    # RAG Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 15))
    RERANK_K: int = int(os.getenv("RERANK_K", 5))
    HISTORICAL_K: int = int(os.getenv("HISTORICAL_K", 3))
    RRF_K: int = int(os.getenv("RRF_K", 60))
    
    # Paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    KB_DIR: Path = BASE_DIR / "data" / "knowledge_base"
    EVAL_DIR: Path = BASE_DIR / "data" / "evaluation_results"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Model Artifact Paths
    CLASSIFIER_PATH: Path = BASE_DIR / "models" / "ticket_classifier.joblib"
    CLASSIFIER_METRICS_PATH: Path = BASE_DIR / "models" / "classifier_metrics.json"
    ESCALATION_MODEL_PATH: Path = BASE_DIR / "models" / "escalation_model.joblib"
    ESCALATION_METRICS_PATH: Path = BASE_DIR / "models" / "escalation_metrics.json"
    FAISS_INDEX_DIR: Path = BASE_DIR / "models" / "faiss_index"
    HISTORICAL_FAISS_DIR: Path = BASE_DIR / "models" / "historical_faiss_index"
    
    SEED: int = 42

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
