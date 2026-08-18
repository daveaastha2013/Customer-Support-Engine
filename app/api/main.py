import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from app.rag.pipeline import SupportIntelligencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-Grade Customer Support Intelligence Engine (RAG, Ticket Classification, Escalation Risk Prediction & Observability)"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy pipeline singleton instance
pipeline: Optional[SupportIntelligencePipeline] = None

@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("Initializing SupportIntelligencePipeline on FastAPI startup...")
    pipeline = SupportIntelligencePipeline(load_models=True)

# Pydantic Schemas
class TicketRequest(BaseModel):
    text: str = Field(..., example="My credit card was charged twice for order #9821.")
    user_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, example={"interaction_count": 2})

class ClassifyRequest(BaseModel):
    text: str = Field(..., example="I want to cancel my subscription.")

class EscalationRequest(BaseModel):
    text: str = Field(..., example="I have messaged support 4 times with no response!")
    category: Optional[str] = "payment"
    interaction_count: Optional[int] = 3
    similarity_score: Optional[float] = 0.5

# Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint validating system modules and status."""
    global pipeline
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "project": settings.PROJECT_NAME,
        "pipeline_loaded": pipeline is not None,
        "classifier_loaded": pipeline.classifier is not None if pipeline else False,
        "escalation_model_loaded": pipeline.escalation_model is not None if pipeline else False,
        "vector_store_loaded": pipeline.vector_store is not None if pipeline else False,
        "redis_connected": pipeline.cache.enabled if pipeline else False
    }

@app.post("/api/v1/predict")
def predict_ticket(req: TicketRequest):
    """End-to-End Pipeline: Classify, Retrieve, Rerank, Predict Escalation Risk, and Generate Grounded Response."""
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    try:
        result = pipeline.run(query=req.text, user_metadata=req.user_metadata)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/classify")
def classify_ticket(req: ClassifyRequest):
    """Stand-alone ticket classification endpoint."""
    global pipeline
    if not pipeline or not pipeline.classifier:
        raise HTTPException(status_code=503, detail="Classifier model unavailable")
        
    try:
        res = pipeline.classifier.predict_proba(req.text)[0]
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/escalation")
def predict_escalation(req: EscalationRequest):
    """Stand-alone experimental escalation risk prediction endpoint."""
    global pipeline
    if not pipeline or not pipeline.escalation_model:
        raise HTTPException(status_code=503, detail="Escalation model unavailable")
        
    try:
        payload = {
            "text": req.text,
            "category": req.category,
            "interaction_count": req.interaction_count,
            "similarity_score": req.similarity_score
        }
        res = pipeline.escalation_model.predict_risk(payload)[0]
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
