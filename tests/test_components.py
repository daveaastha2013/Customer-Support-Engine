import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from app.classification.classifier import TicketClassifier
from app.escalation.escalation_model import EscalationRiskModel
from app.retrieval.hybrid import HybridRRFRetriever
from app.rag.pipeline import SupportIntelligencePipeline

def test_classifier_loading_and_prediction():
    """Test loading trained TicketClassifier and making a prediction."""
    assert settings.CLASSIFIER_PATH.exists(), "Classifier joblib file missing!"
    classifier = TicketClassifier.load(settings.CLASSIFIER_PATH)
    
    query = "My credit card payment was declined."
    res = classifier.predict_proba(query)
    
    assert len(res) == 1
    assert "category" in res[0]
    assert "confidence" in res[0]
    assert "probabilities" in res[0]
    assert res[0]["category"] == "payment"

def test_escalation_model_prediction_and_disclaimer():
    """Test EscalationRiskModel output and disclaimer presence."""
    assert settings.ESCALATION_MODEL_PATH.exists(), "Escalation model joblib file missing!"
    esc_model = EscalationRiskModel.load(settings.ESCALATION_MODEL_PATH)
    
    payload = {
        "text": "Urgent! Contacted support 5 times regarding payment!",
        "category": "payment",
        "interaction_count": 5,
        "similarity_score": 0.8
    }
    
    res = esc_model.predict_risk(payload)
    assert len(res) == 1
    assert "escalation_score" in res[0]
    assert "escalation_level" in res[0]
    assert "disclaimer" in res[0]
    assert "EXPERIMENTAL PROXY MODEL DISCLAIMER" in res[0]["disclaimer"]

def test_hybrid_rrf_fusion():
    """Test Reciprocal Rank Fusion blending logic."""
    vec_results = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.8}]
    bm25_results = [{"chunk_id": "c2", "score": 10.0}, {"chunk_id": "c3", "score": 5.0}]
    
    rrf = HybridRRFRetriever(rrf_k=60)
    fused = rrf.fuse_results(vec_results, bm25_results, top_k=3)
    
    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "c2" # c2 appears in both, so gets highest RRF rank score!

def test_full_support_intelligence_pipeline():
    """Integration test for full SupportIntelligencePipeline execution."""
    pipeline = SupportIntelligencePipeline()
    
    query = "My order has not arrived yet."
    result = pipeline.run(query)
    
    assert "predicted_category" in result
    assert "escalation_prediction" in result
    assert "retrieved_kb_evidence" in result
    assert "response" in result
    assert "latency_ms" in result
    assert result["latency_ms"] > 0
