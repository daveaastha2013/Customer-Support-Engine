import os
import logging
from typing import Dict, Any
import mlflow
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PipelineTracer:
    """Observability manager tracking pipeline runs with MLflow and LangSmith."""
    
    def __init__(self, experiment_name: str = "customer_support_intelligence"):
        self.experiment_name = experiment_name
        self.mlflow_enabled = False
        
        try:
            mlflow.set_tracking_uri("file://" + str(settings.BASE_DIR / "mlruns"))
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_enabled = True
            logger.info(f"Initialized MLflow tracking URI: file://{settings.BASE_DIR / 'mlruns'}")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow tracking ({e}).")

    def log_pipeline_run(self, result: Dict[str, Any]):
        """Logs pipeline metrics, parameters, and outputs to MLflow."""
        if not self.mlflow_enabled:
            return
            
        try:
            with mlflow.start_run(run_name="pipeline_inference_step", nested=True):
                # Parameters
                mlflow.log_param("query", result.get("query", "")[:100])
                mlflow.log_param("category", result.get("predicted_category", {}).get("category", "unknown"))
                mlflow.log_param("escalation_level", result.get("escalation_prediction", {}).get("escalation_level", "UNKNOWN"))
                mlflow.log_param("cached", result.get("cached", False))
                
                # Metrics
                mlflow.log_metric("latency_ms", result.get("latency_ms", 0.0))
                mlflow.log_metric("classifier_confidence", result.get("predicted_category", {}).get("confidence", 0.0))
                mlflow.log_metric("escalation_score", result.get("escalation_prediction", {}).get("escalation_score", 0.0))
                
                kb_count = len(result.get("retrieved_kb_evidence", []))
                hist_count = len(result.get("retrieved_historical_tickets", []))
                mlflow.log_metric("kb_chunks_retrieved", kb_count)
                mlflow.log_metric("historical_tickets_retrieved", hist_count)
                
                logger.info("Logged pipeline run to MLflow experiment.")
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")
