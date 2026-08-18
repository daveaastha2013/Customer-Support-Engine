import json
import logging
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from app.escalation.escalation_model import EscalationRiskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_and_eval():
    train_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "test.csv")
    
    logger.info(f"Loaded datasets -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    model = EscalationRiskModel()
    model.fit(train_df, train_df['is_escalated_proxy'])
    
    logger.info("Evaluating Escalation Model on Validation set...")
    val_metrics = model.evaluate(val_df, val_df['is_escalated_proxy'])
    logger.info(f"Validation ROC-AUC: {val_metrics['roc_auc']}, F1: {val_metrics['f1_score']}")
    
    logger.info("Evaluating Escalation Model on Test set...")
    test_metrics = model.evaluate(test_df, test_df['is_escalated_proxy'])
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']}, F1: {test_metrics['f1_score']}")
    
    # Save model and metrics
    model.save(settings.ESCALATION_MODEL_PATH)
    
    metrics_payload = {
        "validation": val_metrics,
        "test": test_metrics
    }
    
    with open(settings.ESCALATION_METRICS_PATH, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved escalation metrics to {settings.ESCALATION_METRICS_PATH}")

if __name__ == "__main__":
    train_and_eval()
