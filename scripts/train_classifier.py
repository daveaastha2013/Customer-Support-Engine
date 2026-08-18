import json
import logging
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from app.classification.classifier import TicketClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_and_eval():
    train_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "test.csv")
    
    logger.info(f"Loaded datasets -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    clf = TicketClassifier()
    clf.fit(train_df['text'], train_df['category'])
    
    logger.info("Evaluating on Validation set...")
    val_metrics = clf.evaluate(val_df['text'], val_df['category'])
    logger.info(f"Validation Accuracy: {val_metrics['accuracy']}, Macro F1: {val_metrics['macro_f1']}")
    
    logger.info("Evaluating on Test set...")
    test_metrics = clf.evaluate(test_df['text'], test_df['category'])
    logger.info(f"Test Accuracy: {test_metrics['accuracy']}, Macro F1: {test_metrics['macro_f1']}")
    
    # Save model and metrics
    clf.save(settings.CLASSIFIER_PATH)
    
    metrics_payload = {
        "validation": val_metrics,
        "test": test_metrics
    }
    
    with open(settings.CLASSIFIER_METRICS_PATH, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved classification metrics to {settings.CLASSIFIER_METRICS_PATH}")

if __name__ == "__main__":
    train_and_eval()
