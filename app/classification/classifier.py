import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TicketClassifier:
    """TF-IDF + Logistic Regression ticket classifier for support category prediction."""
    
    def __init__(self, max_features=10000, C=1.0):
        self.max_features = max_features
        self.C = C
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features, sublinear_tf=True)),
            ('clf', LogisticRegression(C=self.C, max_iter=1000, random_state=settings.SEED, multi_class='multinomial'))
        ])
        self.classes_ = None

    def fit(self, texts, labels):
        logger.info(f"Training TicketClassifier on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.named_steps['clf'].classes_.tolist()
        logger.info(f"TicketClassifier trained. Classes: {self.classes_}")
        return self

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        probs = self.pipeline.predict_proba(texts)
        results = []
        for prob in probs:
            top_idx = int(np.argmax(prob))
            top_class = self.classes_[top_idx]
            top_confidence = float(prob[top_idx])
            class_probs = {cls: float(p) for cls, p in zip(self.classes_, prob)}
            results.append({
                "category": top_class,
                "confidence": round(top_confidence, 4),
                "probabilities": class_probs
            })
        return results

    def evaluate(self, texts, true_labels):
        preds = self.predict(texts)
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro')
        precision = precision_score(true_labels, preds, average='macro', zero_division=0)
        recall = recall_score(true_labels, preds, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, preds, labels=self.classes_)
        report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
        
        metrics = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "confusion_matrix": cm.tolist(),
            "classes": self.classes_,
            "report": report
        }
        return metrics

    def save(self, path=None):
        save_path = path or settings.CLASSIFIER_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes_": self.classes_}, save_path)
        logger.info(f"Saved TicketClassifier to {save_path}")

    @classmethod
    def load(cls, path=None):
        load_path = path or settings.CLASSIFIER_PATH
        if not Path(load_path).exists():
            raise FileNotFoundError(f"No classifier model found at {load_path}")
        data = joblib.load(load_path)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.classes_ = data["classes_"]
        logger.info(f"Loaded TicketClassifier from {load_path}")
        return instance
