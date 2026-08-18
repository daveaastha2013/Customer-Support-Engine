import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from textblob import TextBlob
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DISCLAIMER_MSG = (
    "EXPERIMENTAL PROXY MODEL DISCLAIMER: This model uses heuristic proxy labels derived from "
    "sentiment, message length, and contact frequency. It is an experimental modeling component "
    "and DOES NOT represent real-world customer escalation outcomes."
)

CATEGORIES = ['payment', 'account', 'delivery', 'refund', 'technical', 'subscription']
URGENT_KEYWORDS = ['twice', 'urgent', 'immediately', 'cancel', 'lawyer', 'failed', 'shattered', 'crushed', 'contacted', 'stolen', 'pending']

class EscalationRiskModel:
    """XGBoost model for predicting ticket escalation risk score."""
    
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=settings.SEED,
            eval_metric='logloss'
        )
        self.feature_names = [
            'sentiment_polarity', 'sentiment_subjectivity', 'message_length',
            'interaction_count', 'urgency_keywords', 'similarity_score'
        ] + [f"cat_{c}" for c in CATEGORIES]
        
    def extract_features(self, df_or_dict):
        """Converts raw ticket text or DataFrame into feature array."""
        if isinstance(df_or_dict, dict):
            df = pd.DataFrame([df_or_dict])
        elif isinstance(df_or_dict, pd.DataFrame):
            df = df_or_dict.copy()
        else:
            raise ValueError("Input must be a dict or pandas DataFrame")
            
        rows = []
        for idx, row in df.iterrows():
            text = str(row.get('text', ''))
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            length = len(text)
            
            # Urgency count
            t_lower = text.lower()
            urgency_count = sum(1 for kw in URGENT_KEYWORDS if kw in t_lower)
            
            # Interaction count
            interactions = row.get('interaction_count', 1)
            if 'twice' in t_lower or 'contacted' in t_lower:
                interactions = max(interactions, 2)
                
            similarity_score = float(row.get('similarity_score', 0.5))
            category = str(row.get('category', 'payment')).lower()
            
            feat_vec = [polarity, subjectivity, length, interactions, urgency_count, similarity_score]
            
            # One-hot encode category
            for cat in CATEGORIES:
                feat_vec.append(1.0 if category == cat else 0.0)
                
            rows.append(feat_vec)
            
        return np.array(rows, dtype=np.float32)

    def fit(self, df_train, labels):
        logger.info(f"Extracting features for {len(df_train)} training rows...")
        X = self.extract_features(df_train)
        logger.info(f"Training XGBoost EscalationRiskModel on shape {X.shape}...")
        self.model.fit(X, labels)
        logger.info("EscalationRiskModel training complete.")
        return self

    def predict_risk(self, ticket_data):
        X = self.extract_features(ticket_data)
        probs = self.model.predict_proba(X)[:, 1]
        
        results = []
        for prob in probs:
            score = float(prob)
            if score >= 0.7:
                level = "HIGH"
                recommendation = "Immediate escalation to senior support tier recommended."
            elif score >= 0.4:
                level = "MEDIUM"
                recommendation = "Monitor ticket. Standard support workflow with prior history review."
            else:
                level = "LOW"
                recommendation = "Standard automated response flow."
                
            results.append({
                "escalation_score": round(score, 4),
                "escalation_level": level,
                "recommendation": recommendation,
                "disclaimer": DISCLAIMER_MSG
            })
        return results

    def evaluate(self, df_eval, true_labels):
        X = self.extract_features(df_eval)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        
        roc_auc = roc_auc_score(true_labels, probs)
        f1 = f1_score(true_labels, preds)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        cm = confusion_matrix(true_labels, preds)
        
        metrics = {
            "roc_auc": round(float(roc_auc), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "confusion_matrix": cm.tolist(),
            "disclaimer": DISCLAIMER_MSG
        }
        return metrics

    def save(self, path=None):
        save_path = path or settings.ESCALATION_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, save_path)
        logger.info(f"Saved EscalationRiskModel to {save_path}")

    @classmethod
    def load(cls, path=None):
        load_path = path or settings.ESCALATION_MODEL_PATH
        if not Path(load_path).exists():
            raise FileNotFoundError(f"No escalation model found at {load_path}")
        data = joblib.load(load_path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        logger.info(f"Loaded EscalationRiskModel from {load_path}")
        return instance
