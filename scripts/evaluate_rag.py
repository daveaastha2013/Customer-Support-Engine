import json
import time
import logging
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from app.rag.pipeline import SupportIntelligencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Benchmark evaluation set with 30 diverse test tickets
EVAL_DATASET = [
    # High-Evidence Payment Queries
    {"query": "My credit card was charged twice for order #9821.", "expected_category": "payment", "expected_doc": "payment_policy.md", "is_low_evidence": False},
    {"query": "What payment methods do you accept?", "expected_category": "payment", "expected_doc": "payment_policy.md", "is_low_evidence": False},
    {"query": "How do I request a refund for a failed transaction?", "expected_category": "refund", "expected_doc": "refund_policy.md", "is_low_evidence": False},
    {"query": "Why was my debit card payment declined?", "expected_category": "payment", "expected_doc": "payment_policy.md", "is_low_evidence": False},
    {"query": "Can I pay using PayPal or Apple Pay?", "expected_category": "payment", "expected_doc": "payment_policy.md", "is_low_evidence": False},
    
    # High-Evidence Delivery Queries
    {"query": "Where is my package? The tracking number is 88210.", "expected_category": "delivery", "expected_doc": "delivery_policy.md", "is_low_evidence": False},
    {"query": "How long does standard shipping take?", "expected_category": "delivery", "expected_doc": "delivery_policy.md", "is_low_evidence": False},
    {"query": "Do you offer international shipping?", "expected_category": "delivery", "expected_doc": "delivery_policy.md", "is_low_evidence": False},
    {"query": "My order says delivered but I haven't received it.", "expected_category": "delivery", "expected_doc": "delivery_policy.md", "is_low_evidence": False},
    {"query": "Can I change my shipping address after placing an order?", "expected_category": "delivery", "expected_doc": "delivery_policy.md", "is_low_evidence": False},
    
    # High-Evidence Refund Queries
    {"query": "What is your return and refund window?", "expected_category": "refund", "expected_doc": "refund_policy.md", "is_low_evidence": False},
    {"query": "I received a damaged item and want my money back.", "expected_category": "refund", "expected_doc": "refund_policy.md", "is_low_evidence": False},
    {"query": "How many days until the refund hits my bank account?", "expected_category": "refund", "expected_doc": "refund_policy.md", "is_low_evidence": False},
    {"query": "Are return shipping fees refundable?", "expected_category": "refund", "expected_doc": "refund_policy.md", "is_low_evidence": False},
    
    # High-Evidence Account Queries
    {"query": "How do I reset my account password?", "expected_category": "account", "expected_doc": "account_management.md", "is_low_evidence": False},
    {"query": "I can't log in to my account.", "expected_category": "account", "expected_doc": "account_management.md", "is_low_evidence": False},
    {"query": "How do I enable two-factor authentication?", "expected_category": "account", "expected_doc": "account_management.md", "is_low_evidence": False},
    {"query": "How can I delete my personal account and data?", "expected_category": "account", "expected_doc": "account_management.md", "is_low_evidence": False},

    # High-Evidence Subscription Queries
    {"query": "How do I cancel my monthly subscription plan?", "expected_category": "subscription", "expected_doc": "subscription_policy.md", "is_low_evidence": False},
    {"query": "Can I upgrade from basic to premium subscription?", "expected_category": "subscription", "expected_doc": "subscription_policy.md", "is_low_evidence": False},
    {"query": "Will I get a prorated refund if I cancel my subscription mid-month?", "expected_category": "subscription", "expected_doc": "subscription_policy.md", "is_low_evidence": False},
    {"query": "How do I pause my active subscription for a month?", "expected_category": "subscription", "expected_doc": "subscription_policy.md", "is_low_evidence": False},

    # Out-of-Scope / Low-Evidence Queries
    {"query": "What is the capital of France?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "How do I bake a chocolate cake?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "What is quantum computing?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "Can you solve this math equation 2 + 2 * 10?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "Who won the FIFA world cup in 2022?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "What are the rules for playing chess?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "What is the weather forecast in Tokyo today?", "expected_category": "account", "expected_doc": None, "is_low_evidence": True},
    {"query": "Tell me a joke about software engineers.", "expected_category": "account", "expected_doc": None, "is_low_evidence": True}
]

def run_evaluation():
    logger.info("Starting RAG & Pipeline Evaluation Framework...")
    pipeline = SupportIntelligencePipeline()
    
    cat_correct = 0
    hit_count = 0
    reciprocal_ranks = []
    low_evidence_correct = 0
    latencies = []
    
    eval_results = []
    
    for idx, item in enumerate(EVAL_DATASET, 1):
        query = item["query"]
        expected_cat = item["expected_category"]
        expected_doc = item["expected_doc"]
        should_be_low_evidence = item["is_low_evidence"]
        
        t0 = time.time()
        res = pipeline.run(query)
        latency = round((time.time() - t0) * 1000, 2)
        latencies.append(latency)
        
        # 1. Classification Evaluation
        pred_cat = res["predicted_category"].get("category", "")
        cat_match = (pred_cat == expected_cat)
        if cat_match:
            cat_correct += 1
            
        # 2. Retrieval Evaluation (MRR & Hit Rate)
        retrieved_docs = []
        for chunk in res.get("retrieved_kb_evidence", []):
            src = chunk.get("metadata", {}).get("source", "") or chunk.get("source", "")
            if src:
                retrieved_docs.append(src)
        rank = 0
        hit = False
        if expected_doc:
            for r_idx, doc in enumerate(retrieved_docs, 1):
                if expected_doc in doc:
                    rank = r_idx
                    hit = True
                    break
        
        if hit:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
        elif expected_doc:
            reciprocal_ranks.append(0.0)
            
        # 3. Low-Evidence Detection Evaluation
        actual_low_ev = res["response"].get("low_evidence", False)
        low_ev_match = (actual_low_ev == should_be_low_evidence)
        if low_ev_match:
            low_evidence_correct += 1
            
        eval_results.append({
            "test_id": idx,
            "query": query,
            "expected_category": expected_cat,
            "predicted_category": pred_cat,
            "category_match": cat_match,
            "expected_doc": expected_doc,
            "retrieved_docs": retrieved_docs[:3],
            "retrieval_hit": hit,
            "reciprocal_rank": (1.0 / rank) if hit else 0.0,
            "expected_low_evidence": should_be_low_evidence,
            "actual_low_evidence": actual_low_ev,
            "low_evidence_match": low_ev_match,
            "latency_ms": latency
        })

    # Summary Metrics Calculation
    total_samples = len(EVAL_DATASET)
    high_ev_samples = sum(1 for d in EVAL_DATASET if not d["is_low_evidence"])
    
    cat_accuracy = round(cat_correct / total_samples, 4)
    hit_rate = round(hit_count / high_ev_samples, 4)
    mrr = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0
    low_evidence_accuracy = round(low_evidence_correct / total_samples, 4)
    avg_latency_ms = round(float(np.mean(latencies)), 2)
    p95_latency_ms = round(float(np.percentile(latencies, 95)), 2)

    report = {
        "summary": {
            "total_test_queries": total_samples,
            "high_evidence_queries": high_ev_samples,
            "low_evidence_queries": total_samples - high_ev_samples,
            "category_classification_accuracy": cat_accuracy,
            "retrieval_hit_rate_at_k": hit_rate,
            "retrieval_mrr": mrr,
            "low_evidence_detection_accuracy": low_evidence_accuracy,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms
        },
        "detailed_results": eval_results
    }
    
    out_dir = settings.EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_report.json"
    
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("=" * 60)
    logger.info("EVALUATION BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Test Queries: {total_samples}")
    logger.info(f"Category Classification Accuracy: {cat_accuracy * 100:.1f}%")
    logger.info(f"Retrieval Hit Rate @ K: {hit_rate * 100:.1f}%")
    logger.info(f"Retrieval MRR: {mrr:.4f}")
    logger.info(f"Low-Evidence Detection Accuracy: {low_evidence_accuracy * 100:.1f}%")
    logger.info(f"Avg Latency: {avg_latency_ms} ms | P95 Latency: {p95_latency_ms} ms")
    logger.info(f"Report saved to: {out_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_evaluation()
