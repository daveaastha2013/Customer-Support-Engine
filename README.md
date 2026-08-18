# Customer Support Intelligence Engine

A production-grade **Customer Support Intelligence Engine** designed for customer support teams. Instead of serving as a generic chatbot, this system automates end-to-end support triage: it **classifies incoming support tickets**, **retrieves relevant knowledge base policies and historical resolution pairs**, **predicts experimental escalation risk**, and **generates grounded LLM responses** with low-evidence fallback detection and full observability.

---

## Key Features

1. **Multi-Class Ticket Classification**: TF-IDF + Logistic Regression model categorizing tickets into `payment`, `delivery`, `refund`, `account`, and `subscription` with **99.8% accuracy**.
2. **Hybrid RAG Retrieval**: Combines dense vector search (`FAISS` + `all-MiniLM-L6-v2`) and sparse keyword search (`BM25`) fused via **Reciprocal Rank Fusion (RRF)**.
3. **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank top candidates, improving precision and document relevance.
4. **Historical Support Interaction Matching**: Index of over 26,000 real Bitext customer interactions to provide support agents with true past resolutions.
5. **Experimental Escalation Risk Predictor**: XGBoost model predicting potential escalation risk based on ticket sentiment, length, contact frequency, and historical similarity.
6. **Low-Evidence Guardrail Detection**: Identifies out-of-scope or unverified queries (e.g. "How to bake a cake") and returns a deterministic, grounded fallback response rather than forcing hallucinated answers.
7. **Redis Performance Optimization**: Optional caching layer with MD5 key hashing and graceful degradation if Redis is offline.
8. **LLMOps & Observability**: Complete experiment tracking via **MLflow** and optional **LangSmith** pipeline tracing.
9. **Interactive Streamlit Dashboard**: Clean 2-tab web console (`http://localhost:8502`) providing live ticket triage, full-width grounded LLM response rendering, RAG citation links, probability charts, and interactive evaluation benchmark reports.

---

## Experimental Proxy Escalation Disclaimer

> **IMPORTANT**: The escalation prediction model is an **experimental component** trained on heuristic proxy labels (derived from negative sentiment, message length, and prior contact count). These proxy labels **do not represent real-world customer escalation outcomes**. The resulting risk scores and metrics are strictly experimental and intended for system validation.

---

## 📊 Benchmark & Evaluation Metrics

| Metric | Score | Details / Benchmark |
| :--- | :--- | :--- |
| **Ticket Classification Accuracy** | **99.8%** | Evaluated on 5,375 Bitext test tickets |
| **Retrieval Hit Rate @ K** | **40.9%** | Evaluated on 30 benchmark test queries |
| **Retrieval MRR (Mean Reciprocal Rank)** | **0.3864** | Evaluated across high-evidence queries |
| **Low-Evidence Detection Accuracy** | **73.3%** | Accurately flags ungrounded/out-of-scope inputs |
| **Average End-to-End Latency** | **224.8 ms** | Sub-250ms average pipeline latency |

---

## Repository Structure

```
├── app/
│   ├── api/                  # FastAPI REST API endpoints
│   │   └── main.py
│   ├── cache/                # Redis cache manager with fallback
│   │   └── redis_cache.py
│   ├── classification/       # TF-IDF + Logistic Regression Ticket Classifier
│   │   └── classifier.py
│   ├── escalation/           # XGBoost Experimental Escalation Model
│   │   └── escalation_model.py
│   ├── llm/                  # Prompt engineering & LLM generator
│   │   ├── generator.py
│   │   └── prompts.py
│   ├── observability/        # MLflow & LangSmith tracing
│   │   └── tracer.py
│   ├── rag/                  # Ingestion & End-to-End Orchestrator
│   │   ├── ingestion.py
│   │   └── pipeline.py
│   ├── reranking/            # Cross-Encoder Reranker
│   │   └── reranker.py
│   ├── retrieval/            # Embeddings, FAISS, BM25, Hybrid RRF, Historical
│   │   ├── bm25_retriever.py
│   │   ├── embeddings.py
│   │   ├── historical.py
│   │   ├── hybrid.py
│   │   └── vector_store.py
│   └── ui/                   # Streamlit Dashboard Console
│       └── streamlit_app.py
├── config.py                 # Pydantic environment configuration
├── data/                     # Raw, processed, knowledge base, & evaluation data
├── models/                   # Saved ML models and FAISS vector indices
├── scripts/                  # Data acquisition, training, indexing & evaluation scripts
│   ├── build_indices.py
│   ├── download_data.py
│   ├── evaluate_rag.py
│   ├── preprocess_data.py
│   ├── train_classifier.py
│   └── train_escalation.py
├── tests/                    # Pytest test suite
│   └── test_components.py
├── Dockerfile                # Container image build file
├── docker-compose.yml        # Multi-container service stack
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quickstart Guide

### 1. Local Environment Setup

```bash
# Clone repository
cd rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to include your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_ENABLED=false
```

### 3. Run Pipeline Data Setup & Model Training

```bash
# Step 1: Download Bitext dataset
python scripts/download_data.py

# Step 2: Preprocess dataset & generate Knowledge Base
python scripts/preprocess_data.py

# Step 3: Train Ticket Classifier
python scripts/train_classifier.py

# Step 4: Train Escalation Risk Model
python scripts/train_escalation.py

# Step 5: Build FAISS Vector & BM25 Indices
python scripts/build_indices.py
```

### 4. Run RAG Benchmark Evaluation

```bash
python scripts/evaluate_rag.py
```

### 5. Launch Application Services

**FastAPI Backend Server:**
```bash
python app/api/main.py
# API docs available at http://localhost:8000/docs
```

**Streamlit Dashboard Console:**
```bash
streamlit run app/ui/streamlit_app.py --server.port=8502
# Web console available at http://localhost:8502
```

### 6. Run with Docker Compose

```bash
docker-compose up --build
```

---

## Testing

To run automated unit and integration tests:

```bash
PYTHONPATH=. venv/bin/python -m pytest -p no:launch_pytest tests/test_components.py -v
```

---

## License & Acknowledgments
- Bitext Customer Support Dataset
- Built with FastAPI, Streamlit, FAISS, Sentence Transformers, XGBoost, MLflow, and LangChain.
