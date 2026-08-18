import streamlit as st
import json
import requests
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from app.rag.pipeline import SupportIntelligencePipeline

st.set_page_config(
    page_title="Customer Support Intelligence Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI with white headers
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #F8FAFC !important;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #78350F;
    }
    .evidence-box {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache pipeline initialization
@st.cache_resource
def get_pipeline():
    return SupportIntelligencePipeline()

pipeline = get_pipeline()

# Title
st.markdown('<div class="main-header">Customer Support Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade RAG, Ticket Classification, Proxy Escalation Prediction, and LLMOps Observability</div>', unsafe_allow_html=True)

# Navigation Tabs (2 Tabs)
tab1, tab2 = st.tabs(["Live Support Console", "RAG Evaluation & Benchmark"])

# ==========================================
# TAB 1: LIVE SUPPORT INTELLIGENCE CONSOLE
# ==========================================
with tab1:
    col_input, col_preset = st.columns([3, 1])
    
    preset_map = {
        "Payment charged twice (#9821)": "My credit card was charged twice for order #9821. I have messaged support twice.",
        "Order status pending after 3 days": "My payment went through but the order still says pending after 3 days. Please help!",
        "Want to cancel monthly subscription": "How do I cancel my monthly subscription plan and will I get a refund?",
        "Damaged item return request": "I received a damaged item yesterday and I want to return it for a full refund.",
        "Out-of-scope: Bake chocolate cake": "How do I bake a delicious chocolate cake with vanilla icing?"
    }

    with col_preset:
        sample_choice = st.selectbox(
            "Load Preset Query:",
            [
                "Custom Input",
                "Payment charged twice (#9821)",
                "Order status pending after 3 days",
                "Want to cancel monthly subscription",
                "Damaged item return request",
                "Out-of-scope: Bake chocolate cake"
            ]
        )

    default_text = preset_map.get(sample_choice, "My payment went through but the order still says pending.")

    with col_input:
        query_text = st.text_area(
            "Enter Customer Support Ticket / User Query:",
            value=default_text,
            height=100
        )
    
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        contact_freq = st.slider("Prior Contact Count (User Interaction Count):", min_value=1, max_value=10, value=2)
    with col_opt2:
        use_cache = st.checkbox("Enable Redis Cache Lookup", value=True)

    if st.button("Analyze Ticket & Generate Response", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a valid support ticket message.")
        else:
            with st.spinner("Processing ticket through Support Intelligence Pipeline..."):
                t0 = time.time()
                result = pipeline.run(
                    query=query_text,
                    user_metadata={"interaction_count": contact_freq}
                )
                latency = round((time.time() - t0) * 1000, 2)
                
            st.markdown("---")
            
            # Row 1: Key Metrics & Risk Indicators
            m1, m2, m3, m4 = st.columns(4)
            
            cat_info = result.get("predicted_category", {})
            pred_cat = cat_info.get("category", "N/A").upper()
            conf = cat_info.get("confidence", 0.0) * 100
            
            esc_info = result.get("escalation_prediction", {})
            esc_score = esc_info.get("escalation_score", 0.0)
            esc_level = esc_info.get("escalation_level", "LOW")
            
            m1.metric("Predicted Category", pred_cat, f"{conf:.1f}% confidence")
            m2.metric("Escalation Risk Score", f"{esc_score:.3f}", esc_level)
            m3.metric("Pipeline Latency", f"{result.get('latency_ms', latency)} ms")
            m4.metric("Redis Cache Hit", "Yes" if result.get("cached") else "No")
            
            # Escalation Disclaimer Callout
            if "disclaimer" in esc_info:
                st.markdown(f'<div class="disclaimer-box"><b>Experimental Proxy Model:</b> {esc_info["disclaimer"]}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Classification Probabilities Chart
            st.subheader("Issue Classification Probabilities")
            probs = cat_info.get("probabilities", {})
            if probs:
                st.bar_chart(probs)

            st.markdown("---")

            # Grounded LLM Response (Full Width)
            st.subheader("Grounded LLM Response")
            resp_data = result.get("response", {})
            
            if resp_data.get("low_evidence"):
                st.error("Low Evidence Alert: The knowledge base does not contain sufficient verified evidence for this query. Responding with safe fallback notice.")
                
            generated_ans = resp_data.get("generated_text", resp_data.get("answer", "No response generated."))
            st.markdown(generated_ans)
            
            evidence_list = resp_data.get("evidence_sources", resp_data.get("evidence_used", []))
            if evidence_list:
                st.markdown("**Citations & Grounded Evidence Sources:**")
                for src in evidence_list:
                    st.markdown(f"- `{src}`")

            st.markdown("---")

            # Row 3: 2-Column Layout for RAG Evidence & Historical Ticket Interactions
            col_rag_kb, col_rag_hist = st.columns([1, 1])
            
            with col_rag_kb:
                st.subheader("Hybrid RAG Retrieved Knowledge Base Chunks")
                kb_chunks = result.get("retrieved_kb_evidence", [])
                if kb_chunks:
                    for i, chunk in enumerate(kb_chunks, 1):
                        meta = chunk.get("metadata", {})
                        with st.expander(f"Chunk #{i}: {meta.get('document_title', 'KB Doc')} (Rerank Score: {chunk.get('rerank_score', chunk.get('score', 0)):.4f})"):
                            st.write(chunk.get("content", ""))
                            st.caption(f"Source: `{meta.get('source', '')}` | Section: `{meta.get('section', '')}`")
                else:
                    st.write("No knowledge base chunks retrieved.")
                    
            with col_rag_hist:
                st.subheader("Similar Historical Support Ticket Interactions")
                hist_tickets = result.get("retrieved_historical_tickets", [])
                if hist_tickets:
                    for j, ticket in enumerate(hist_tickets, 1):
                        with st.expander(f"Historical Case #{j} (Sim Score: {ticket.get('similarity_score', 0):.3f})"):
                            st.write(f"**Category:** {ticket.get('category', '').upper()}")
                            st.write(f"**Customer Issue:** {ticket.get('instruction', '')}")
                            st.write(f"**Resolution:** {ticket.get('response', '')}")
                else:
                    st.write("No historical support tickets matched.")

# ==========================================
# TAB 2: RAG EVALUATION & BENCHMARK METRICS
# ==========================================
with tab2:
    st.subheader("End-to-End RAG & Pipeline Evaluation Metrics")
    
    eval_file = settings.EVAL_DIR / "eval_report.json"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_report = json.load(f)
            
        summary = eval_report.get("summary", {})
        
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Total Test Queries", summary.get("total_test_queries", 0))
        e2.metric("Classification Accuracy", f"{summary.get('category_classification_accuracy', 0)*100:.1f}%")
        e3.metric("Retrieval Hit Rate @ K", f"{summary.get('retrieval_hit_rate_at_k', 0)*100:.1f}%")
        e4.metric("Retrieval MRR", f"{summary.get('retrieval_mrr', 0):.4f}")
        e5.metric("Low-Evidence Accuracy", f"{summary.get('low_evidence_detection_accuracy', 0)*100:.1f}%")
        
        st.markdown("---")
        st.subheader("Detailed Benchmark Evaluation Logs")
        detailed = eval_report.get("detailed_results", [])
        if detailed:
            st.dataframe(detailed, use_container_width=True)
    else:
        st.info("No evaluation report found. Run `python scripts/evaluate_rag.py` to generate evaluation benchmark results.")
