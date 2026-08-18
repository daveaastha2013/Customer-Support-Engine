import os
import logging
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from app.llm.prompts import SYSTEM_PROMPT, format_context_for_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMResponseGenerator:
    """RAG response generator using Groq LLM API with graceful fallback behavior."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_key = api_key or settings.GROQ_API_KEY
        self.llm = None
        
        if self.api_key:
            try:
                self.llm = ChatGroq(
                    model_name=self.model_name,
                    groq_api_key=self.api_key,
                    temperature=0.1
                )
                logger.info(f"Initialized Groq ChatGroq LLM ({self.model_name}).")
            except Exception as e:
                logger.warning(f"Could not initialize Groq LLM ({e}). Will use fallback generator.")
        else:
            logger.info("GROQ_API_KEY not found in environment. Defaulting to fallback response generator.")

    def generate_response(
        self,
        query: str,
        kb_evidence: List[Dict[str, Any]],
        historical_evidence: List[Dict[str, Any]],
        escalation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates grounded response using LLM or structured fallback.
        Checks for low evidence detection.
        """
        # Low Evidence Detection Check
        all_evidence = kb_evidence + historical_evidence
        max_score = max([e.get("reranker_score", e.get("score", 0.0)) for e in all_evidence], default=0.0)
        
        if not all_evidence or (len(kb_evidence) == 0 and len(historical_evidence) == 0):
            logger.info("Insufficient evidence detected (empty retrieved context).")
            return self._build_low_evidence_response(query, escalation_info)
            
        context_str = format_context_for_prompt(kb_evidence, historical_evidence)
        user_prompt = f"CUSTOMER TICKET QUERY: {query}\n\nRETRIEVED EVIDENCE:\n{context_str}\n\nPlease analyze the query and evidence to generate the response."
        
        # Try Groq LLM if configured
        if self.llm:
            try:
                logger.info("Calling Groq LLM API...")
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt)
                ]
                response_msg = self.llm.invoke(messages)
                raw_text = response_msg.content
                
                return {
                    "generated_text": raw_text,
                    "model_used": f"Groq/{self.model_name}",
                    "evidence_sources": [e.get("metadata", {}).get("source", e.get("chunk_id")) for e in all_evidence],
                    "escalation_summary": escalation_info,
                    "is_fallback": False
                }
            except Exception as e:
                logger.warning(f"Groq LLM API invocation failed ({e}). Falling back to rule-based generator.")
                
        # Structured Fallback Response Generator
        return self._generate_fallback_response(query, kb_evidence, historical_evidence, escalation_info)

    def _build_low_evidence_response(self, query: str, escalation_info: Dict[str, Any]) -> Dict[str, Any]:
        text = (
            "### Suggested Response\n"
            "I don't have enough evidence to answer this query based on our knowledge base and past tickets. "
            "Your ticket has been routed to our specialized support team for manual review.\n\n"
            "### Evidence Used\n"
            "None (No relevant knowledge base articles or historical tickets matched the confidence threshold).\n\n"
            "### Recommended Support Resolution\n"
            "Manual agent review required. Route to tier-2 support."
        )
        return {
            "generated_text": text,
            "model_used": "Low-Evidence-Fallback",
            "evidence_sources": [],
            "escalation_summary": escalation_info,
            "is_fallback": True,
            "low_evidence": True
        }

    def _generate_fallback_response(
        self,
        query: str,
        kb_evidence: List[Dict[str, Any]],
        historical_evidence: List[Dict[str, Any]],
        escalation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates clear, deterministic grounded support response based directly on top evidence."""
        top_kb = kb_evidence[0] if kb_evidence else None
        top_hist = historical_evidence[0] if historical_evidence else None
        
        sources = []
        snippets = []
        
        if top_kb:
            src = top_kb["metadata"].get("source", "KB Document")
            sources.append(src)
            snippets.append(f"According to [{src}]: {top_kb['content'].strip()}")
            
        if top_hist:
            ticket_id = top_hist.get("ticket_id", "Past Case")
            sources.append(ticket_id)
            snippets.append(f"In past case [{ticket_id}], resolution: {top_hist.get('resolution', top_hist.get('content', ''))}")
            
        ans_text = " ".join(snippets) if snippets else "Our support guidelines provide standard operating procedures for this query."
        
        text = (
            f"### Suggested Response\n"
            f"Thank you for contacting support. {ans_text}\n\n"
            f"### Evidence Used\n"
            f"- " + "\n- ".join([f"[{s}]" for s in sources]) + "\n\n"
            f"### Recommended Support Resolution\n"
            f"Apply resolution steps detailed in [{sources[0] if sources else 'Support Policy'}]. "
            f"Escalation Level: {escalation_info.get('escalation_level', 'LOW')} ({escalation_info.get('recommendation', '')})"
        )
        
        return {
            "generated_text": text,
            "model_used": "Deterministic-Grounded-Engine",
            "evidence_sources": sources,
            "escalation_summary": escalation_info,
            "is_fallback": True,
            "low_evidence": False
        }
