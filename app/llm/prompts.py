import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert, empathetic, and grounded Customer Support Intelligence Assistant.

Your objective is to help the customer support team by providing a grounded response, cited evidence, and recommended support resolution.

CRITICAL CONSTRAINTS:
1. Answer ONLY using the provided Knowledge Base context and Historical Support Cases.
2. Do NOT invent policies, details, or claims that are not explicitly present in the provided evidence.
3. If the provided context is insufficient to answer the query, clearly state: "I don't have enough evidence to answer this query based on our knowledge base and past tickets."
4. Cite all sources using brackets, e.g., [payment_policy.md] or [HIST-TICK-10025].
5. Maintain a professional, clear customer-facing tone.

FORMAT YOUR RESPONSE IN CLEAR SECTIONS:
### Suggested Response
[Your customer-facing message]

### Evidence Used
[List of cited sources with short description]

### Recommended Support Resolution
[Internal support action recommended]
"""

def format_context_for_prompt(kb_evidence: List[Dict[str, Any]], historical_evidence: List[Dict[str, Any]]) -> str:
    """Formats retrieved KB chunks and historical tickets into a clean context block."""
    context_parts = []
    
    if kb_evidence:
        context_parts.append("=== KNOWLEDGE BASE DOCUMENTS ===")
        for item in kb_evidence:
            source = item["metadata"].get("source", "kb_doc")
            title = item["metadata"].get("document_title", "Document")
            section = item["metadata"].get("section", "Section")
            context_parts.append(f"Source: [{source}] (Title: {title}, Section: {section})\nContent: {item['content'].strip()}\n")
            
    if historical_evidence:
        context_parts.append("=== HISTORICAL SUPPORT TICKETS ===")
        for item in historical_evidence:
            ticket_id = item.get("ticket_id", item.get("chunk_id", "HIST"))
            query = item.get("query", "")
            resolution = item.get("resolution", item.get("content", ""))
            context_parts.append(f"Ticket ID: [{ticket_id}]\nCustomer Issue: {query}\nPast Resolution: {resolution}\n")
            
    return "\n".join(context_parts)
