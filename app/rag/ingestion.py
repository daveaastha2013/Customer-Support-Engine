import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentIngestion:
    """Ingests Knowledge Base docs, cleans text, and splits into chunks with rich metadata."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )

    def load_kb_documents(self, kb_dir: Path = None) -> List[Dict[str, Any]]:
        target_dir = kb_dir or settings.KB_DIR
        if not target_dir.exists():
            raise FileNotFoundError(f"Knowledge Base directory does not exist: {target_dir}")
            
        chunks = []
        chunk_counter = 0
        
        for file_path in sorted(target_dir.glob("*.md")):
            category = file_path.stem.replace("_policy", "").replace("_faq", "").replace("_troubleshooting", "").replace("_management", "").replace("_security", "")
            title = file_path.stem.replace("_", " ").title()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split document into chunks
            raw_chunks = self.text_splitter.split_text(content)
            
            for idx, chunk_text in enumerate(raw_chunks):
                chunk_counter += 1
                # Extract first heading in chunk as section header if present
                lines = chunk_text.strip().split("\n")
                section = "General"
                for line in lines:
                    if line.startswith("#"):
                        section = line.strip("#").strip()
                        break
                        
                chunks.append({
                    "chunk_id": f"KB-{chunk_counter:04d}",
                    "content": chunk_text,
                    "metadata": {
                        "document_title": title,
                        "category": category,
                        "source": file_path.name,
                        "section": section,
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks)
                    }
                })
                
        logger.info(f"Ingested {len(chunks)} chunks from {len(list(target_dir.glob('*.md')))} Knowledge Base documents.")
        return chunks
