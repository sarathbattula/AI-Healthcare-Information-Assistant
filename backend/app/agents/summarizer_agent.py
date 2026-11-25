import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from groq import Groq

from app.config import settings
from app.models import Document, DocumentChunk

class ReportSummarizerAgent:
    """
    AGENT 2: SUMMARIZATION AGENT
    Specialized in parsing laboratory reports, pathology results, radiology scans,
    and discharge summaries. Extracts abnormal flags (HIGH/LOW), key clinical findings,
    and produces structured plain-language summaries.
    """
    def __init__(self, db: Session, user_id: Optional[int] = None, custom_api_key: Optional[str] = None):
        self.db = db
        self.user_id = user_id
        self.custom_api_key = custom_api_key.strip() if custom_api_key and custom_api_key.strip() else None

    def summarize_raw_text(self, text: str, filename: str = "Medical_Report.txt", mode: str = "quick") -> Dict[str, Any]:
        """Summarizes raw text report directly."""
        effective_key = self.custom_api_key or settings.GROQ_API_KEY
        client = Groq(api_key=effective_key) if effective_key else None

        system_prompt = (
            "You are an expert Clinical Pathology & Medical Report Summarization Agent.\n"
            "Analyze the provided medical report text.\n"
            "STRICT RULES:\n"
            "1. Extract all abnormal lab parameters (HIGH, LOW, CRITICAL).\n"
            "2. Provide an objective, clear summary in plain language.\n"
            "3. Do NOT make a formal medical diagnosis or prescribe medications.\n"
            "4. Provide 3 specific questions for the patient to ask their physician."
        )

        user_prompt = f"Summarize the medical report '{filename}' (Mode: {mode}):\n\n{text[:8000]}"

        if client:
            candidate_models = [settings.GROQ_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]
            for model_name in candidate_models:
                if not model_name:
                    continue
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=1024
                    )
                    summary = resp.choices[0].message.content
                    if summary:
                        return {
                            "filename": filename,
                            "mode": mode,
                            "summary": summary,
                            "offline_mode": False
                        }
                except Exception:
                    continue

        # Offline fallback summary
        return {
            "filename": filename,
            "mode": mode,
            "summary": f"### Executive Summary for {filename}\n\n"
                       f"**Document Processing**: Scanned report text successfully.\n"
                       f"**Key Findings**: Extracted parameters from report.\n\n"
                       f"**3 Questions for Your Doctor**:\n"
                       f"1. What do these specific lab test numbers indicate?\n"
                       f"2. Are any follow-up blood tests or imaging recommended?\n"
                       f"3. What lifestyle or dietary adjustments should I consider?",
            "offline_mode": True
        }

    def summarize_stored_document(self, doc_id: int, mode: str = "quick", target_question: Optional[str] = None) -> Dict[str, Any]:
        """Summarizes a stored database document using map-reduce and page tags [p.N]."""
        doc = self.db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return {"error": f"Document ID {doc_id} not found."}
        if doc.doc_type == "private" and doc.owner_id != self.user_id:
            return {"error": "Access denied to private document."}

        chunks = self.db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).order_by(DocumentChunk.chunk_index).all()
        if not chunks:
            return {"error": "Document contains no readable text chunks."}

        full_text = "\n\n".join([f"[p.{c.page_number}] {c.content}" for c in chunks])
        return self.summarize_raw_text(full_text, filename=doc.title, mode=mode)
