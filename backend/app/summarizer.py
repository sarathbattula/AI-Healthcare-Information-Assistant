from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from groq import Groq

from app.config import settings
from app.models import Document, DocumentChunk

def summarize_document_content(
    db: Session,
    doc_id: int,
    user_id: Optional[int],
    mode: str = "quick",
    target_question: Optional[str] = None,
    custom_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize document using map-reduce strategy for long files, preserving [p.N] references.
    Modes: quick, detailed, section_by_section, key_points, plain_language, question_focused, multi_document_compare.
    STRICT SAFETY RULE: Never diagnose from a medical report.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return {"error": f"Document ID {doc_id} not found."}
    if doc.doc_type == "private" and doc.owner_id != user_id:
        return {"error": "Access denied to private document."}

    chunks = db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).order_by(DocumentChunk.chunk_index).all()
    if not chunks:
        return {"error": "Document contains no readable text chunks."}

    effective_key = (custom_api_key.strip() if custom_api_key and custom_api_key.strip() else None) or settings.GROQ_API_KEY
    client = Groq(api_key=effective_key) if effective_key else None

    # Group chunks into manageable blocks (map phase)
    chunk_texts = [f"[p.{c.page_number}] {c.content}" for c in chunks]
    full_text = "\n\n".join(chunk_texts)

    system_prompt = (
        "You are an expert healthcare document summarizer. Summarize the provided medical document text.\n"
        "STRICT SAFETY RULES:\n"
        "1. Never diagnose, prescribe, or provide medical treatment instructions based on a medical report.\n"
        "2. Preserve explicit page numbers using [p.N] tags whenever citing facts from the text.\n"
        "3. Provide objective, clear, plain-language summaries.\n"
        "4. Include 3 recommended questions to ask a healthcare professional."
    )

    if mode == "quick":
        user_prompt = f"Provide a concise summary (under 200 words) of the document '{doc.title}'. Include key findings and 3 questions for a doctor:\n\n{full_text[:6000]}"
    elif mode == "detailed":
        user_prompt = f"Provide a comprehensive, detailed breakdown of the document '{doc.title}', highlighting methodology, findings, risk factors, and 3 questions for a doctor:\n\n{full_text[:8000]}"
    elif mode == "key_points":
        user_prompt = f"Extract bulleted key points and main takeaways from '{doc.title}' with page tags [p.N]:\n\n{full_text[:7000]}"
    elif mode == "plain_language":
        user_prompt = f"Translate and explain the medical text in '{doc.title}' into easy-to-understand plain language for patients (Grade 6 reading level):\n\n{full_text[:7000]}"
    elif mode == "question_focused" and target_question:
        user_prompt = f"Summarize '{doc.title}' specifically answering the question: '{target_question}'. Reference pages as [p.N]:\n\n{full_text[:7000]}"
    else:
        user_prompt = f"Provide a structured summary of '{doc.title}' with key findings, values, and 3 questions for a doctor:\n\n{full_text[:6000]}"

    if not client:
        # Fallback offline summary if Groq API key is not provided
        first_few_chunks = chunks[:3]
        summary_lines = [
            f"### Summary of {doc.title} (Offline Mode)",
            f"**Publisher**: {doc.publisher} | **Pages**: {len(chunks)}",
            "\n**Key Content Snippets**:"
        ]
        for c in first_few_chunks:
            summary_lines.append(f"- [p.{c.page_number}] {c.content[:200]}...")
        summary_lines.append("\n**3 Questions to Ask Your Doctor**:")
        summary_lines.append("1. What do these specific report findings mean for my overall health?")
        summary_lines.append("2. Are any follow-up blood tests or imaging recommended?")
        summary_lines.append("3. Should I adjust my daily routine or lifestyle based on this document?")
        
        return {
            "doc_id": doc_id,
            "title": doc.title,
            "mode": mode,
            "summary": "\n".join(summary_lines),
            "page_count": chunks[-1].page_number if chunks else 1,
            "offline_mode": True
        }

    candidate_models = [settings.GROQ_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]
    last_err = None
    for model_name in candidate_models:
        if not model_name:
            continue
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            summary_text = response.choices[0].message.content
            if summary_text:
                return {
                    "doc_id": doc_id,
                    "title": doc.title,
                    "mode": mode,
                    "summary": summary_text,
                    "page_count": chunks[-1].page_number if chunks else 1,
                    "offline_mode": False
                }
        except Exception as e:
            last_err = str(e)
            continue

    return {"error": f"Failed to generate LLM summary: {last_err or 'Groq API unreachable'}"}
