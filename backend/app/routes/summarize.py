from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.security import get_current_user_optional, log_audit
from app.agents.summarizer_agent import ReportSummarizerAgent

router = APIRouter(prefix="/api/summarize", tags=["Summarizer"])

class SummarizeInput(BaseModel):
    doc_id: int
    mode: str = "quick" # quick, detailed, key_points, plain_language, question_focused, multi_document_compare
    target_question: Optional[str] = None

class SummarizeResponse(BaseModel):
    doc_id: int
    title: str
    mode: str
    summary: str
    page_count: int
    offline_mode: bool = False

@router.post("/document", response_model=SummarizeResponse)
def summarize_doc(
    payload: SummarizeInput,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    user_id = current_user.id if current_user else None
    agent = ReportSummarizerAgent(db, user_id=user_id)
    res = agent.summarize_stored_document(
        doc_id=payload.doc_id,
        mode=payload.mode,
        target_question=payload.target_question
    )

    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])

    log_audit(db, user_id, "SUMMARIZE_DOCUMENT", "documents", str(payload.doc_id), f"Mode: {payload.mode}")
    return res
