import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, HistoryEntry
from app.security import get_current_user_optional, log_audit
from app.agents.chatbot_agent import AIChatbotAgent

router = APIRouter(prefix="/api/chat", tags=["RAG Chat"])

class ChatQueryInput(BaseModel):
    query: str
    api_key: Optional[str] = None
    report_context: Optional[str] = None
    session_id: Optional[str] = None

class CitationOutput(BaseModel):
    citation_id: str
    chunk_id: int
    doc_id: int
    title: str
    publisher: str
    page_number: int
    publication_date: str
    content: str
    similarity_score: float
    is_outdated: bool = False
    is_superseded: bool = False

class ChatQueryResponse(BaseModel):
    answer: str
    citations: List[CitationOutput]
    safety_category: str
    escalated: bool
    ticket_id: Optional[int] = None
    execution_time_sec: float

@router.post("/query", response_model=ChatQueryResponse)
def query_rag_chat(
    payload: ChatQueryInput,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    user_obj = current_user if isinstance(current_user, User) else None
    agent = AIChatbotAgent(db, current_user=user_obj, custom_api_key=payload.api_key)
    result = agent.run_chat_query(payload.query, report_context=payload.report_context)

    # Save to history entry
    try:
        title_summary = payload.query.strip().split("\n")[0][:60]
        if len(payload.query.strip()) > 60:
            title_summary += "..."
        entry = HistoryEntry(
            user_id=user_obj.id if user_obj else None,
            session_id=payload.session_id,
            entry_type="chat",
            title=title_summary,
            problem_or_input=payload.query,
            solution=result.get("answer", ""),
            metadata_json=json.dumps({
                "safety_category": result.get("safety_category"),
                "citations_count": len(result.get("citations", [])),
                "execution_time_sec": result.get("execution_time_sec")
            })
        )
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()

    log_audit(
        db,
        user_obj.id if user_obj else None,
        "CHAT_QUERY",
        "chat",
        None,
        f"Category: {result.get('safety_category')}, Escalated: {result.get('escalated')}"
    )

    return result

