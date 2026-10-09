import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_

from app.database import get_db
from app.models import User, HistoryEntry
from app.security import get_current_user_optional

router = APIRouter(prefix="/api/history", tags=["History"])

class HistoryItemOut(BaseModel):
    id: int
    entry_type: str
    title: str
    problem_or_input: str
    summary: Optional[str]
    solution: str
    metadata: Optional[dict]
    created_at: str

@router.get("", response_model=List[HistoryItemOut])
def get_history(
    entry_type: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    query = db.query(HistoryEntry)
    
    if current_user:
        if session_id:
            query = query.filter(or_(HistoryEntry.user_id == current_user.id, HistoryEntry.session_id == session_id))
        else:
            query = query.filter(HistoryEntry.user_id == current_user.id)
    else:
        if session_id:
            query = query.filter(or_(HistoryEntry.session_id == session_id, HistoryEntry.user_id == None))
        else:
            query = query.filter(HistoryEntry.user_id == None)

    if entry_type:
        query = query.filter(HistoryEntry.entry_type == entry_type)

    entries = query.order_by(desc(HistoryEntry.created_at)).limit(limit).all()

    results = []
    for e in entries:
        meta = None
        if e.metadata_json:
            try:
                meta = json.loads(e.metadata_json)
            except Exception:
                meta = {}
        results.append({
            "id": e.id,
            "entry_type": e.entry_type,
            "title": e.title,
            "problem_or_input": e.problem_or_input,
            "summary": e.summary,
            "solution": e.solution,
            "metadata": meta,
            "created_at": e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else ""
        })
    return results

@router.delete("/{entry_id}")
def delete_history_item(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    entry = db.query(HistoryEntry).filter(HistoryEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found.")
    
    if entry.user_id and current_user and entry.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    
    db.delete(entry)
    db.commit()
    return {"message": "History entry deleted successfully."}

@router.delete("")
def clear_history(
    session_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    query = db.query(HistoryEntry)
    if current_user:
        query = query.filter(HistoryEntry.user_id == current_user.id)
    elif session_id:
        query = query.filter(HistoryEntry.session_id == session_id)
    else:
        query = query.filter(HistoryEntry.user_id == None)
        
    deleted_count = query.delete(synchronize_session=False)
    db.commit()
    return {"message": f"Cleared {deleted_count} history records."}
