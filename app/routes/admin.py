from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, TrustedSource, EscalationTicket, AuditLog, Document
from app.security import require_admin, log_audit

router = APIRouter(prefix="/api/admin", tags=["Admin & Governance"])

class TrustedSourceInput(BaseModel):
    title: str
    publisher: str # WHO, CDC, MedlinePlus
    url: Optional[str] = None
    publication_date: Optional[str] = None
    version: str = "1.0"
    status: str = "approved" # approved, superseded, outdated

class TrustedSourceOut(BaseModel):
    id: int
    title: str
    publisher: str
    url: Optional[str]
    publication_date: Optional[str]
    version: str
    status: str

class EscalationTicketOut(BaseModel):
    id: int
    user_id: int
    username: str
    category: str
    query: str
    redacted_query: Optional[str]
    status: str
    created_at: str

class AuditLogOut(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Optional[str]
    timestamp: str

@router.get("/sources", response_model=List[TrustedSourceOut])
def list_trusted_sources(db: Session = Depends(get_db)):
    sources = db.query(TrustedSource).all()
    return sources

@router.post("/sources", response_model=TrustedSourceOut)
def register_trusted_source(
    payload: TrustedSourceInput,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    source = TrustedSource(
        title=payload.title,
        publisher=payload.publisher,
        url=payload.url,
        publication_date=payload.publication_date,
        version=payload.version,
        status=payload.status
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    log_audit(db, admin.id, "REGISTER_TRUSTED_SOURCE", "trusted_sources", str(source.id), f"Added '{source.title}'.")
    return source

@router.put("/sources/{source_id}/status")
def update_source_status(
    source_id: int,
    status: str,
    superseded_by_id: Optional[int] = None,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    source = db.query(TrustedSource).filter(TrustedSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")
    source.status = status
    if superseded_by_id:
        source.superseded_by_id = superseded_by_id
    db.commit()

    log_audit(db, admin.id, "UPDATE_SOURCE_STATUS", "trusted_sources", str(source_id), f"Status changed to {status}.")
    return {"message": f"Source {source_id} status updated to {status}."}

@router.get("/tickets", response_model=List[EscalationTicketOut])
def list_escalation_tickets(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    tickets = db.query(EscalationTicket).order_by(EscalationTicket.created_at.desc()).all()
    out = []
    for t in tickets:
        out.append({
            "id": t.id,
            "user_id": t.user_id,
            "username": t.user.username if t.user else "Anonymous",
            "category": t.category,
            "query": t.query,
            "redacted_query": t.redacted_query,
            "status": t.status,
            "created_at": t.created_at.isoformat()
        })
    return out

@router.put("/tickets/{ticket_id}/status")
def update_ticket_status(
    ticket_id: int,
    status: str, # REVIEWED, CLOSED
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    ticket = db.query(EscalationTicket).filter(EscalationTicket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    ticket.status = status
    db.commit()

    log_audit(db, admin.id, "UPDATE_TICKET_STATUS", "escalation_tickets", str(ticket_id), f"Updated to {status}")
    return {"message": f"Ticket {ticket_id} status updated to {status}."}

@router.get("/audit-logs", response_model=List[AuditLogOut])
def list_audit_logs(limit: int = 50, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return [{
        "id": l.id,
        "user_id": l.user_id,
        "action": l.action,
        "resource_type": l.resource_type,
        "resource_id": l.resource_id,
        "details": l.details,
        "timestamp": l.timestamp.isoformat()
    } for l in logs]
