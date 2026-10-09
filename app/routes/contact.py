import uuid
import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db
from app.agents.contact_email_agent import ContactEmailAgent

router = APIRouter(prefix="/api/contact", tags=["Contact & Support"])

class ContactMessageInput(BaseModel):
    name: str
    email: str
    category: str = "general" # emergency, clinical_inquiry, report_review, technical
    subject: str
    message: str

class ContactSubmissionResponse(BaseModel):
    ticket_id: str
    status: str
    message: str
    emergency_notice: str
    received_at: str
    email_dispatched_to: Optional[str] = None
    email_status: Optional[str] = None

@router.get("/helplines")
def get_helplines():
    return {
        "emergency": [
            {"service": "National Emergency Service", "number": "112", "description": "All-in-one emergency response (Police, Fire, Ambulance)", "badge": "Critical"},
            {"service": "Medical Emergency & Ambulance", "number": "108", "description": "Free 24/7 emergency medical transit & trauma care", "badge": "Ambulance"},
            {"service": "Tele-MANAS Mental Health Helpline", "number": "14416", "description": "Toll-free 24/7 mental health counseling & crisis support", "badge": "24/7 Toll-Free"},
            {"service": "National Health Helpline", "number": "104", "description": "Medical advice and healthcare consultation directory", "badge": "Advisory"},
            {"service": "Women & Child Helpline", "number": "1091", "description": "Dedicated 24/7 support & emergency protection", "badge": "Support"}
        ],
        "support_hours": "24 Hours / 7 Days a week for emergencies. Clinical email review response within 2-4 hours.",
        "hospital_network": "Affiliated with Apollo, Fortis, AIIMS, and Manipal Health networks."
    }

from app.models import ContactInquiry

@router.get("/inquiries")
def get_contact_inquiries(db: Session = Depends(get_db)):
    inquiries = db.query(ContactInquiry).order_by(ContactInquiry.created_at.desc()).all()
    return [
        {
            "ticket_id": inq.ticket_id,
            "name": inq.name,
            "email": inq.email,
            "category": inq.category,
            "subject": inq.subject,
            "message": inq.message,
            "recipient_email": inq.recipient_email,
            "email_status": inq.email_status,
            "created_at": inq.created_at.strftime("%Y-%m-%d %H:%M:%S") if inq.created_at else None
        }
        for inq in inquiries
    ]

@router.post("/submit", response_model=ContactSubmissionResponse)
def submit_contact_inquiry(payload: ContactMessageInput, db: Session = Depends(get_db)):
    agent = ContactEmailAgent(db, target_email=settings.SUPPORT_EMAIL)
    result = agent.process_and_send_email(
        name=payload.name,
        user_email=payload.email,
        category=payload.category,
        subject=payload.subject,
        message=payload.message
    )
    return result

