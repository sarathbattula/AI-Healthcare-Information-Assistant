import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user") # 'user', 'admin', 'reviewer'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="owner")
    audit_logs = relationship("AuditLog", back_populates="user")
    tickets = relationship("EscalationTicket", back_populates="user")


class TrustedSource(Base):
    __tablename__ = "trusted_sources"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    publisher = Column(String(100), nullable=False) # 'WHO', 'CDC', 'MedlinePlus'
    url = Column(String(500), nullable=True)
    publication_date = Column(String(50), nullable=True)
    version = Column(String(50), default="1.0")
    status = Column(String(30), default="approved") # 'approved', 'superseded', 'outdated'
    superseded_by_id = Column(Integer, ForeignKey("trusted_sources.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="trusted_source")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_hash = Column(String(64), index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), unique=True, nullable=False)
    file_type = Column(String(20), nullable=False) # 'pdf', 'docx', 'txt'
    file_size = Column(Integer, nullable=False)
    doc_type = Column(String(20), nullable=False, default="private") # 'trusted', 'private'
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    trusted_source_id = Column(Integer, ForeignKey("trusted_sources.id"), nullable=True)
    
    title = Column(String(255), nullable=False)
    publisher = Column(String(100), nullable=False, default="User Upload")
    url = Column(String(500), nullable=True)
    publication_date = Column(String(50), nullable=True)
    status = Column(String(30), default="indexed") # 'indexed', 'processing', 'failed', 'deleted'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="documents")
    trusted_source = relationship("TrustedSource", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False, default=1)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    embedding_id = Column(String(100), nullable=False)

    document = relationship("Document", back_populates="chunks")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")


class EscalationTicket(Base):
    __tablename__ = "escalation_tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category = Column(String(50), nullable=False) # 'A_EMERGENCY', 'B_PROFESSIONAL_REVIEW'
    query = Column(Text, nullable=False)
    redacted_query = Column(Text, nullable=True)
    status = Column(String(30), default="OPEN") # 'OPEN', 'REVIEWED', 'CLOSED'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="tickets")


class HistoryEntry(Base):
    __tablename__ = "history_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(100), nullable=True)
    entry_type = Column(String(50), nullable=False) # 'chat' or 'document'
    title = Column(String(255), nullable=False)
    problem_or_input = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    solution = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True) # JSON with citations, verification status, metrics, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User")


class ContactInquiry(Base):
    __tablename__ = "contact_inquiries"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    category = Column(String(50), default="general")
    subject = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    recipient_email = Column(String(100), default="battulasarath751@gmail.com")
    email_status = Column(String(50), default="SENT")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


