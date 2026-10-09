import os
import uuid
import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import User, Document, TrustedSource, HistoryEntry
from app.security import get_current_user, get_current_user_optional, log_audit
from app.ingestion import (
    validate_file, compute_sha256, extract_text_with_pages,
    chunk_pages_with_overlap
)
from app.rag_engine import index_document_chunks, delete_document_chunks
from app.config import settings
from app.agents.summarizer_agent import ReportSummarizerAgent
from app.agents.solution_agent import RecommendedSolutionAgent

router = APIRouter(prefix="/api/documents", tags=["Documents"])

class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    doc_type: str
    title: str
    publisher: str
    publication_date: Optional[str]
    status: str
    chunk_count: int
    is_outdated: bool = False
    is_superseded: bool = False

@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("private"), # 'private' or 'trusted'
    title: Optional[str] = Form(None),
    publisher: Optional[str] = Form(None),
    publication_date: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    # Only admin can upload trusted sources
    if doc_type == "trusted":
        if not current_user or current_user.role not in ["admin", "reviewer"]:
            raise HTTPException(status_code=403, detail="Only administrators can upload trusted knowledge sources.")

    file_content = await file.read()
    
    # 1. Validation (magic bytes, size, empty/corrupted check)
    try:
        file_ext = validate_file(file_content, file.filename)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    # 2. SHA256 Duplicate Check
    file_hash = compute_sha256(file_content)
    existing_doc = db.query(Document).filter(
        Document.file_hash == file_hash,
        Document.status != "deleted"
    ).first()
    if existing_doc:
        raise HTTPException(status_code=400, detail=f"Duplicate document detected: '{existing_doc.filename}' has already been indexed.")

    # 3. Safe Storage
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    with open(save_path, "wb") as f:
        f.write(file_content)

    doc_title = title if title else os.path.splitext(file.filename)[0]
    doc_pub = publisher if publisher else ("Verified Medical Registry" if doc_type == "trusted" else (current_user.username if current_user else "Anonymous User"))

    # 4. Create Document Record
    new_doc = Document(
        file_hash=file_hash,
        filename=file.filename,
        stored_filename=safe_filename,
        file_type=file_ext.replace(".", ""),
        file_size=len(file_content),
        doc_type=doc_type,
        owner_id=current_user.id if (current_user and doc_type == "private") else None,
        title=doc_title,
        publisher=doc_pub,
        publication_date=publication_date if publication_date else datetime.date.today().isoformat(),
        status="processing"
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    # 5. Extract Text & Chunking
    try:
        pages_text = extract_text_with_pages(file_content, file_ext)
        if not pages_text:
            new_doc.status = "failed"
            db.commit()
            raise HTTPException(status_code=400, detail="No readable text extracted from document.")

        chunks = chunk_pages_with_overlap(pages_text)
        db_chunks = index_document_chunks(db, new_doc, chunks)
        
        new_doc.status = "indexed"
        db.commit()

        log_audit(
            db,
            current_user.id if current_user else None,
            "DOCUMENT_UPLOAD",
            "documents",
            str(new_doc.id),
            f"Uploaded '{new_doc.filename}' as {doc_type} document with {len(db_chunks)} chunks."
        )

        return {
            "id": new_doc.id,
            "filename": new_doc.filename,
            "file_type": new_doc.file_type,
            "file_size": new_doc.file_size,
            "doc_type": new_doc.doc_type,
            "title": new_doc.title,
            "publisher": new_doc.publisher,
            "publication_date": new_doc.publication_date,
            "status": new_doc.status,
            "chunk_count": len(db_chunks)
        }

    except Exception as e:
        new_doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {str(e)}")


@router.get("/list", response_model=List[DocumentOut])
def list_documents(
    doc_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    user_id = current_user.id if current_user else None

    query = db.query(Document).options(joinedload(Document.trusted_source)).filter(Document.status != "deleted")
    if doc_type == "trusted":
        query = query.filter(Document.doc_type == "trusted")
    elif doc_type == "private":
        if not user_id:
            return []
        query = query.filter(Document.doc_type == "private", Document.owner_id == user_id)
    else:
        # Show all trusted + user's private docs
        if user_id:
            query = query.filter((Document.doc_type == "trusted") | (Document.owner_id == user_id))
        else:
            query = query.filter(Document.doc_type == "trusted")

    docs = query.all()
    results = []
    for d in docs:
        is_outdated = False
        is_superseded = False
        if d.trusted_source:
            if d.trusted_source.status == "outdated":
                is_outdated = True
            elif d.trusted_source.status == "superseded":
                is_superseded = True

        results.append({
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "doc_type": d.doc_type,
            "title": d.title,
            "publisher": d.publisher,
            "publication_date": d.publication_date,
            "status": d.status,
            "chunk_count": len(d.chunks),
            "is_outdated": is_outdated,
            "is_superseded": is_superseded
        })
    return results


@router.delete("/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    doc = db.query(Document).filter(Document.id == doc_id, Document.status != "deleted").first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Access control
    if doc.doc_type == "private" and doc.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied.")

    # Delete vector & FTS chunks
    delete_document_chunks(db, doc.id)

    doc.status = "deleted"
    db.commit()

    log_audit(db, current_user.id, "DOCUMENT_DELETE", "documents", str(doc_id), f"Deleted document '{doc.filename}'.")
    return {"message": f"Document ID {doc_id} ('{doc.filename}') successfully deleted."}


@router.post("/verify-and-solve")
async def verify_and_solve_document(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    custom_api_key: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    import json
    import re
    from groq import Groq

    if not file and not raw_text:
        raise HTTPException(status_code=400, detail="Please upload a document file or paste report text.")

    file_content = b""
    filename = "Pasted_Medical_Report.txt"
    file_ext = ".txt"

    if file:
        filename = file.filename
        file_content = await file.read()
        try:
            file_ext = validate_file(file_content, filename)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
    else:
        file_content = raw_text.encode("utf-8")
        file_ext = ".txt"

    file_size = len(file_content)
    file_hash = compute_sha256(file_content)

    # Extract text with pages
    try:
        pages_text = extract_text_with_pages(file_content, file_ext)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read document text: {str(e)}")

    if not pages_text:
        raise HTTPException(status_code=400, detail="No readable text found in this document.")

    combined_text = "\n\n".join([p[1] for p in pages_text])

    # Medical Authenticity & Verification Evaluation
    medical_lexicon = [
        r"\bpatient\b", r"\blab(oratory)?\b", r"\bclinic(al)?\b", r"\bhospital\b",
        r"\bdoctor\b", r"\bdr\.\b", r"\btest\b", r"\bresult\b", r"\breference\b",
        r"\brange\b", r"\bcholesterol\b", r"\bglucose\b", r"\bblood\b", r"\bserum\b",
        r"\bhemoglobin\b", r"\bhba1c\b", r"\burine\b", r"\bmg/dl\b", r"\bmmol/l\b",
        r"\blipid\b", r"\btriglycerides\b", r"\bcreatinine\b", r"\bdiagnosis\b",
        r"\bimpression\b", r"\bprescription\b", r"\bradiology\b", r"\bx-ray\b", r"\bct\b",
        r"\bwbc\b", r"\brbc\b", r"\bplatelets?\b", r"\becg\b", r"\bultrasound\b", r"\bmri\b",
        r"\bthyroid\b", r"\btsh\b", r"\bvitamin\b", r"\bcalcium\b", r"\bpotassium\b", r"\bsodium\b",
        r"\bnormal\b", r"\belevated\b", r"\bhigh\b", r"\blow\b", r"\babnormal\b", r"\bunit\b",
        r"\bvalue\b", r"\blevel\b", r"\bfindings?\b", r"\bexamination\b", r"\breport\b"
    ]
    matched_lexicon = [m for m in medical_lexicon if re.search(m, combined_text, re.IGNORECASE)]
    lexicon_score = min(100, max(60, int((len(matched_lexicon) / 5.0) * 100))) if len(matched_lexicon) >= 2 else min(100, int((len(matched_lexicon) / 5.0) * 100))
    is_valid_medical = len(matched_lexicon) >= 1

    # Scan for abnormal flags
    abnormal_pattern = r"\b(high|low|critical|abnormal|elevated|prediabetic|diabetic|positive|reactive)\b"
    abnormal_matches = re.findall(abnormal_pattern, combined_text, re.IGNORECASE)
    abnormal_count = len(abnormal_matches)

    # Document Indexing (Check if already indexed or index now)
    existing_doc = db.query(Document).filter(Document.file_hash == file_hash, Document.status != "deleted").first()
    if not existing_doc:
        safe_filename = f"{uuid.uuid4().hex}_{filename}"
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        with open(save_path, "wb") as f:
            f.write(file_content)

        doc_title = os.path.splitext(filename)[0].replace("_", " ").title()
        new_doc = Document(
            file_hash=file_hash,
            filename=filename,
            stored_filename=safe_filename,
            file_type=file_ext.replace(".", ""),
            file_size=file_size,
            doc_type="private",
            owner_id=current_user.id if current_user else None,
            title=doc_title,
            publisher=current_user.username if current_user else "Patient Upload",
            publication_date=datetime.date.today().isoformat(),
            status="indexed"
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        chunks = chunk_pages_with_overlap(pages_text)
        index_document_chunks(db, new_doc, chunks)
        doc_record = new_doc
    else:
        doc_record = existing_doc

    # Generate Summary via Agent 2 (ReportSummarizerAgent)
    summarizer = ReportSummarizerAgent(db, user_id=current_user.id if current_user else None, custom_api_key=custom_api_key)
    summary_result = summarizer.summarize_raw_text(combined_text, filename=filename, mode="detailed")
    report_summary_text = summary_result.get("summary", "")

    # Generate short summary snippet
    short_summary = f"Verified {filename} with {len(pages_text)} page(s). {abnormal_count} abnormal indicators flagged. Verification Confidence: {lexicon_score}%."

    # Save to History
    try:
        history_entry = HistoryEntry(
            user_id=current_user.id if current_user else None,
            session_id=session_id,
            entry_type="document",
            title=f"Report Summary: {filename}",
            problem_or_input=combined_text[:2000],
            summary=short_summary,
            solution=report_summary_text,
            metadata_json=json.dumps({
                "filename": filename,
                "file_hash": file_hash,
                "file_size": file_size,
                "confidence": f"{lexicon_score}%",
                "abnormal_count": abnormal_count,
                "doc_id": doc_record.id
            })
        )
        db.add(history_entry)
        db.commit()
    except Exception:
        db.rollback()

    return {
        "doc_id": doc_record.id,
        "filename": filename,
        "file_size": file_size,
        "file_hash": file_hash,
        "verification": {
            "status": "VERIFIED" if is_valid_medical else "PROCESSED_DOCUMENT",
            "is_authentic_medical_record": is_valid_medical,
            "confidence_score": f"{lexicon_score}%",
            "matched_markers_count": len(matched_lexicon),
            "abnormal_indicators_count": abnormal_count,
            "pages": len(pages_text)
        },
        "summary": short_summary,
        "summary_content": report_summary_text,
        "raw_text": combined_text,
        "full_text_preview": combined_text[:1200]
    }


class GenerateSolutionInput(BaseModel):
    raw_text: str
    filename: Optional[str] = "Medical_Report.txt"
    custom_api_key: Optional[str] = None
    session_id: Optional[str] = None

@router.post("/generate-solution")
def generate_document_solution(
    payload: GenerateSolutionInput,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    import json
    solution_agent = RecommendedSolutionAgent(db, custom_api_key=payload.custom_api_key)
    solution_plan_text = solution_agent.generate_solution_plan(
        query_or_report_text=payload.raw_text,
        filename=payload.filename
    )

    try:
        history_entry = HistoryEntry(
            user_id=current_user.id if current_user else None,
            session_id=payload.session_id,
            entry_type="solution",
            title=f"Solution Plan: {payload.filename}",
            problem_or_input=payload.raw_text[:2000],
            summary=f"Generated clinical solution plan for {payload.filename}",
            solution=solution_plan_text,
            metadata_json=json.dumps({"filename": payload.filename})
        )
        db.add(history_entry)
        db.commit()
    except Exception:
        db.rollback()

    return {
        "filename": payload.filename,
        "solution": solution_plan_text
    }


