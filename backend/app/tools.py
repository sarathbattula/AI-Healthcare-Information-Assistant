from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.rag_engine import hybrid_search
from app.models import Document, TrustedSource
from app.safety_engine import classify_safety_category

# Pydantic Schemas for 8 Tool Arguments
class SearchHealthDocumentsInput(BaseModel):
    query: str = Field(..., description="Healthcare query or keywords to search for.")
    top_k: int = Field(5, ge=1, le=10, description="Number of passages to retrieve.")

class FilterDocumentsInput(BaseModel):
    publisher: Optional[str] = Field(None, description="Publisher filter e.g. WHO, CDC, MedlinePlus.")
    doc_type: Optional[str] = Field(None, description="Document type filter e.g. trusted or private.")

class GetDocumentMetadataInput(BaseModel):
    doc_id: int = Field(..., description="Database ID of the document.")

class SummarizeDocumentInput(BaseModel):
    doc_id: int = Field(..., description="Database ID of the document.")
    mode: str = Field("quick", description="Summary mode: quick, detailed, key_points, plain_language, section_by_section.")

class ExplainMedicalTermInput(BaseModel):
    term: str = Field(..., description="Medical term to explain in plain language.")

class CompareSourcesInput(BaseModel):
    topic: str = Field(..., description="Topic to compare between sources.")

class AssessEscalationInput(BaseModel):
    query: str = Field(..., description="User query to evaluate for emergency or professional review escalation.")

class FormatCitationsInput(BaseModel):
    citations: List[Dict[str, Any]] = Field(..., description="List of citation metadata items to validate and format.")

# Tool Implementation Functions
def tool_search_health_documents(db: Session, user_id: Optional[int], args: SearchHealthDocumentsInput) -> List[Dict[str, Any]]:
    return hybrid_search(db, query=args.query, user_id=user_id, top_k=args.top_k)

def tool_filter_documents(db: Session, user_id: Optional[int], args: FilterDocumentsInput) -> List[Dict[str, Any]]:
    query = db.query(Document).filter(Document.status == "indexed")
    if args.doc_type == "trusted":
        query = query.filter(Document.doc_type == "trusted")
    elif args.doc_type == "private":
        query = query.filter(Document.doc_type == "private", Document.owner_id == user_id)
    else:
        query = query.filter((Document.doc_type == "trusted") | (Document.owner_id == user_id))

    if args.publisher:
        query = query.filter(Document.publisher.ilike(f"%{args.publisher}%"))

    docs = query.all()
    return [{
        "id": d.id,
        "title": d.title,
        "publisher": d.publisher,
        "doc_type": d.doc_type,
        "publication_date": d.publication_date,
        "url": d.url
    } for d in docs]

def tool_get_document_metadata(db: Session, user_id: Optional[int], args: GetDocumentMetadataInput) -> Dict[str, Any]:
    doc = db.query(Document).filter(Document.id == args.doc_id).first()
    if not doc:
        return {"error": f"Document ID {args.doc_id} not found."}
    # Check access permission
    if doc.doc_type == "private" and doc.owner_id != user_id:
        return {"error": "Access denied to private document."}

    return {
        "id": doc.id,
        "title": doc.title,
        "publisher": doc.publisher,
        "doc_type": doc.doc_type,
        "publication_date": doc.publication_date,
        "file_type": doc.file_type,
        "file_size": doc.file_size,
        "status": doc.status,
        "chunk_count": len(doc.chunks)
    }

def tool_explain_medical_term(args: ExplainMedicalTermInput) -> Dict[str, Any]:
    # Common medical term plain-language dictionary for quick fallback
    terms_db = {
        "hypertension": "High blood pressure - a condition where the force of blood against artery walls is consistently too high.",
        "dyspnea": "Shortness of breath or difficulty breathing.",
        "myocardial infarction": "Heart attack - occurs when blood flow to part of the heart muscle is blocked.",
        "hypoglycemia": "Low blood sugar level.",
        "hyperglycemia": "High blood sugar level.",
        "asthma": "A chronic condition that inflames and narrows the airways in the lungs.",
        "influenza": "The flu - a contagious respiratory illness caused by influenza viruses."
    }
    explanation = terms_db.get(args.term.lower().strip())
    if explanation:
        return {"term": args.term, "plain_explanation": explanation}
    return {"term": args.term, "plain_explanation": f"Medical term '{args.term}': Requires plain-language explanation from verified guidelines."}

def tool_compare_sources(db: Session, user_id: Optional[int], args: CompareSourcesInput) -> Dict[str, Any]:
    results = hybrid_search(db, query=args.topic, user_id=user_id, top_k=6)
    who_claims = [r for r in results if "WHO" in r.get("publisher", "").upper()]
    cdc_claims = [r for r in results if "CDC" in r.get("publisher", "").upper()]
    medline_claims = [r for r in results if "MEDLINEPLUS" in r.get("publisher", "").upper()]

    return {
        "topic": args.topic,
        "who_evidence": who_claims,
        "cdc_evidence": cdc_claims,
        "medlineplus_evidence": medline_claims,
        "total_sources_found": len(results)
    }

def tool_assess_escalation(args: AssessEscalationInput) -> Dict[str, Any]:
    return classify_safety_category(args.query)

def tool_format_citations(args: FormatCitationsInput) -> List[Dict[str, Any]]:
    valid_citations = []
    for c in args.citations:
        if c.get("citation_id") and c.get("title") and c.get("content"):
            valid_citations.append({
                "citation_id": c["citation_id"],
                "title": c["title"],
                "publisher": c.get("publisher", "Verified Source"),
                "page_number": c.get("page_number", 1),
                "publication_date": c.get("publication_date", "N/A"),
                "url": c.get("url", ""),
                "is_outdated": c.get("is_outdated", False),
                "is_superseded": c.get("is_superseded", False)
            })
    return valid_citations
