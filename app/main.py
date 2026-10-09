import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine, Base, init_fts5, SessionLocal
from app.routes import auth, documents, chat, summarize, admin, health, history, contact
from app.models import User, TrustedSource, Document
from app.security import hash_password
from app.ingestion import validate_file, extract_text_with_pages, chunk_pages_with_overlap
from app.rag_engine import index_document_chunks

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Document-based AI Healthcare Information Assistant with multi-layer safety and escalation queue."
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(summarize.router)
app.include_router(admin.router)
app.include_router(health.router)
app.include_router(history.router)
app.include_router(contact.router)


@app.on_event("startup")
def startup_event():
    # Initialize DB tables & SQLite FTS5
    Base.metadata.create_all(bind=engine)
    init_fts5()

    # Seed Default Admin User & Trusted Sources if database is new
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@healthcare.org",
                hashed_password=hash_password("Admin@12345"),
                role="admin"
            )
            db.add(admin_user)

        sample_sources = [
            {
                "title": "WHO Guidelines on Hypertension",
                "publisher": "WHO",
                "url": "https://www.who.int/publications/i/item/9789240033986",
                "publication_date": "2023-01-15",
                "version": "2.1",
                "status": "approved",
                "file_path": "knowledge/who_hypertension_guide.txt"
            },
            {
                "title": "CDC Diabetes Basics and Management",
                "publisher": "CDC",
                "url": "https://www.cdc.gov/diabetes/basics/index.html",
                "publication_date": "2024-02-10",
                "version": "3.0",
                "status": "approved",
                "file_path": "knowledge/cdc_diabetes_basics.txt"
            },
            {
                "title": "MedlinePlus Asthma Care & Triggers",
                "publisher": "MedlinePlus",
                "url": "https://medlineplus.gov/asthma.html",
                "publication_date": "2023-11-20",
                "version": "1.5",
                "status": "approved",
                "file_path": "knowledge/medlineplus_asthma_management.txt"
            },
            {
                "title": "CDC Influenza Guidelines 2024",
                "publisher": "CDC",
                "url": "https://www.cdc.gov/flu/guidelines",
                "publication_date": "2024-01-05",
                "version": "2024.1",
                "status": "approved",
                "file_path": "knowledge/cdc_influenza_flu_guidelines.txt"
            },
            {
                "title": "WHO First Aid Basics & Emergency Guidelines",
                "publisher": "WHO",
                "url": "https://www.who.int/first-aid",
                "publication_date": "2023-08-30",
                "version": "1.0",
                "status": "approved",
                "file_path": "knowledge/who_first_aid_basics.txt"
            }
        ]

        for src_data in sample_sources:
            existing_src = db.query(TrustedSource).filter(TrustedSource.title == src_data["title"]).first()
            if not existing_src:
                ts = TrustedSource(
                    title=src_data["title"],
                    publisher=src_data["publisher"],
                    url=src_data["url"],
                    publication_date=src_data["publication_date"],
                    version=src_data["version"],
                    status=src_data["status"]
                )
                db.add(ts)
                db.flush()

                # Seed sample document from file if available
                file_p = src_data["file_path"]
                if os.path.exists(file_p):
                    with open(file_p, "rb") as f:
                        content = f.read()

                    doc = Document(
                        file_hash=f"seed_{ts.id}",
                        filename=os.path.basename(file_p),
                        stored_filename=f"seed_{ts.id}_{os.path.basename(file_p)}",
                        file_type="txt",
                        file_size=len(content),
                        doc_type="trusted",
                        trusted_source_id=ts.id,
                        title=src_data["title"],
                        publisher=src_data["publisher"],
                        url=src_data["url"],
                        publication_date=src_data["publication_date"],
                        status="indexed"
                    )
                    db.add(doc)
                    db.flush()

                    pages_text = extract_text_with_pages(content, ".txt")
                    chunks = chunk_pages_with_overlap(pages_text)
                    index_document_chunks(db, doc, chunks)

        db.commit()
    finally:
        db.close()

# Mount Static UI if present
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
if os.path.exists(templates_dir):
    app.mount("/static", StaticFiles(directory=templates_dir), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "..", "templates", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/health"
    }
