from typing import Optional, List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from groq import Groq

from app.database import get_db
from app.config import settings
from app.rag_engine import _collection, get_embedding_model

router = APIRouter(prefix="/api/health", tags=["Health"])

VALID_GROQ_MODELS = [
    settings.GROQ_MODEL,
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b"
]

class ComponentStatus(BaseModel):
    status: str
    details: str

class SystemHealthResponse(BaseModel):
    overall_status: str
    llm: ComponentStatus
    embeddings: ComponentStatus
    database: ComponentStatus
    vector_store: ComponentStatus

class TestKeyInput(BaseModel):
    api_key: Optional[str] = None

class TestKeyResponse(BaseModel):
    success: bool
    message: str
    model_used: Optional[str] = None
    available_models: Optional[List[str]] = None

@router.get("", response_model=SystemHealthResponse)
def get_system_health(db: Session = Depends(get_db)):
    # 1. Check Database
    db_status = ComponentStatus(status="HEALTHY", details="SQLite connection established.")
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = ComponentStatus(status="DEGRADED", details=f"Database error: {str(e)}")

    # 2. Check Embeddings
    embed_status = ComponentStatus(status="HEALTHY", details=f"SentenceTransformer '{settings.EMBEDDING_MODEL_NAME}' loaded.")
    try:
        model = get_embedding_model()
        vec = model.encode("health check").tolist()
        if len(vec) == 0:
            embed_status = ComponentStatus(status="DEGRADED", details="Embedding returned 0 dimensions.")
    except Exception as e:
        embed_status = ComponentStatus(status="UNHEALTHY", details=f"Embedding model failure: {str(e)}")

    # 3. Check Vector Store
    vector_status = ComponentStatus(status="HEALTHY", details="ChromaDB collection active.")
    try:
        count = _collection.count()
        vector_status.details = f"ChromaDB active with {count} indexed vectors."
    except Exception as e:
        vector_status = ComponentStatus(status="DEGRADED", details=f"ChromaDB error: {str(e)}")

    # 4. Check LLM Configuration & Connection
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.strip() == "":
        llm_status = ComponentStatus(
            status="UNCONFIGURED",
            details=f"GROQ_API_KEY is missing from environment. Active model: '{settings.GROQ_MODEL}'."
        )
    else:
        # Perform quick ping to verify Groq API connection
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)
            models_to_try = [m for m in VALID_GROQ_MODELS if m]
            connected_model = None
            for m in models_to_try:
                try:
                    res = client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=2,
                        temperature=0.0
                    )
                    if res and res.choices:
                        connected_model = m
                        break
                except Exception:
                    continue
            
            if connected_model:
                llm_status = ComponentStatus(
                    status="CONFIGURED",
                    details=f"Groq API connected successfully. Active model: '{connected_model}'."
                )
            else:
                llm_status = ComponentStatus(
                    status="DEGRADED",
                    details=f"Groq API key found but failed to respond on candidate models."
                )
        except Exception as e:
            llm_status = ComponentStatus(
                status="UNHEALTHY",
                details=f"Groq client error: {str(e)}"
            )

    # Determine overall status
    statuses = [db_status.status, embed_status.status, vector_status.status]
    if "UNHEALTHY" in statuses:
        overall = "UNHEALTHY"
    elif "DEGRADED" in statuses or llm_status.status in ["UNCONFIGURED", "DEGRADED"]:
        overall = "DEGRADED"
    else:
        overall = "HEALTHY"

    return {
        "overall_status": overall,
        "llm": llm_status,
        "embeddings": embed_status,
        "database": db_status,
        "vector_store": vector_status
    }

@router.post("/test-key", response_model=TestKeyResponse)
def test_groq_api_key(payload: Optional[TestKeyInput] = None):
    key_to_test = None
    if payload and payload.api_key and payload.api_key.strip():
        key_to_test = payload.api_key.strip()
    else:
        key_to_test = settings.GROQ_API_KEY.strip() if settings.GROQ_API_KEY else None

    if not key_to_test:
        return TestKeyResponse(
            success=False,
            message="No API key provided and no default GROQ_API_KEY set in system configuration."
        )

    try:
        client = Groq(api_key=key_to_test)
        models_to_try = [m for m in VALID_GROQ_MODELS if m]
        
        last_error = None
        for m in models_to_try:
            try:
                res = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                    temperature=0.0
                )
                if res and res.choices:
                    return TestKeyResponse(
                        success=True,
                        message=f"✅ API Key verified! Connected to Groq AI using model '{m}'.",
                        model_used=m
                    )
            except Exception as e:
                last_error = str(e)
                continue

        return TestKeyResponse(
            success=False,
            message=f"Groq API Key test failed: {last_error or 'All model endpoints unreachable'}"
        )
    except Exception as e:
        return TestKeyResponse(
            success=False,
            message=f"Failed to connect to Groq: {str(e)}"
        )

