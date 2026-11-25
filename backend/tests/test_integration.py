import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import engine, Base, init_fts5

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    init_fts5()

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "database" in data
        assert "embeddings" in data
        assert "llm" in data

def test_emergency_query_chat():
    with TestClient(app) as client:
        payload = {"query": "I have severe chest pain and slurred speech help me!"}
        response = client.post("/api/chat/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safety_category"] == "A_EMERGENCY"
        assert data["escalated"] is True
        assert "112" in data["answer"] or "108" in data["answer"] or "Tele-MANAS" in data["answer"]

def test_general_rag_query():
    with TestClient(app) as client:
        payload = {"query": "What are the guidelines for blood pressure management in hypertension?"}
        response = client.post("/api/chat/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "safety_category" in data

def test_document_list():
    with TestClient(app) as client:
        response = client.get("/api/documents/list?doc_type=trusted")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

def test_api_key_test_endpoint():
    with TestClient(app) as client:
        response = client.post("/api/health/test-key", json={})
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert data["success"] is True


def test_liveness_probe_endpoint():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_register_forces_user_role():
    import uuid
    uname = f"testuser_{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        response = client.post("/api/auth/register", json={
            "username": uname,
            "email": f"{uname}@example.com",
            "password": "StrongPassword123!",
            "role": "admin"
        })
        assert response.status_code == 200
        assert response.json()["role"] == "user"


def test_history_session_isolation():
    with TestClient(app) as client:
        response_no_session = client.get("/api/history")
        assert response_no_session.status_code == 200
        assert response_no_session.json() == []

        response_unique = client.get("/api/history?session_id=nonexistent_sess_xyz_999")
        assert response_unique.status_code == 200
        assert response_unique.json() == []

