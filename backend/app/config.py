import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
ROOT_DIR = os.path.dirname(BASE_DIR)  # project root

_backend_env = os.path.join(BASE_DIR, ".env")
_root_env = os.path.join(ROOT_DIR, ".env")
ENV_PATH = _backend_env if os.path.exists(_backend_env) else _root_env
load_dotenv(ENV_PATH)


def _resolve_path(path_value: str, default_rel: str) -> str:
    raw = (path_value or default_rel).strip()
    if os.path.isabs(raw):
        return raw
    return os.path.normpath(os.path.join(BASE_DIR, raw))


def _resolve_database_url(url_value: str) -> str:
    default_db = os.path.join(BASE_DIR, "healthcare_v2.db").replace("\\", "/")
    raw = (url_value or f"sqlite:///{default_db}").strip()
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql://", 1)
    if raw.startswith("sqlite:///./"):
        rel = raw[len("sqlite:///./"):]
        abs_path = os.path.join(BASE_DIR, rel).replace("\\", "/")
        return f"sqlite:///{abs_path}"
    return raw


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Healthcare Information Assistant"
    VERSION: str = "2.0.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-healthcare-jwt-key-batch-09")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    DATABASE_URL: str = _resolve_database_url(os.getenv("DATABASE_URL", ""))
    CHROMA_DB_DIR: str = _resolve_path(os.getenv("CHROMA_DB_DIR", ""), "chroma_v2_data")
    UPLOAD_DIR: str = _resolve_path(os.getenv("UPLOAD_DIR", ""), "uploaded_docs")
    KNOWLEDGE_DIR: str = _resolve_path(os.getenv("KNOWLEDGE_DIR", ""), "knowledge")
    FRONTEND_DIR: str = _resolve_path(os.getenv("FRONTEND_DIR", ""), os.path.join(ROOT_DIR, "frontend"))

    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", "")
    SUPPORT_EMAIL: str = os.getenv("SUPPORT_EMAIL", "battulasarath751@gmail.com")

    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")

    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")


settings = Settings()
