import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Healthcare Information Assistant"
    VERSION: str = "2.0.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-healthcare-jwt-key-batch-09")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./healthcare_v2.db")
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_v2_data")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploaded_docs")
    SUPPORT_EMAIL: str = os.getenv("SUPPORT_EMAIL", "battulasarath751@gmail.com")
    
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
