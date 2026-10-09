"""
AI Healthcare Information Assistant - Main Server Entry Point
Runs the FastAPI application server using Uvicorn.
"""
import uvicorn
from app.main import app

if __name__ == "__main__":
    print("Starting AI Healthcare Information Assistant on http://127.0.0.1:5000 & http://127.0.0.1:8000 ...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)
