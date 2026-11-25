"""
AI Healthcare Information Assistant - Main Server Entry Point
Runs the FastAPI application server using Uvicorn.
"""
import os
import socket
import uvicorn
from app.main import app


def _find_available_port(host: str, preferred_port: int, max_attempts: int = 20) -> int:
    """Return preferred_port if free, or find next free local port when PORT is not pinned."""
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((bind_host, candidate)) != 0:
                return candidate
    return preferred_port


if __name__ == "__main__":
    env_port = os.getenv("PORT")
    host = os.getenv("HOST", "0.0.0.0" if env_port else "127.0.0.1")
    port = int(env_port) if env_port else _find_available_port(host, 8000)
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    print(f"Starting AI Healthcare Information Assistant on http://{host}:{port} ...")
    uvicorn.run(
        "app.main:app",
        app_dir=os.path.dirname(os.path.abspath(__file__)),
        host=host,
        port=port,
        reload=reload_enabled,
    )
