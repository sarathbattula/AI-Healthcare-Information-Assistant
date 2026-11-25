"""Root launcher delegating to backend/app.py so `python app.py` works from project root."""
import os
import runpy
import sys

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    runpy.run_path(os.path.join(BACKEND_DIR, "app.py"), run_name="__main__")
