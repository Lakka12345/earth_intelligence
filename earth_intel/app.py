# app.py
# FastAPI backend for the Earth Intelligence Platform.
#
# Run with:
#   uvicorn app:app --reload --host 0.0.0.0 --port 8000
#
# Then in a second terminal run the Streamlit dashboard:
#   streamlit run dashboard_app.py --server.port 8501
#
# The FastAPI backend serves:
#   /api/stt   — Speech-to-text  (POST)
#   /api/tts   — Text-to-speech  (POST)
#   /api/chat  — Agent 1 chat    (POST)
#   /          — Static frontend (index.html voice-only UI)
#
# The Streamlit dashboard at :8501 embeds the voice panel as an iframe
# pointing to http://localhost:8000 for the API calls.

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from api.routes import router

app = FastAPI(
    title="Earth Intelligence Platform",
    description=(
        "Multi-agent scientific dataset discovery system. "
        "Provides STT, TTS, and Agent 1 chat endpoints consumed by both "
        "the standalone voice frontend (index.html) and the full Streamlit dashboard."
    ),
    version="2.0.0",
)

app.include_router(router)


@app.get("/dashboard", include_in_schema=False)
async def dashboard_redirect():
    """Convenience redirect to the Streamlit dashboard."""
    return RedirectResponse(url="http://localhost:8501", status_code=302)


# Serves frontend/index.html (standalone voice UI) at http://localhost:8000
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
