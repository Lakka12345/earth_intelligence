"""
fastapi_backend.py — Voice & Chat API backend for Earth Intelligence Platform
==============================================================================
Exposes three endpoints consumed by the voice component in dashboard/components/voice.py:

  POST /api/stt   — audio file → transcribed text (faster-whisper)
  POST /api/chat  — text query → Agent 1 ScientificIntentOutput
  POST /api/tts   — text → audio bytes (edge-tts)

Run alongside the Streamlit dashboard:

  # Terminal 1 — this file
  uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000

  # Terminal 2 — Streamlit
  streamlit run dashboard_app.py

All heavy imports (whisper model, translation pipeline) are lazy-loaded on
first use so the server starts instantly even on a slow machine.
"""

from __future__ import annotations

import os
import sys
import uuid
import tempfile
import asyncio
from pathlib import Path
from typing import Optional

# ── Make sure project root is on sys.path so agent imports resolve ────────────
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Earth Intelligence Voice API", version="1.0.0")

# Allow the Streamlit origin (localhost:8501) to call this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to ["http://localhost:8501"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # "base" is fast and accurate enough for scientific queries.
        # Switch to "small" or "medium" for better multilingual accuracy.
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


_translator = None
_translator_tokenizer = None

def _get_translator():
    """Helsinki-NLP opus-mt multilingual → English translator."""
    global _translator, _translator_tokenizer
    if _translator is None:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        _translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translator = MarianMTModel.from_pretrained(model_name)
    return _translator, _translator_tokenizer


def _translate_to_english(text: str, detected_lang: str) -> tuple[str, bool]:
    """
    Translate `text` to English if it's not already English.
    Returns (translated_text, was_translated).
    Falls back to the original text gracefully if translation fails.
    """
    if detected_lang in ("en", "unknown"):
        return text, False
    try:
        model, tokenizer = _get_translator()
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated_ids = model.generate(**inputs)
        translated = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        return translated, True
    except Exception:
        return text, False


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store  (good enough for local/internship use)
# ─────────────────────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: Optional[str]) -> str:
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {"history": []}
    return session_id


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/stt
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Receive a WebM/WAV/OGG audio file, transcribe with faster-whisper,
    optionally translate to English, and return JSON:
      { text, language, was_translated, original, session_id }
    """
    session_id = _get_or_create_session(session_id)

    # Write upload to a temp file — faster-whisper needs a path
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        model = _get_whisper()
        segments, info = model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = info.language or "unknown"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        os.unlink(tmp_path)

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected in audio.")

    original = transcript
    translated_text, was_translated = _translate_to_english(transcript, detected_lang)

    return {
        "text":           translated_text,
        "language":       detected_lang,
        "was_translated": was_translated,
        "original":       original if was_translated else "",
        "session_id":     session_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Run Agent 1 on the user's text query and return:
      { summary, agent1_output (dict), session_id }
    """
    session_id = _get_or_create_session(req.session_id)

    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message.")

    try:
        from agents.agent1 import run_agent1
        result = run_agent1(req.message.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 1 failed: {exc}")

    # Build a human-readable summary from the structured output
    try:
        variables = ", ".join(
            v.variable for v in (result.scientific_variables or []) if v.variable
        ) or "not specified"
        location = (
            getattr(result.spatial_context, "primary_location", None)
            or getattr(result.spatial_context, "location", None)
            or "not specified"
        )
        period = (
            getattr(result.temporal_context, "period_description", None)
            or getattr(result.temporal_context, "date_range", None)
            or "not specified"
        )
        summary = (
            f"I've parsed your query. "
            f"Variables: {variables}. "
            f"Location: {location}. "
            f"Time period: {period}. "
            f"Use the full dashboard pipeline to proceed to Agent 2 and discovery."
        )
    except Exception:
        summary = "Agent 1 analysis complete. See the structured output below."

    _sessions[session_id]["history"].append({
        "role": "user", "content": req.message
    })
    _sessions[session_id]["history"].append({
        "role": "agent1", "content": summary
    })

    return {
        "summary":       summary,
        "agent1_output": result.model_dump(),
        "session_id":    session_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/tts
# ─────────────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    voice: str = "en-IN-NeerjaNeural"   # Indian English — fits INCOIS context


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convert text to speech using edge-tts and stream the audio bytes back.
    The voice component plays this directly via the Web Audio API.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Empty text.")

    # edge-tts is async-native
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="edge-tts not installed. Run: pip install edge-tts"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(req.text.strip(), req.voice)
        await communicate.save(tmp_path)
        audio_bytes = Path(tmp_path).read_bytes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")
    finally:
        os.unlink(tmp_path)

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=tts.mp3"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Earth Intelligence Voice API"}


# ─────────────────────────────────────────────────────────────────────────────
# Run directly: python fastapi_backend.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)
