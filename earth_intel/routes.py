"""
api/routes.py
Three voice/chat endpoints wired to your real Agent 1.

Agent 1 (run_agent1) is a pure analysis step — it takes the user's query
and returns a ScientificIntentOutput object. It asks no questions itself.
So the web flow is:

  1. User submits query  →  POST /api/chat
  2. Agent 1 runs, produces ScientificIntentOutput
  3. Frontend receives the summary and moves to the next stage

STT and TTS endpoints are independent of agent logic.
"""

import os
import logging
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from voice.stt import transcribe_bytes
from voice.language import translate_to_english, needs_translation
from voice.tts import synthesise
from voice.session import get_or_create

# ── Real Agent 1 import ────────────────────────────────────────────────────
from agents.agent1 import run_agent1
from models.schemas import ScientificIntentOutput

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ── Pydantic schemas ───────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    session_id: str | None = None


class ChatRequest(BaseModel):
    message: str          # always plain English by the time it reaches here
    session_id: str | None = None


class STTResponse(BaseModel):
    text: str             # English transcript (translated if needed)
    original_text: str | None
    language: str
    language_prob: float
    was_translated: bool
    session_id: str


class ChatResponse(BaseModel):
    # Human-readable one-line summary of what Agent 1 understood.
    # The frontend shows this as a confirmation before moving to Agent 2.
    summary: str
    # The full ScientificIntentOutput serialised to a dict so the frontend
    # (or a future Agent 2 API endpoint) can use it directly.
    agent1_output: dict
    session_id: str


# ── POST /api/stt ──────────────────────────────────────────────────────────

@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    audio: UploadFile = File(...),
    session_id: str | None = None,
):
    """
    Accept a voice recording (webm/wav/mp3),
    transcribe it, translate to English if needed.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"

    result = transcribe_bytes(audio_bytes, suffix=suffix)
    original_text = result["text"]
    language = result["language"]
    lang_prob = result["language_prob"]

    was_translated = needs_translation(language)
    english_text = translate_to_english(original_text, language) if was_translated else original_text

    session = get_or_create(session_id)
    session.add_user_turn(
        text=english_text,
        via="voice",
        original=original_text if was_translated else "",
        language=language,
    )

    logger.info(
        f"STT [{session.session_id}]: lang={language} "
        f"translated={was_translated} text={english_text[:60]!r}"
    )

    return STTResponse(
        text=english_text,
        original_text=original_text if was_translated else None,
        language=language,
        language_prob=lang_prob,
        was_translated=was_translated,
        session_id=session.session_id,
    )


# ── POST /api/tts ──────────────────────────────────────────────────────────

@router.post("/tts")
async def text_to_speech(body: TTSRequest):
    """Convert text to speech (mp3). Used by the 🔊 button in the frontend."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty.")

    output_path = f"/tmp/tts_{uuid4().hex}.mp3"
    try:
        await synthesise(body.text, output_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return FileResponse(output_path, media_type="audio/mpeg", filename="response.mp3")


# ── POST /api/chat ─────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    Receive the user's scientific query (plain English, typed or transcribed),
    run Agent 1, and return:
      - a one-line human-readable summary for the frontend to display
      - the full ScientificIntentOutput as a dict for downstream use

    Agent 1 never sees whether input came from voice or keyboard.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message field is empty.")

    session = get_or_create(body.session_id)
    session.add_user_turn(text=body.message, via="text")

    try:
        result: ScientificIntentOutput = run_agent1(body.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent 1 error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 1 failed: {e}")

    # Build a concise summary line from the ScientificIntentOutput fields
    # so the frontend can confirm what Agent 1 understood.
    primary_domain = (
        result.domain_confidences[0].domain.value
        if result.domain_confidences
        else "unknown domain"
    )
    n_vars = len(result.scientific_variables)
    location = (
        result.spatial_context.primary_location
        if result.spatial_context and result.spatial_context.primary_location
        else "unspecified location"
    )
    summary = (
        f"Understood: {primary_domain} query about {n_vars} variable(s) "
        f"at {location}. Moving to clarification."
    )

    session.add_agent_turn(summary)

    return ChatResponse(
        summary=summary,
        agent1_output=result.model_dump(),
        session_id=session.session_id,
    )
