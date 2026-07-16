"""
api/routes.py
FastAPI router that adds three voice endpoints to your existing server:

  POST /api/stt   — upload audio → get English transcript back
  POST /api/tts   — send text → get mp3 audio back
  POST /api/chat  — send text (already English) → get Agent 1's next question

HOW TO WIRE INTO YOUR EXISTING MAIN:
-------------------------------------
In your main FastAPI app file, add:

    from api.routes import router
    app.include_router(router)

That's it. Your existing routes are untouched.

HOW AGENT 1 IS CALLED:
------------------------
At the bottom of this file is a stub: `_call_agent1(text, session_id)`.
Replace its body with however you currently invoke Agent 1 from the CLI.
For example, if you do:  python agent1.py --query "..."
just import and call the relevant function directly instead.
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    session_id: str | None = None   # optional; used for voice analytics only


class ChatRequest(BaseModel):
    message: str                    # always plain English by the time it arrives here
    session_id: str | None = None


class STTResponse(BaseModel):
    text: str                       # English transcript (translated if needed)
    original_text: str | None       # non-English original, if translation occurred
    language: str                   # ISO 639-1 code e.g. "hi", "en", "te"
    language_prob: float
    was_translated: bool
    session_id: str


class ChatResponse(BaseModel):
    question: str | None            # Agent 1's next clarification question, or None if done
    done: bool                      # True when Agent 1 has enough info
    session_id: str


# ---------------------------------------------------------------------------
# POST /api/stt
# ---------------------------------------------------------------------------

@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    audio: UploadFile = File(...),
    session_id: str | None = None,
):
    """
    Accept a voice recording (webm/wav/mp3 — whatever MediaRecorder produces),
    transcribe it, translate to English if needed, update session language state.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"

    # 1. Transcribe
    result = transcribe_bytes(audio_bytes, suffix=suffix)
    original_text = result["text"]
    language = result["language"]
    lang_prob = result["language_prob"]

    # 2. Translate if not English
    was_translated = needs_translation(language)
    english_text = translate_to_english(original_text, language) if was_translated else original_text

    # 3. Update session
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


# ---------------------------------------------------------------------------
# POST /api/tts
# ---------------------------------------------------------------------------

@router.post("/tts")
async def text_to_speech(body: TTSRequest):
    """
    Convert text to speech (mp3). The browser plays this when the user
    clicks the 🔊 button next to any Agent 1 question.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty.")

    output_path = f"/tmp/tts_{uuid4().hex}.mp3"
    try:
        await synthesise(body.text, output_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # FileResponse streams the file then FastAPI auto-deletes it (background_tasks
    # pattern not needed here because FileResponse itself handles the lifecycle).
    return FileResponse(
        output_path,
        media_type="audio/mpeg",
        filename="response.mp3",
    )


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    Receive plain English text (typed or transcribed+translated voice),
    pass it to Agent 1, return Agent 1's next question.

    Agent 1 is completely unaware whether the input came from voice or keyboard.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message field is empty.")

    session = get_or_create(body.session_id)

    try:
        result = _call_agent1(body.message, session.session_id)
    except Exception as e:
        logger.error(f"Agent 1 error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 1 error: {e}")

    if result.get("question"):
        session.add_agent_turn(result["question"])

    return ChatResponse(
        question=result.get("question"),
        done=result.get("done", False),
        session_id=session.session_id,
    )


# ---------------------------------------------------------------------------
# ── STUB: replace this with your real Agent 1 call ─────────────────────────
#
# Current CLI equivalent:  python agent1.py --query "<message>"
#
# To wire in your agent, import whatever function processes the user query
# and replace the body of this function. The return value must be a dict:
#   {"question": "<next clarification question>", "done": False}
#   {"question": None, "done": True}   ← when Agent 1 is satisfied
# ---------------------------------------------------------------------------

def _call_agent1(message: str, session_id: str) -> dict:
    """
    REPLACE THIS STUB with your real Agent 1 invocation.

    Example (adjust import path to match your project):

        from agents.agent1 import Agent1
        agent = Agent1()
        response = agent.process(message, session_id=session_id)
        return {"question": response.next_question, "done": response.complete}
    """
    # Stub: echoes the message back as a mock question for testing
    return {
        "question": f"[STUB] Agent 1 received: '{message}'. Replace _call_agent1() in api/routes.py.",
        "done": False,
    }
