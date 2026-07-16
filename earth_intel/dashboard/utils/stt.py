"""
utils/stt.py
Speech-to-text helper for the native (in-Streamlit) voice input widget.

Uses Groq's hosted Whisper models (whisper-large-v3 / whisper-large-v3-turbo)
via the same `groq` client already used by Agent 1 for its LLM calls
(see agents/agent1.py) -- one API key, one provider, for both text and
speech in this project.

Requires:
    pip install groq
    export GROQ_API_KEY=...
"""

import io
import os
from typing import Optional

# "turbo" is faster and cheaper; swap to "whisper-large-v3" if you need
# the (slightly) higher-accuracy non-turbo model instead.
_STT_MODEL = "whisper-large-v3-turbo"


def transcribe_audio(audio_bytes: bytes, filename: str = "recording.wav") -> Optional[str]:
    """
    Transcribe recorded audio to text using Groq's Whisper endpoint.

    Args:
        audio_bytes: Raw audio bytes as captured by st.audio_input
            (WAV-encoded by Streamlit).
        filename: Filename hint passed to the API; the extension matters
            for format detection on some providers.

    Returns:
        The transcribed text, or None if transcription failed. Callers
        must treat None as "no text available" and must not crash the
        pipeline on a failed transcription -- surface it as a UI warning
        instead (see app.py's usage).
    """
    if not audio_bytes:
        return None

    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "The `groq` package is required for transcribe_audio(). "
            "Install it with `pip install groq`."
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in your environment "
            "(the same key used by agents/agent1.py for Groq LLM calls)."
        )

    client = Groq(api_key=api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename  # the SDK uses this for format detection

    try:
        result = client.audio.transcriptions.create(
            model=_STT_MODEL,
            file=audio_file,
        )
        text = getattr(result, "text", None)
        return text.strip() if text else None
    except Exception as exc:
        # Non-fatal by design: a failed transcription should surface as a
        # warning in the UI, not crash the whole page.
        print(f"[STT] Transcription failed (non-fatal): {exc}")
        return None
