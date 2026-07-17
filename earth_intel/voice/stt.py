"""
voice/stt.py
Speech-to-text using faster-whisper.
Runs fully locally on CPU — no API key needed.
"""

import os
import tempfile
from faster_whisper import WhisperModel

# "base" is a good balance of speed vs accuracy for Indian-accented English / Hindi.
# Upgrade to "small" or "medium" if accuracy is poor; costs more RAM.
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load so import doesn't block startup."""
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> dict:
    """
    Transcribe an audio file to text.

    Returns:
        {
            "text": str,           # full transcript
            "language": str,       # ISO 639-1 code e.g. "en", "hi", "te"
            "language_prob": float # confidence in detected language
        }
    """
    model = _get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_prob": round(info.language_probability, 3),
    }


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".webm") -> dict:
    """
    Convenience wrapper: accepts raw bytes from an HTTP upload,
    writes to a temp file, then transcribes.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
