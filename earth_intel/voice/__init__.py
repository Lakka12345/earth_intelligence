# voice/__init__.py
# Re-export the public surface so callers can do:
#   from voice import transcribe_bytes, translate_to_english, synthesise, get_or_create

from .stt import transcribe, transcribe_bytes
from .language import translate_to_english, needs_translation
from .tts import synthesise, synthesise_sync
from .session import VoiceSession, get_or_create, get

__all__ = [
    "transcribe",
    "transcribe_bytes",
    "translate_to_english",
    "needs_translation",
    "synthesise",
    "synthesise_sync",
    "VoiceSession",
    "get_or_create",
    "get",
]
