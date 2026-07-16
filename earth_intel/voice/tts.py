"""
voice/tts.py
Text-to-speech using edge-tts (Microsoft neural voices, free, no key needed).
Falls back to Coqui TTS if edge-tts fails or is unavailable (fully offline).

Usage:
    audio_path = await synthesise("What region are you querying?")
    # returns path to a .mp3 file you can serve directly
"""

import asyncio
import logging
import os
import tempfile
from uuid import uuid4

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Primary: edge-tts
# Neural voice, streams fast, requires internet on first call.
# Full voice list: `edge-tts --list-voices`
# Good choices for neutral English: en-US-JennyNeural, en-IN-NeerjaNeural
# ------------------------------------------------------------------
PRIMARY_VOICE = "en-IN-NeerjaNeural"   # Indian-accented English — fits INCOIS context well
FALLBACK_VOICE = "en-US-JennyNeural"


async def _synthesise_edge(text: str, output_path: str) -> bool:
    """Try edge-tts. Returns True on success, False on any failure."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice=PRIMARY_VOICE)
        await communicate.save(output_path)
        return True
    except Exception as e:
        logger.warning(f"edge-tts failed ({e}), will try fallback.")
        return False


def _synthesise_coqui(text: str, output_path: str) -> bool:
    """Try Coqui TTS (offline fallback). Returns True on success."""
    try:
        from TTS.api import TTS as CoquiTTS
        tts = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        # Coqui outputs .wav; we save as .wav and rename
        wav_path = output_path.replace(".mp3", ".wav")
        tts.tts_to_file(text=text, file_path=wav_path)
        os.rename(wav_path, output_path)
        return True
    except Exception as e:
        logger.error(f"Coqui TTS also failed: {e}")
        return False


async def synthesise(text: str, output_path: str | None = None) -> str:
    """
    Convert text to speech. Returns the path to the generated audio file.

    Args:
        text: The text to speak.
        output_path: Where to write the audio file.
                     If None, a temp file is created (caller must delete it).

    Returns:
        Path to the audio file (mp3 or wav depending on which backend succeeded).

    Raises:
        RuntimeError: If both edge-tts and Coqui fail.
    """
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid4().hex}.mp3")

    success = await _synthesise_edge(text, output_path)
    if not success:
        success = _synthesise_coqui(text, output_path)

    if not success:
        raise RuntimeError(
            "TTS failed: both edge-tts and Coqui TTS are unavailable. "
            "Install either with: pip install edge-tts  OR  pip install TTS"
        )

    return output_path


def synthesise_sync(text: str, output_path: str | None = None) -> str:
    """Synchronous wrapper around synthesise() for use outside async contexts."""
    return asyncio.run(synthesise(text, output_path))
