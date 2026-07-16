"""
voice/language.py
Language detection and translation to English.

Translation uses Helsinki-NLP/opus-mt models from HuggingFace — fully local,
no API key. Models are downloaded on first use and cached in ~/.cache/huggingface.

Supported source languages for translation (non-exhaustive):
  hi (Hindi), te (Telugu), ta (Tamil), kn (Kannada), ml (Malayalam),
  bn (Bengali), mr (Marathi), fr, de, es, zh, ar, ...
  Full list: https://huggingface.co/Helsinki-NLP
"""

import logging
from functools import lru_cache
from transformers import pipeline

logger = logging.getLogger(__name__)

# Languages we can pass straight through without translation
_ENGLISH_CODES = {"en"}

# Whisper uses ISO 639-1 two-letter codes; opus-mt uses the same convention.
# A few Whisper codes differ from opus-mt — patch them here as needed.
_LANG_CODE_MAP = {
    "zh": "zh",   # Chinese (Whisper returns "zh", opus-mt uses "zh")
    "jw": "jv",   # Javanese (Whisper uses "jw", opus-mt uses "jv")
}


@lru_cache(maxsize=8)
def _get_translator(source_lang: str):
    """
    Load (and cache) the Helsinki-NLP translation pipeline for a given source language.
    lru_cache means each language pair is loaded only once per process lifetime.
    """
    lang = _LANG_CODE_MAP.get(source_lang, source_lang)
    model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
    logger.info(f"Loading translation model: {model_name}")
    try:
        return pipeline("translation", model=model_name)
    except Exception as e:
        raise ValueError(
            f"No translation model found for language '{source_lang}'. "
            f"Check https://huggingface.co/Helsinki-NLP for available pairs. "
            f"Original error: {e}"
        )


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text to English if it isn't already English.
    Returns the original text unchanged if source_lang == "en".
    """
    if source_lang in _ENGLISH_CODES:
        return text
    if not text.strip():
        return text

    translator = _get_translator(source_lang)
    result = translator(text, max_length=512)
    return result[0]["translation_text"]


def needs_translation(language_code: str) -> bool:
    """Return True if the detected language requires translation."""
    return language_code not in _ENGLISH_CODES
