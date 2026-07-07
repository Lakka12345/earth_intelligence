import re
from security.audit_logger import log_security_event

PROMPT_INJECTION_PATTERNS = [

    r"ignore previous instructions",
    r"ignore all instructions",
    r"system prompt",
    r"developer instructions",
    r"chain of thought",
    r"show hidden prompt",
    r"jailbreak",
    r"roleplay as",
    r"pretend to be",
    r"bypass restrictions",
    r"override safety",
]


class PromptInjectionException(Exception):
    pass


def detect_prompt_injection(query: str):

    lowered = query.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:

        if re.search(pattern, lowered):

            log_security_event(
                "PROMPT_INJECTION",
                pattern
            )

            raise PromptInjectionException(
                f"Potential prompt injection detected: {pattern}"
            )

    return True