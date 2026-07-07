import json
import re
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from config.settings import GEMINI_API_KEY, MODEL_NAME, MAX_RETRIES

from prompts.agent2_prompt import build_agent2_system_prompt

from models.schemas import (
    ScientificIntentOutput,
    ClarificationAgentOutput,
    UserResponse,
    ClarificationRound,
)
from security.hallucination_guard import (
    validate_agent2_consistency
)

def extract_json_from_text(text: str) -> dict:
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


def pydantic_schema_for_anthropic(model_class: type[BaseModel]) -> dict:
    schema = model_class.model_json_schema()

    def clean_schema(obj):
        if isinstance(obj, dict):
            obj.pop("title", None)

            for value in obj.values():
                clean_schema(value)

        elif isinstance(obj, list):

            for item in obj:
                clean_schema(item)

    clean_schema(schema)

    return schema


AGENT2_SCHEMA = pydantic_schema_for_anthropic(
    ClarificationAgentOutput
)

def normalize_agent1_plan(agent1_output) -> ScientificIntentOutput:
    if isinstance(agent1_output, ScientificIntentOutput):
        return agent1_output

    if isinstance(agent1_output, str):
        return ScientificIntentOutput.model_validate_json(
            agent1_output
        )

    if isinstance(agent1_output, dict):
        return ScientificIntentOutput.model_validate(
            agent1_output
        )

    raise TypeError(
        "agent1_output must be ScientificIntentOutput, "
        "dict, or JSON string."
    )


def normalize_user_responses(user_responses=None):
    if user_responses is None:
        return []

    normalized = []

    for item in user_responses:

        if isinstance(item, UserResponse):
            normalized.append(
                item.model_dump()
            )

        elif isinstance(item, dict):
            normalized.append(
                UserResponse(**item).model_dump()
            )

        else:
            raise TypeError(
                "Each user response must be a dict "
                "or UserResponse."
            )

    return normalized


def normalize_clarification_history(
    clarification_history=None
):
    if clarification_history is None:
        return []

    normalized = []

    for item in clarification_history:

        if isinstance(item, ClarificationRound):
            normalized.append(
                item.model_dump()
            )

        elif isinstance(item, dict):
            normalized.append(
                ClarificationRound(**item).model_dump()
            )

        else:
            raise TypeError(
                "Each clarification history item "
                "must be a dict or ClarificationRound."
            )

    return normalized

def run_agent2(
    agent1_output,
    user_responses=None,
    clarification_history=None
) -> ClarificationAgentOutput:

    agent1_plan = normalize_agent1_plan(
        agent1_output
    )

    responses = normalize_user_responses(
        user_responses
    )

    history = normalize_clarification_history(
        clarification_history
    )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = f"""
{build_agent2_system_prompt()}

Return only valid JSON matching this schema:

{json.dumps(AGENT2_SCHEMA, indent=2)}

Agent 1 ScientificIntentOutput:

{agent1_plan.model_dump_json(indent=2)}

Optional user clarification responses:

{json.dumps(responses, indent=2)}

Existing clarification history:

{json.dumps(history, indent=2)}
"""

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt,

                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            parsed = json.loads(
                response.text
            )

            validated_output = (
                ClarificationAgentOutput
                .model_validate(parsed)
            )

            validate_agent2_consistency(
                validated_output
            )

            return validated_output

        except ValidationError:
            raise

        except Exception as exc:

            last_error = exc

            error_text = str(exc)

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                wait_time = (
                    2 ** (attempt + 1)
                )

                print(
                    f"Gemini overloaded "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {wait_time} seconds..."
)

                time.sleep(
                    wait_time
                )

                continue

            raise RuntimeError(
                f"Agent 2 failed: {exc}"
            ) from exc

    raise RuntimeError(
        f"Agent 2 failed after retries: "
        f"{last_error}"
    )