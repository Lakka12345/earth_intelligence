import json
import re
import time
import os

from groq import Groq
from pydantic import BaseModel, ValidationError

from models.schemas import DomainName, ScientificIntentOutput
from prompts.agent1_prompt import build_system_prompt
from config.settings import MAX_RETRIES


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


SCHEMA = pydantic_schema_for_anthropic(
    ScientificIntentOutput
)


DOMAIN_SYNONYMS: dict[str, DomainName] = {
    "geography": DomainName.gis,
    "geographic information science": DomainName.gis,
    "geospatial science": DomainName.gis,
    "geospatial analysis": DomainName.gis,
    "cartography": DomainName.gis,
    "earth observation": DomainName.remote_sensing,
    "satellite remote sensing": DomainName.remote_sensing,
    "climatology": DomainName.climate_science,
    "climate": DomainName.climate_science,
    "weather science": DomainName.meteorology,
    "atmospheric science": DomainName.meteorology,
    "water resources": DomainName.hydrology,
    "water resource management": DomainName.hydrology,
    "coastal engineering": DomainName.coastal_processes,
    "coastal geomorphology": DomainName.coastal_processes,
    "environmental science": DomainName.environmental_monitoring,
    "environmental management": DomainName.environmental_monitoring,
    "disaster risk reduction": DomainName.disaster_management,
    "emergency management": DomainName.disaster_management,
    "urban studies": DomainName.urban_planning,
    "city planning": DomainName.urban_planning,
    "infrastructure engineering": DomainName.civil_engineering,
}


def _normalize_domain_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _canonical_domain(value: str) -> str | None:
    normalized = _normalize_domain_name(value)

    for domain in DomainName:
        if normalized == _normalize_domain_name(domain.value):
            return domain.value

    synonym = DOMAIN_SYNONYMS.get(normalized)
    return synonym.value if synonym else None


def _build_abbreviation_map(scientific_variables: list[dict]) -> dict[str, str]:
    """
    Build a lookup table from every abbreviation/synonym → canonical variable name.

    The LLM defines canonical names in scientific_variables[].variable, e.g.:
        "Sea Surface Temperature (SST)"
        "Chlorophyll-a (Chl-a)"
        "Normalized Difference Vegetation Index (NDVI)"

    This function extracts every token inside parentheses as an abbreviation
    and maps it to the full canonical name, so measurements that use the
    abbreviation ("SST", "Chl-a", "NDVI") can be rewritten to the full name
    before the validator runs.

    Also maps the lowercase full name to itself so exact matches pass through.

    Returns: { normalised_abbreviation: canonical_variable_name }
    """
    abbr_map: dict[str, str] = {}

    for var in scientific_variables:
        canonical = var.get("variable", "")
        if not canonical:
            continue

        # Always map the full canonical name to itself (case-insensitive)
        abbr_map[canonical.lower().strip()] = canonical

        # Extract anything in parentheses: "Sea Surface Temperature (SST)" → "SST"
        for match in re.finditer(r"\(([^)]+)\)", canonical):
            abbr = match.group(1).strip()
            if abbr:
                abbr_map[abbr.lower()] = canonical

        # Also map the part before the first parenthesis as a short-form
        # e.g. "Sea Surface Temperature" from "Sea Surface Temperature (SST)"
        base = re.split(r"\s*\(", canonical)[0].strip()
        if base and base.lower() != canonical.lower():
            abbr_map[base.lower()] = canonical

    return abbr_map


def normalize_measurement_references(parsed: dict) -> dict:
    """
    Rewrite measurement.variable_measured to use the canonical variable name
    defined in scientific_variables before ScientificIntentOutput.model_validate
    is called.

    The validator requires every measurement.variable_measured to match an
    entry in scientific_variables[].variable (case-insensitive). LLMs
    frequently use abbreviations ("SST", "NDVI", "Chl-a") in measurements
    even when the canonical name in scientific_variables is the full form
    ("Sea Surface Temperature (SST)"). This function bridges that gap
    without weakening or modifying the validator.

    Also normalizes variable_priorities[].variable by the same map so that
    the second validator check (all scientific_variables must have a priority
    record) does not fail for the same reason.

    Operates on the raw parsed dict in-place and returns it.
    """
    sci_vars = parsed.get("scientific_variables", [])
    if not sci_vars or not isinstance(sci_vars, list):
        return parsed

    abbr_map = _build_abbreviation_map(sci_vars)

    # Normalize measurements
    for measurement in parsed.get("measurements", []):
        if not isinstance(measurement, dict):
            continue
        raw = measurement.get("variable_measured", "")
        canonical = abbr_map.get(raw.lower().strip())
        if canonical and canonical != raw:
            print(
                f"[Agent 1] Normalizing measurement reference: "
                f"'{raw}' → '{canonical}'"
            )
            measurement["variable_measured"] = canonical

    # Normalize variable_priorities for the same reason
    for vp in parsed.get("variable_priorities", []):
        if not isinstance(vp, dict):
            continue
        raw = vp.get("variable", "")
        canonical = abbr_map.get(raw.lower().strip())
        if canonical and canonical != raw:
            print(
                f"[Agent 1] Normalizing variable_priority reference: "
                f"'{raw}' → '{canonical}'"
            )
            vp["variable"] = canonical

    # Normalize variable_dependencies for consistency
    for vd in parsed.get("variable_dependencies", []):
        if not isinstance(vd, dict):
            continue
        raw = vd.get("variable", "")
        canonical = abbr_map.get(raw.lower().strip())
        if canonical and canonical != raw:
            vd["variable"] = canonical
        vd["depends_on"] = [
            abbr_map.get(dep.lower().strip(), dep)
            for dep in vd.get("depends_on", [])
        ]

    return parsed


def normalize_domain_references(parsed: dict) -> dict:
    """
    Rewrite synonymous scientific domain names to the canonical DomainName
    enum values before strict Pydantic validation.

    This keeps downstream agents seeing the same canonical domain strings
    they already expect, while allowing common LLM equivalents such as
    "Geography" or "Earth observation" to validate when they clearly map to
    an existing project domain.
    """
    for domain_confidence in parsed.get("domain_confidences", []):
        if not isinstance(domain_confidence, dict):
            continue
        raw = domain_confidence.get("domain", "")
        canonical = _canonical_domain(raw)
        if canonical and canonical != raw:
            print(
                f"[Agent 1] Normalizing domain reference: "
                f"'{raw}' -> '{canonical}'"
            )
            domain_confidence["domain"] = canonical

    for dependency in parsed.get("cross_domain_dependencies", []):
        if not isinstance(dependency, dict):
            continue
        normalized_domains = []
        for raw in dependency.get("domains", []):
            canonical = _canonical_domain(raw)
            if canonical and canonical != raw:
                print(
                    f"[Agent 1] Normalizing cross-domain reference: "
                    f"'{raw}' -> '{canonical}'"
                )
            normalized_domains.append(canonical or raw)
        dependency["domains"] = normalized_domains

    return parsed


def run_agent1(query: str) -> ScientificIntentOutput:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
{build_system_prompt()}

Return only valid JSON matching the ScientificIntentOutput schema.

JSON schema:
{json.dumps(SCHEMA, indent=2)}

User query:
{query}
"""

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                },
            )

            raw = (
                response
                .choices[0]
                .message.content
            )

            parsed = extract_json_from_text(
                raw
            )

            print(
                "\n===== RAW PARSED JSON ====="
            )

            print(
                json.dumps(
                    parsed,
                    indent=2
                )
            )

            print(
                "===========================\n"
            )

            # Normalize measurement and priority references before validation.
            # The LLM often uses abbreviations ("SST", "NDVI", "Chl-a") in
            # measurements even when scientific_variables uses the full canonical
            # form. This rewrites abbreviated references to the canonical name so
            # the schema validator's cross-reference check always passes.
            parsed = normalize_measurement_references(parsed)
            parsed = normalize_domain_references(parsed)

            return (
                ScientificIntentOutput
                .model_validate(parsed)
            )

        except ValidationError:
            raise

        except Exception as exc:

            last_error = exc

            wait_time = (
                2 ** (attempt + 1)
            )

            print(
                f"Groq failed "
                f"(attempt {attempt + 1}/{MAX_RETRIES}). "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )

            continue

    raise RuntimeError(
        f"Agent 1 failed after retries: {last_error}"
    )
