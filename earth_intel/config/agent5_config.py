"""
Agent 5 configuration -- environment-variable overrides, following the
same pattern established for Agent 4's config.py.

ASSUMPTION FLAGGED: settings.py (used elsewhere in the project) is
configured for Gemini (GEMINI_API_KEY / gemini-2.5-flash). Agent 3
(agent3_discovery.py) instead uses Groq via GROQ_MODEL. This file
follows the Groq pattern to stay consistent with Agent 3's scoring
calls, since Agent 5's judgment calls (choosing preprocessing
strategy, validating against the scientific objective) are the same
kind of reasoning task. If the project is meant to standardize on
Gemini instead, only this file and agent5.py's client construction
need to change.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_RETRIES = int(os.getenv("AGENT5_MAX_RETRIES", "3"))

# Where cleaned/standardized outputs are written. Zarr is used as the
# canonical on-disk format for cleaned/merged datasets (chunked,
# self-describing, works natively with xarray) regardless of what
# format the source arrived in.
OUTPUT_STORE_DIR = os.getenv("AGENT5_OUTPUT_DIR", "./data/agent5_outputs")

# Thresholds used by Agent 5's autonomous decision-making (Step 4/5 of
# the system prompt). Overridable per-deployment without touching code.
MISSING_VALUE_INTERPOLATE_MAX_FRACTION = float(
    os.getenv("AGENT5_MISSING_VALUE_INTERPOLATE_MAX_FRACTION", "0.30")
)  # above this fraction missing, interpolation is no longer scientifically
   # defensible for a single variable -- see agent5.py's autonomous
   # decision logic for how this is used (it does not by itself stop
   # the pipeline; STEP 2 validation decides that against required vars).

OUTLIER_ZSCORE_THRESHOLD = float(os.getenv("AGENT5_OUTLIER_ZSCORE_THRESHOLD", "4.0"))

# Minimum fraction of requested temporal/spatial coverage that must be
# actually retrievable for the objective to be considered achievable.
MIN_COVERAGE_FRACTION_FOR_OBJECTIVE = float(
    os.getenv("AGENT5_MIN_COVERAGE_FRACTION", "0.5")
)
