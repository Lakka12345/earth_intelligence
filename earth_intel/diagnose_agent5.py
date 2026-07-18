"""
Run this from your project root (same folder as main.py):

    python diagnose_agent5.py

It imports every piece of the agent5 chain one at a time and tells you
EXACTLY which one fails and why, instead of the generic ImportError
main.py currently swallows.
"""
import sys
import os
import traceback

print(f"Running from: {os.getcwd()}")
print(f"sys.path[0]:  {sys.path[0]}")
print("-" * 70)

checks = [
    ("os", "import os"),
    ("numpy", "import numpy"),
    ("pandas", "import pandas"),
    ("xarray", "import xarray"),
    ("groq (pip package)", "from groq import Groq"),
    ("pint (used by agent5_units)", "import pint"),
    ("python-dotenv", "from dotenv import load_dotenv"),
    ("agent5_config (root)", "import agent5_config"),
    ("agent5_units (root)", "import agent5_units"),
    ("agent5_analysis (root)", "import agent5_analysis"),
    ("agent5_preprocessing_extra (root)", "import agent5_preprocessing_extra"),
    ("models package __init__", "import models"),
    ("models.discovery_schemas", "from models import discovery_schemas"),
    ("models.retrieval_request", "from models import retrieval_request"),
    ("models.agent5_schemas", "from models import agent5_schemas"),
    ("prompts package __init__", "import prompts"),
    ("prompts.agent5_prompt", "from prompts import agent5_prompt"),
    ("storage package __init__", "import storage"),
    ("storage.zarr_store", "from storage import zarr_store"),
    ("agents package __init__", "import agents"),
    ("agents.agent5 (the real target)", "from agents.agent5 import run_agent5"),
]

failures = []
for label, stmt in checks:
    try:
        exec(stmt)
        print(f"[OK]   {label}")
    except Exception as e:
        print(f"[FAIL] {label}  ->  {type(e).__name__}: {e}")
        failures.append((label, stmt))

print("-" * 70)
if not failures:
    print("Everything imported cleanly. If main.py still says 'not found',")
    print("main.py is not being run from this same directory/interpreter.")
else:
    print(f"{len(failures)} failure(s) found. Full traceback of the FIRST one:\n")
    label, stmt = failures[0]
    try:
        exec(stmt)
    except Exception:
        traceback.print_exc()
