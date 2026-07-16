# Voice Layer Integration Guide

## 1. Copy the files into your project

```
your_project/
├── voice/                 ← copy entire folder here
│   ├── __init__.py
│   ├── stt.py
│   ├── language.py
│   ├── tts.py
│   └── session.py
│
├── api/
│   ├── __init__.py
│   └── routes.py          ← copy this (or merge into your existing routes file)
│
└── frontend/
    └── index.html         ← copy this; serve it as a static file (see step 3)
```

---

## 2. Install dependencies

```bash
pip install -r requirements_voice.txt
```

First run will download:
- Whisper `base` model (~150 MB, cached to `~/.cache/huggingface`)
- Helsinki-NLP opus-mt translation model for each new language (~300 MB each, on demand)

---

## 3. Register the router in your main FastAPI file

```python
# main.py (your existing FastAPI entrypoint)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.routes import router          # ← add this

app = FastAPI()
app.include_router(router)             # ← add this

# Serve the HTML frontend at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

---

## 4. Wire Agent 1 — the only real change you need to make

Open `api/routes.py` and find the `_call_agent1()` function at the bottom.
Replace the stub body with your actual Agent 1 call.

**Example A — Agent 1 is a class:**
```python
from agents.agent1 import Agent1
_agent = Agent1()

def _call_agent1(message: str, session_id: str) -> dict:
    response = _agent.process(message, session_id=session_id)
    return {"question": response.next_question, "done": response.complete}
```

**Example B — Agent 1 is a function:**
```python
from agents.agent1 import process_query

def _call_agent1(message: str, session_id: str) -> dict:
    result = process_query(message)
    return {"question": result["question"], "done": result["done"]}
```

**Example C — Agent 1 calls your Groq/LLaMA pipeline:**
```python
from agents.agent1 import run_agent1

def _call_agent1(message: str, session_id: str) -> dict:
    next_q = run_agent1(user_input=message, session=session_id)
    return {"question": next_q, "done": next_q is None}
```

The return dict shape is always:
```python
{"question": "<string or None>", "done": <bool>}
```

---

## 5. Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open: `http://localhost:8000`

---

## Endpoints summary

| Method | Path | What it does |
|--------|------|-------------|
| POST | `/api/stt` | Upload webm audio → get English transcript |
| POST | `/api/tts` | Send text → get mp3 audio |
| POST | `/api/chat` | Send English text → get Agent 1's next question |
| GET | `/` | Serves the HTML frontend |

---

## Notes

- **Agent 1 is never touched.** It receives a plain Python string exactly as it did from the CLI.
- **Translation is on-demand.** If your user speaks Hindi, the first Hindi voice turn downloads the `opus-mt-hi-en` model once, then caches it for the rest of the session.
- **TTS is opt-in per question.** Users click 🔊 to hear a question; nothing auto-plays.
- **Auto-send after STT is off by default.** The transcript populates the textarea so the user can review and edit before sending. To enable auto-send, uncomment the block in `index.html`.
- **Session IDs.** The frontend stores `sessionId` in JS memory. Refreshing the page starts a new session. For persistence across refreshes, store it in `localStorage` (add one line in `index.html`).
