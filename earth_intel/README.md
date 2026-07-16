# Earth Intelligence Platform — Integrated Dashboard

Streamlit dashboard + FastAPI voice backend for the multi-agent scientific
dataset discovery system, with full voice/chat interface integrated.

---

## File Structure

```
earth_intelligence/
├── app.py                        ← FastAPI entry point (backend + voice API)
├── dashboard_app.py              ← Streamlit entry point (full dashboard)
├── requirements.txt
├── README.md
│
├── api/
│   └── routes.py                 ← /api/stt, /api/tts, /api/chat endpoints
│
├── frontend/
│   └── index.html                ← Standalone voice-only UI (served by FastAPI at /)
│
├── components/
│   ├── auth.py                   ← Login page, profile header, welcome modal
│   ├── clarification.py          ← Agent 2 clarification question UI
│   ├── dataset_cards.py          ← Agent 1 panel, Agent 3 cards, table, detail view
│   ├── logs.py                   ← Live execution log panel
│   ├── map.py                    ← PyDeck spatial coverage map
│   ├── metrics.py                ← System metric cards row
│   ├── pages.py                  ← Secondary page renderers
│   ├── pipeline.py               ← Horizontal agent pipeline status bar
│   ├── sidebar.py                ← Navigation, query input, status pill
│   └── voice.py                  ← 🆕 Voice/Chat panel (embedded HTML+JS component)
│
└── utils/
    ├── session_manager.py        ← Session state init/reset (incl. voice state)
    └── styles.py                 ← Global CSS injection
```

---

## What's New — Voice Integration

### `components/voice.py`
A self-contained Streamlit component that embeds the full voice/chat UI
(`frontend/index.html`) inline using `st.components.v1.html`.

**Features:**
- 🎤 **Microphone recording** via `MediaRecorder` API → sent to `/api/stt`
- 🌐 **Automatic language detection & translation** (Hindi, Telugu, etc. → English)
- 🔊 **Text-to-speech** on every agent response via `/api/tts`
- 📋 **Structured Agent 1 result card** (domain, intent, location, time, variables, goal)
- 💬 **Multi-turn conversation thread** with language badges
- ⌨️ `Ctrl+Enter` keyboard shortcut to submit

### `api/routes.py`
Three FastAPI endpoints:

| Endpoint    | Method | Purpose                                        |
|-------------|--------|------------------------------------------------|
| `/api/stt`  | POST   | Transcribe audio → English (Whisper + translate)|
| `/api/tts`  | POST   | Synthesise text → MP3 audio                    |
| `/api/chat` | POST   | Run Agent 1 on query, return structured output |

### Navigation
A **🎙 Voice Interface** page has been added to the sidebar between
"New Analysis" and "Previous Analyses". It renders the voice panel inside
the full authenticated dashboard with full access to agent pipeline controls.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running

You need **two processes** running simultaneously:

```bash
# Terminal 1 — FastAPI backend (voice API + static frontend)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit full dashboard
streamlit run dashboard_app.py --server.port 8501
```

Then open:
- **Full dashboard:** http://localhost:8501
- **Voice-only UI:**  http://localhost:8000

---

## Architecture

```
Browser (Streamlit : 8501)
  │
  ├─ Sidebar query → pipeline → Agent 1/2/3 (Python, in-process)
  │
  └─ Voice Interface page
       └─ st.components.v1.html  ← embedded voice panel
            │
            ├─ POST /api/stt  ─┐
            ├─ POST /api/chat ─┤→ FastAPI (:8000) → Agents
            └─ POST /api/tts  ─┘

Browser (Standalone : 8000)
  └─ frontend/index.html → same /api/* endpoints
```

---

## What Was NOT Changed

- `agents/agent1.py`
- `agents/agent2.py`
- `agents/agent3_discovery.py`
- `models/schemas.py`
- `models/discovery_schemas.py`
- `models/retrieval_request.py`
- `security/`
- `prompts/`
- `sources/`
- `discovery/`
- `main.py`
- `voice/stt.py`, `voice/tts.py`, `voice/language.py`, `voice/session.py`

Agent 4 shows a placeholder panel — wire it in once implemented.
