# Scientific Dataset Discovery Dashboard

Streamlit frontend for the Earth Intelligence multi-agent pipeline.

## Files created (do NOT touch the agents)

```
dashboard/
├── app.py                        ← entry point
├── requirements.txt
├── README.md
├── components/
│   ├── sidebar.py                ← logo, query input, settings, buttons
│   ├── pipeline.py               ← horizontal status bar
│   ├── logs.py                   ← live execution log
│   ├── metrics.py                ← system metric cards
│   ├── clarification.py          ← Agent 2 question/answer UI
│   ├── dataset_cards.py          ← Agent 1 panel, Agent 3 cards, table, detail
│   └── map.py                    ← PyDeck spatial coverage map
└── utils/
    ├── session_manager.py        ← session_state initialisation & reset
    └── styles.py                 ← global CSS injection
```

## Installation

Install dashboard-only dependencies (your agent dependencies are already installed):

```bash
pip install streamlit plotly pydeck pandas
```

## Running

`app.py` lives inside the `dashboard/` subfolder but needs the project root on
`sys.path` so it can import `agents/`, `models/`, `security/`, etc.
It adds `..` to `sys.path` automatically, so just run:

```bash
# From your project root:
streamlit run dashboard/app.py
```

Or from inside the dashboard/ folder:

```bash
streamlit run app.py
```

## What was NOT changed

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

Agent 4 shows a placeholder panel — wire it in once implemented.
