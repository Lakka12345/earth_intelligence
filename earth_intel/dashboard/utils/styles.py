"""
utils/styles.py
Injects global CSS into the Streamlit page.
"""

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Page base ─────────────────────────────────────────── */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* ── Metric cards ───────────────────────────────────────── */
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
        }
        .metric-card .label {
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-card .value {
            font-size: 28px;
            font-weight: 700;
            color: #1e293b;
        }

        /* ── Status badges ──────────────────────────────────────── */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-waiting   { background:#f1f5f9; color:#64748b; }
        .badge-running   { background:#dbeafe; color:#1d4ed8; }
        .badge-completed { background:#dcfce7; color:#15803d; }
        .badge-failed    { background:#fee2e2; color:#b91c1c; }

        /* ── Pipeline node ──────────────────────────────────────── */
        .pipeline-node {
            border-radius: 12px;
            padding: 14px 18px;
            text-align: center;
            font-size: 13px;
            font-weight: 600;
            border: 2px solid transparent;
        }
        .pipeline-node.waiting   { background:#f8fafc; border-color:#e2e8f0; color:#94a3b8; }
        .pipeline-node.running   { background:#eff6ff; border-color:#3b82f6; color:#1d4ed8; }
        .pipeline-node.completed { background:#f0fdf4; border-color:#22c55e; color:#15803d; }
        .pipeline-node.failed    { background:#fff1f2; border-color:#f43f5e; color:#be123c; }

        /* ── Dataset card ───────────────────────────────────────── */
        .dataset-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 12px;
            transition: box-shadow 0.15s;
        }
        .dataset-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
        .dataset-card .ds-name  { font-size:16px; font-weight:700; color:#1e293b; }
        .dataset-card .ds-prov  { font-size:13px; color:#64748b; margin-top:2px; }
        .dataset-card .ds-score {
            font-size:20px; font-weight:800; color:#2563eb;
        }
        .score-bar-bg {
            background:#e2e8f0; border-radius:4px; height:6px; margin-top:6px;
        }
        .score-bar-fill {
            height:6px; border-radius:4px; background: linear-gradient(90deg,#3b82f6,#06b6d4);
        }

        /* ── Log panel ──────────────────────────────────────────── */
        .log-entry {
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 12px;
            color: #334155;
            padding: 3px 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .log-ts { color:#94a3b8; margin-right:8px; }

        /* ── Section headers ────────────────────────────────────── */
        .section-header {
            font-size: 15px;
            font-weight: 700;
            color: #1e293b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 12px;
        }

        /* ── Reasoning item ─────────────────────────────────────── */
        .reasoning-item {
            background: #fefce8;
            border-left: 3px solid #eab308;
            padding: 8px 12px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 6px;
            font-size: 13px;
            color: #713f12;
        }

        /* ── Auth badge ─────────────────────────────────────────── */
        .auth-yes { color:#b45309; background:#fef3c7; padding:2px 8px; border-radius:8px; font-size:12px; font-weight:600; }
        .auth-no  { color:#15803d; background:#dcfce7; padding:2px 8px; border-radius:8px; font-size:12px; font-weight:600; }

        /* ── Hide Streamlit branding ────────────────────────────── */
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
