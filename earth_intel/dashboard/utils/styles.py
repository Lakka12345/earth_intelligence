"""
utils/styles.py
Injects global CSS into the Streamlit page.
"""

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Defensive text-color override ─────────────────────────
           This stylesheet assumes a LIGHT page background (see
           [data-testid="stAppViewContainer"] below). If .streamlit/
           config.toml sets a dark theme (textColor near-white), plain
           Streamlit markdown/headings/captions would render white text
           on this light background and become invisible. Force a sane
           default here so that never happens, regardless of what the
           Streamlit theme config says. Specific components below that
           want a different color (e.g. text on the dark login hero)
           already set their own color and are unaffected. */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {
            color: #0f172a;
        }
        [data-testid="stCaptionContainer"] {
            color: #64748b;
        }

        /* ── Page base ─────────────────────────────────────────── */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 18%, rgba(20,184,166,0.14), transparent 28%),
                radial-gradient(circle at 86% 6%, rgba(37,99,235,0.13), transparent 30%),
                #f8fafc;
        }
        .block-container {
            padding-top: 1.4rem;
            max-width: 1500px;
        }

        /* Login */
        .login-hero {
            position: fixed;
            inset: 0;
            z-index: 0;
            background:
                linear-gradient(125deg, rgba(2,6,23,0.96), rgba(15,23,42,0.88)),
                url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=2200&q=80");
            background-size: cover;
            background-position: center;
        }
        .login-grid {
            position:absolute;
            inset:0;
            background-image:
                linear-gradient(rgba(148,163,184,0.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148,163,184,0.10) 1px, transparent 1px);
            background-size: 46px 46px;
            animation: gridDrift 18s linear infinite;
        }
        .login-brand {
            position:absolute;
            left:7vw;
            top:17vh;
            width:min(520px, 42vw);
            color:#e2e8f0;
        }
        .login-mark, .login-card-logo, .platform-mark, .sidebar-logo {
            display:grid;
            place-items:center;
            background: linear-gradient(135deg,#14b8a6,#2563eb);
            color:#ffffff;
            font-weight:900;
            letter-spacing:0;
        }
        .login-mark {
            width:76px;
            height:76px;
            border-radius:22px;
            font-size:24px;
            box-shadow:0 22px 80px rgba(20,184,166,0.35);
            margin-bottom:24px;
        }
        .login-kicker {
            color:#67e8f9;
            font-size:12px;
            font-weight:800;
            letter-spacing:0.12em;
            text-transform:uppercase;
        }
        .login-brand h1 {
            margin:8px 0;
            font-size:48px;
            line-height:1.03;
            letter-spacing:0;
        }
        .login-brand p {
            color:#cbd5e1;
            font-size:17px;
            line-height:1.7;
        }
        .login-chips {
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin-top:24px;
        }
        .login-chips span {
            border:1px solid rgba(226,232,240,0.24);
            color:#e2e8f0;
            border-radius:999px;
            padding:8px 12px;
            font-size:12px;
            background:rgba(15,23,42,0.46);
        }
        .login-card-title {
            position:relative;
            z-index:1;
            margin-top:12vh;
            background:rgba(255,255,255,0.94);
            border:1px solid rgba(226,232,240,0.8);
            border-bottom:0;
            border-radius:18px 18px 0 0;
            padding:22px 24px 10px;
            display:flex;
            gap:14px;
            align-items:center;
            box-shadow:0 24px 70px rgba(15,23,42,0.18);
        }
        .login-card-logo {
            width:46px;
            height:46px;
            border-radius:14px;
        }
        .login-card-title h2 {
            margin:0;
            color:#0f172a;
            font-size:24px;
            letter-spacing:0;
        }
        .login-card-title p {
            margin:2px 0 0;
            color:#64748b;
            font-size:13px;
        }
        .login-card-title + div,
        .login-card-title ~ div[data-testid="stForm"] {
            position:relative;
            z-index:1;
        }
        div[data-testid="stForm"] {
            background:rgba(255,255,255,0.94);
            border:1px solid rgba(226,232,240,0.8);
            border-top:0;
            border-radius:0 0 18px 18px;
            padding:10px 24px 22px;
            box-shadow:0 24px 70px rgba(15,23,42,0.18);
        }

        /* Header and profile */
        .platform-header {
            display:flex;
            align-items:center;
            gap:14px;
            padding:8px 0 18px;
        }
        .platform-mark {
            width:50px;
            height:50px;
            border-radius:16px;
        }
        .platform-header h1 {
            margin:0;
            color:#0f172a;
            font-size:28px;
            letter-spacing:0;
        }
        .platform-header p {
            margin:2px 0 0;
            color:#64748b;
            font-size:13px;
        }
        .profile-pill {
            display:flex;
            gap:10px;
            align-items:center;
            justify-content:flex-end;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:14px;
            padding:10px 12px;
            box-shadow:0 10px 30px rgba(15,23,42,0.06);
        }
        .profile-avatar, .sidebar-avatar {
            display:grid;
            place-items:center;
            border-radius:999px;
            background:#0f172a;
            color:#ffffff;
            font-weight:800;
            letter-spacing:0;
        }
        .profile-avatar { width:38px; height:38px; }
        .profile-copy {
            min-width:0;
            text-align:left;
        }
        .profile-copy strong,
        .profile-copy span {
            display:block;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
        }
        .profile-copy strong { color:#0f172a; font-size:13px; }
        .profile-copy span { color:#64748b; font-size:11px; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background:#ffffff;
            border-right:1px solid #e2e8f0;
        }
        .sidebar-brand {
            display:flex;
            gap:12px;
            align-items:center;
            padding:14px 0 10px;
        }
        .sidebar-logo {
            width:42px;
            height:42px;
            border-radius:13px;
        }
        .sidebar-brand strong,
        .sidebar-brand span,
        .sidebar-user strong,
        .sidebar-user span {
            display:block;
        }
        .sidebar-brand strong { color:#0f172a; font-size:15px; }
        .sidebar-brand span { color:#64748b; font-size:11px; }
        .sidebar-user {
            display:flex;
            gap:10px;
            align-items:center;
            background:#f8fafc;
            border:1px solid #e2e8f0;
            border-radius:12px;
            padding:10px;
            margin:8px 0 14px;
        }
        .sidebar-avatar { width:34px; height:34px; font-size:12px; }
        .sidebar-user strong { color:#0f172a; font-size:12px; }
        .sidebar-user span { color:#64748b; font-size:10px; }
        .nav-label {
            color:#64748b;
            font-size:11px;
            font-weight:800;
            text-transform:uppercase;
            letter-spacing:0.08em;
            margin:8px 0;
        }
        .status-pill {
            background:#ecfeff;
            color:#0e7490;
            border:1px solid #a5f3fc;
            border-radius:999px;
            padding:8px 12px;
            font-size:12px;
            font-weight:700;
            text-align:center;
        }

        /* Secondary pages */
        .page-intro, .welcome-panel {
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:16px;
            padding:22px 24px;
            margin:8px 0 18px;
            box-shadow:0 14px 40px rgba(15,23,42,0.05);
        }
        .page-intro h2, .welcome-panel h3 {
            margin:0 0 6px;
            color:#0f172a;
            letter-spacing:0;
        }
        .page-intro p {
            margin:0;
            color:#64748b;
        }
        .welcome-panel ol {
            margin-bottom:0;
            color:#334155;
        }
        .analysis-row {
            display:flex;
            justify-content:space-between;
            gap:18px;
            align-items:center;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:14px;
            padding:16px 18px;
            margin-bottom:10px;
        }
        .analysis-row strong, .analysis-row span {
            display:block;
        }
        .analysis-row strong { color:#0f172a; }
        .analysis-row span { color:#64748b; font-size:12px; }
        .analysis-stats {
            display:flex;
            gap:8px;
            flex-wrap:wrap;
        }
        .analysis-stats span {
            background:#f1f5f9;
            color:#334155;
            border-radius:999px;
            padding:6px 10px;
            font-weight:700;
        }
        .security-card {
            min-height:150px;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:14px;
            padding:16px;
            box-shadow:0 10px 28px rgba(15,23,42,0.05);
        }
        .security-card span {
            display:inline-block;
            background:#dcfce7;
            color:#15803d;
            border-radius:999px;
            padding:4px 8px;
            font-size:11px;
            font-weight:800;
            margin-bottom:12px;
        }
        .security-card strong {
            display:block;
            color:#0f172a;
            margin-bottom:6px;
        }
        .security-card p {
            color:#64748b;
            font-size:12px;
            margin:0;
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
            box-shadow: 0 8px 24px rgba(15,23,42,0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .pipeline-node:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(15,23,42,0.09);
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
        .dataset-card:hover { box-shadow: 0 12px 28px rgba(15,23,42,0.10); transform: translateY(-1px); }
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

        /* ── Dataset trust badges ───────────────────────────────── */
        /* Shown inside dataset cards ONLY when security DBs confirm status.
           .ds-trust-badges wraps the pills; .ds-trust-badge is each pill.
           To restyle all badges at once, edit this block only. */
        .ds-trust-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 8px 0 12px;
        }
        .ds-trust-badge {
            background: #f0fdf4;
            color: #15803d;
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        /* ── Security Reports page — dark card variant ──────────── */
        /* These styles scope the dark aesthetic used in pages.py
           security section cards. They do NOT affect any other page.
           Class .sec-card is applied exclusively inside render_security_reports(). */
        .sec-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 6px;
        }
        /* Override the global text-color rule for dark-card children only */
        .sec-card p,
        .sec-card span:not(.ds-trust-badge),
        .sec-card label {
            color: #cbd5e1;
        }

        /* ── Pipeline page improvements ─────────────────────────── */
        /* Augments the existing .pipeline-node classes with richer state
           feedback. Existing class rules in .pipeline-node.* are preserved;
           these only add what was previously missing. */
        .pipeline-node.running {
            animation: nodePulse 2s ease-in-out infinite;
        }
        @keyframes nodePulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
            50%       { box-shadow: 0 0 0 6px rgba(59,130,246,0.0); }
        }
        .pipeline-node.completed {
            background: #f0fdf4;
            border-color: #22c55e;
        }
        .pipeline-node.failed {
            animation: nodeShake 0.4s ease;
        }
        @keyframes nodeShake {
            0%, 100% { transform: translateX(0); }
            25%       { transform: translateX(-3px); }
            75%       { transform: translateX(3px); }
        }

        /* ── Metric cards — accent top border support ───────────── */
        /* metrics.py uses inline border-top per card; this ensures the
           base card still picks up the shared radius and shadow. */
        .metric-card {
            transition: box-shadow 0.15s;
        }
        .metric-card:hover {
            box-shadow: 0 10px 28px rgba(15,23,42,0.08);
        }

        /* ── Score breakdown panel inside dataset cards ─────────── */
        .score-breakdown-panel {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 14px;
        }
        .score-breakdown-title {
            font-size: 11px;
            font-weight: 800;
            color: #64748b;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        /* ── Hide Streamlit branding ────────────────────────────── */
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }

        @keyframes gridDrift {
            from { background-position: 0 0, 0 0; }
            to { background-position: 46px 46px, 46px 46px; }
        }
        /* ── Voice Interface nav badge ── */
        .voice-nav-badge {
            display:inline-block;
            background:linear-gradient(135deg,#14b8a6,#2563eb);
            color:#fff;
            font-size:9px;
            font-weight:800;
            letter-spacing:0.06em;
            text-transform:uppercase;
            padding:1px 5px;
            border-radius:4px;
            margin-left:6px;
            vertical-align:middle;
        }

        @media (max-width: 900px) {
            .login-brand {
                position:relative;
                left:auto;
                top:auto;
                width:auto;
                padding:34px 24px 0;
            }
            .login-brand h1 { font-size:34px; }
            .login-card-title { margin-top:20px; }
            .platform-header h1 { font-size:22px; }
            .profile-pill { justify-content:flex-start; }
            .analysis-row { align-items:flex-start; flex-direction:column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
