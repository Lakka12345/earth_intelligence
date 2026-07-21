"""Frontend-only authentication shell for the Streamlit dashboard."""

from datetime import datetime
import re
import time

import streamlit as st


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _display_name(email: str) -> str:
    local = email.split("@", 1)[0].replace(".", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in local.split()) or "Research User"


PLATFORM_NAME = "Design and Development of Multi Agent AI Framework for Data Discovery and Retrieval"
PLATFORM_BADGE = "DMAF"


def _initials(name: str) -> str:
    value = "".join(part[0].upper() for part in name.split()[:2] if part)
    return value or PLATFORM_BADGE


def _login(email: str, remember: bool, guest: bool = False) -> None:
    name = "Guest Analyst" if guest else _display_name(email)
    st.session_state.is_authenticated = True
    st.session_state.remember_me = remember
    st.session_state.last_login = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.session_state.auth_user = {
        "name": name,
        "email": email,
        "avatar": _initials(name),
        "role": "Guest Session" if guest else "Research Analyst",
    }
    if not st.session_state.get("welcome_dismissed", False):
        st.session_state.show_welcome = True


def _register(name: str, email: str, password: str) -> None:
    """Frontend-only sign-up: creates the session the same way login does."""
    st.session_state.is_authenticated = True
    st.session_state.remember_me = False
    st.session_state.last_login = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.session_state.auth_user = {
        "name": name.strip() or _display_name(email),
        "email": email,
        "avatar": _initials(name.strip() or _display_name(email)),
        "role": "Research Analyst",
    }
    if not st.session_state.get("welcome_dismissed", False):
        st.session_state.show_welcome = True


def logout_user() -> None:
    st.session_state.is_authenticated = False
    st.session_state.auth_user = None
    st.session_state.show_welcome = False
    st.session_state.active_page = "Dashboard"


def render_login_page() -> None:
    st.markdown(
        """
        <style>
        /* ── Page reset & dark background ───────────────────────── */
        [data-testid="stAppViewContainer"] {
            background: #0d1521 !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* ── Grid overlay ────────────────────────────────────────── */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(59,130,246,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(59,130,246,0.06) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }

        /* ── Centered glow ───────────────────────────────────────── */
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: fixed;
            width: 700px;
            height: 700px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(37,99,235,0.13) 0%, transparent 70%);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 0;
        }

        /* ── Brand block (above card) ────────────────────────────── */
        .dmaf-brand {
            text-align: center;
            margin-bottom: 28px;
        }
        .dmaf-badge-mark {
            display: inline-block;
            background: #2563eb;
            color: #fff;
            font-size: 13px;
            font-weight: 700;
            padding: 7px 14px;
            border-radius: 10px;
            letter-spacing: 0.5px;
            margin-bottom: 14px;
        }
        .dmaf-brand h1 {
            color: #e2e8f0 !important;
            font-size: 18px !important;
            font-weight: 500 !important;
            max-width: 460px;
            line-height: 1.5 !important;
            margin: 0 auto 6px !important;
        }
        .dmaf-brand p {
            color: #64748b;
            font-size: 13px;
            margin: 0 !important;
        }
        .dmaf-chips {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 14px;
            flex-wrap: wrap;
        }
        .dmaf-chips span {
            border: 0.5px solid rgba(96,165,250,0.25);
            color: #7dd3fc;
            font-size: 11px;
            padding: 4px 12px;
            border-radius: 100px;
            background: rgba(37,99,235,0.08);
        }

        /* ── Login card ──────────────────────────────────────────── */
        .dmaf-card {
            background: rgba(15, 23, 36, 0.85);
            border: 0.5px solid rgba(255,255,255,0.09);
            border-radius: 16px;
            padding: 36px 40px;
            max-width: 440px;
            margin: 0 auto;
        }
        .dmaf-card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        .dmaf-card-mark {
            background: #2563eb;
            color: #fff;
            font-size: 11px;
            font-weight: 700;
            padding: 7px 10px;
            border-radius: 8px;
            flex-shrink: 0;
        }
        .dmaf-card-header h2 {
            font-size: 18px !important;
            font-weight: 500 !important;
            color: #f1f5f9 !important;
            margin: 0 !important;
        }
        .dmaf-card-header p {
            font-size: 12px;
            color: #64748b;
            margin: 2px 0 0 !important;
        }

        /* ── Streamlit widget overrides inside card ──────────────── */
        .dmaf-card .stTextInput input,
        .dmaf-card .stTextInput > div > div > input {
            background: rgba(255,255,255,0.04) !important;
            border: 0.5px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
        }
        .dmaf-card label {
            color: #94a3b8 !important;
            font-size: 12px !important;
        }
        .dmaf-card .stCheckbox label {
            color: #94a3b8 !important;
            font-size: 12px !important;
        }
        .dmaf-card .stFormSubmitButton button,
        .dmaf-card .stButton button[kind="primary"] {
            background: #2563eb !important;
            border: none !important;
            border-radius: 8px !important;
            color: #fff !important;
            font-weight: 500 !important;
        }
        .dmaf-card .stFormSubmitButton button:hover,
        .dmaf-card .stButton button[kind="primary"]:hover {
            background: #1d4ed8 !important;
        }
        .dmaf-card .stButton button[kind="secondary"] {
            background: transparent !important;
            border: 0.5px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
            color: #94a3b8 !important;
        }

        /* ── Signup row text ─────────────────────────────────────── */
        .dmaf-signup-row {
            text-align: center;
            font-size: 12px;
            color: #64748b;
            margin-top: 8px;
        }
        </style>

        <div class="dmaf-brand">
            <div class="dmaf-badge-mark">DMAF</div>
            <h1>Design and Development of Multi-Agent AI Framework<br>for Data Discovery and Retrieval</h1>
            <p>Secure Multi-Agent Scientific Intelligence System</p>
            <div class="dmaf-chips">
                <span>Planner Agent</span>
                <span>Dataset Discovery</span>
                <span>Security Validation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    is_signup = st.session_state.auth_mode == "signup"

    st.markdown(
        f"""
        <div class="dmaf-card">
            <div class="dmaf-card-header">
                <div class="dmaf-card-mark">DMAF</div>
                <div>
                    <h2>{"Create your account" if is_signup else "Welcome back"}</h2>
                    <p>{"Sign up to get started with your dashboard." if is_signup else "Sign in to continue to your dashboard."}</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([0.5, 3, 0.5])
    with center:
        if is_signup:
            _render_signup_form()
        else:
            _render_login_form()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_login_form() -> None:
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="researcher@example.com")
        show_password = st.checkbox("Show password")
        password = st.text_input(
            "Password",
            placeholder="Enter your password",
            type="default" if show_password else "password",
        )
        remember = st.checkbox("Remember Me", value=st.session_state.get("remember_me", False))
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    left, right = st.columns(2)
    with left:
        forgot = st.button("Forgot Password", use_container_width=True)
    with right:
        guest = st.button("Continue as Guest", use_container_width=True)

    if forgot:
        st.info("Password recovery is ready for backend integration.")

    if submitted:
        errors = []
        if not EMAIL_RE.match(email.strip()):
            errors.append("Enter a valid email address.")
        if len(password or "") < 6:
            errors.append("Password must be at least 6 characters.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Signing in..."):
                time.sleep(0.6)
            _login(email.strip(), remember)
            st.success("Login successful. Opening dashboard...")
            time.sleep(0.25)
            st.rerun()

    if guest:
        with st.spinner("Preparing guest workspace..."):
            time.sleep(0.4)
        _login("guest@earth-intelligence.local", False, guest=True)
        st.success("Guest session started.")
        time.sleep(0.25)
        st.rerun()

    st.markdown(
        """
        <div class="login-signup-row">
            Don't have an account?
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Create an account", use_container_width=True, key="go_to_signup"):
        st.session_state.auth_mode = "signup"
        st.rerun()


def _render_signup_form() -> None:
    with st.form("signup_form", clear_on_submit=False):
        full_name = st.text_input("Full name", placeholder="Jane Researcher")
        email = st.text_input("Email", placeholder="researcher@example.com", key="signup_email")
        show_password = st.checkbox("Show password", key="signup_show_password")
        password = st.text_input(
            "Password",
            placeholder="Create a password",
            type="default" if show_password else "password",
            key="signup_password",
        )
        confirm_password = st.text_input(
            "Confirm password",
            placeholder="Re-enter your password",
            type="default" if show_password else "password",
            key="signup_confirm_password",
        )
        agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not full_name.strip():
            errors.append("Enter your full name.")
        if not EMAIL_RE.match(email.strip()):
            errors.append("Enter a valid email address.")
        if len(password or "") < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if not agree:
            errors.append("You must agree to the Terms of Service and Privacy Policy.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Creating your account..."):
                time.sleep(0.6)
            _register(full_name, email.strip(), password)
            st.success("Account created. Opening dashboard...")
            time.sleep(0.25)
            st.rerun()

    st.markdown(
        """
        <div class="login-signup-row">
            Already have an account?
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Back to Login", use_container_width=True, key="go_to_login"):
        st.session_state.auth_mode = "login"
        st.rerun()


def render_profile_header() -> None:
    user = st.session_state.get("auth_user") or {}
    name = user.get("name", "Research User")
    email = user.get("email", "user@example.com")
    avatar = user.get("avatar", PLATFORM_BADGE)
    last_login = st.session_state.get("last_login") or "Current session"

    left, right = st.columns([4.8, 2])
    with left:
        st.markdown(
            f"""
            <div class="platform-header">
                <div class="platform-mark">{PLATFORM_BADGE}</div>
                <div>
                    <h1>{PLATFORM_NAME}</h1>
                    <p>Enterprise multi-agent scientific intelligence workspace</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="profile-pill">
                <div class="profile-avatar">{avatar}</div>
                <div class="profile-copy">
                    <strong>{name}</strong>
                    <span>{email}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Profile Menu", expanded=False):
            st.caption(f"Role: {user.get('role', 'Research Analyst')}")
            st.caption(f"Last Login: {last_login}")
            settings_col, logout_col = st.columns(2)
            with settings_col:
                if st.button("Settings", use_container_width=True):
                    st.session_state.active_page = "Settings"
                    st.rerun()
            with logout_col:
                if st.button("Logout", use_container_width=True):
                    logout_user()
                    st.rerun()


def render_welcome_modal() -> None:
    if not st.session_state.get("show_welcome", False):
        return
    st.markdown(
        f"""
        <div class="welcome-panel">
            <h3>Welcome to {PLATFORM_BADGE}</h3>
            <ol>
                <li>Enter your scientific query.</li>
                <li>The Planner Agent creates a research strategy.</li>
                <li>The Clarification Agent refines the request.</li>
                <li>Dataset Discovery locates relevant scientific data.</li>
                <li>Security Validation verifies integrity.</li>
                <li>The Final Intelligence Report is generated.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    dont_show = st.checkbox("Don't show again", key="dont_show_welcome")
    if st.button("Start Analysis", type="primary"):
        st.session_state.show_welcome = False
        if dont_show:
            st.session_state.welcome_dismissed = True
        st.rerun()
